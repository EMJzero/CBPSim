#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>
#include <vector>
#include <deque>
#include <map>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <algorithm>

/*
IDEA: handle usefulness by selecting the alt_predictor as the second most likely table chosen by the RL policy!
*/

// Configuration defines (values based on L-TAGE)
#define NUM_TAGGED_TABLES 12  // 12 TAGE tables
#define BASE_INDEX_BITS 14    // 16K entries for the base predictor (2^14)
#define BASE_PRED_BITS 2      // 2-bit counters for the base predictor [max is 8]
#define MAX_HISTORY 640       // 640 global history bits
#define MAX_HISTORY_BUFFER 32 // 32 extra global history bits updates after some predictions
#define TAGE_PRED_BITS 3      // 3-bit signed predictor (-4 to 3) [max is 8]
#define TAGE_USEFUL_BITS 2    // 2-bit usefulness [max is 8]
#define USE_ALT_THRESHOLD 8   // 4-bit counter, threshold 8 (mid-point)
#define RL_WEIGHTS_BITS 4     // 4-bit RL weights to pick the table
#define LEARNING_RATE 1       // RL learning rate [>= 1]

#define MAX_BYTES 192 * 1024  // 192KB of memory budget

static constexpr int8_t WEIGHT_MAX = (1 << (RL_WEIGHTS_BITS - 1)) - 1;
static constexpr int8_t WEIGHT_MIN = -(1 << (RL_WEIGHTS_BITS - 1));

// Tagged Table Configuration
struct TableConfig {
    int hist_len;   // History length L(i)
    int index_bits; // log2(number of entries)
    int tag_bits;   // Number of tag bits
    int entries() const { return 1 << index_bits; }
};

// Configure each tagged table (adjust based on your needs)
const TableConfig TAGGED_CONFIGS[NUM_TAGGED_TABLES] = {
    {4, 10, 7},  // T1: 4 history bits, 2^10 entries, 7-bit tag
    {6, 10, 7},  // T2
    {10, 11, 8}, // T3
    {16, 11, 8}, // T4
    {25, 11, 9}, // T5
    {40, 11, 10},// T6
    {64, 10, 11},// T7
    {101,10, 12},// T8
    {160,10, 12},// T9
    {254,9, 13}, // T10
    {403,9, 14}, // T11
    {640,9, 15}  // T12
};

class SampleCondPredictor {
private:
    // Cyclic counter of prediction
    uint8_t pred_cycle;

    // Base predictor: BASE_PRED_BITS-bit counters
    std::vector<int8_t> base_table;

    // RL weights
    // The last weight is the bias!
    std::vector<std::vector<int8_t>> weights;

    // Tagged tables
    struct TaggedEntry {
        uint16_t tag;
        int8_t ctr; // TAGE_PRED_BITS-bit signed (-4 to 3)
        uint8_t u;  // TAGE_USEFUL_BITS-bit usefulness
    };
    std::vector<std::vector<TaggedEntry>> tagged_tables;

    // Global History Register (GHR)
    std::deque<bool> ghr;

    // Alternate prediction counter
    uint8_t use_alt_on_na;

    // Reset counter for usefulness
    uint32_t reset_counter;
    static constexpr uint32_t RESET_PERIOD = 512 * 1024; // From L-TAGE

    // Speculative state tracking
    struct SpeculativeState {
        int provider_table;
        bool altpred;
        bool final_pred;
        uint8_t pred_cycle;
        std::vector<int32_t> prediction;
    };
    std::map<uint64_t, SpeculativeState> speculative_states;

    // Helper functions
    uint64_t compute_hash(uint64_t pc, const std::deque<bool>& hist, int hist_len, int skip_hist, int out_bits) {
        uint64_t hash = pc;
        int bits_used = 0;
        for (auto it = hist.rbegin() + skip_hist; it != hist.rend() && bits_used < hist_len; ++it) {
            hash ^= (*it << (bits_used % out_bits));
            bits_used++;
        }
        return hash & ((1ULL << out_bits) - 1);
    }

    // Get indices sorted by descending values in the array
    template <typename T>
    std::vector<size_t> get_sorted_indices(const std::vector<T>& arr) {
        std::vector<size_t> indices(arr.size());
        for (size_t i = 0; i < arr.size(); ++i)
            indices[i] = i;

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return arr[a] > arr[b]; // descending order
        });

        return indices;
    }

    // Estimate memory usage and check against budget
    void check_memory_budget() {
        size_t size = MAX_HISTORY + MAX_HISTORY_BUFFER; //GHR
        size += (1 << BASE_INDEX_BITS)*BASE_PRED_BITS; // base predictor
        size += RL_WEIGHTS_BITS*(MAX_HISTORY + 1)*NUM_TAGGED_TABLES; // RL weights
        for (const auto& cfg : TAGGED_CONFIGS) {
            size += cfg.entries()*(cfg.tag_bits + TAGE_PRED_BITS + TAGE_USEFUL_BITS); // tables
        }
        size = (size >> 3) + (size & 7 > 0 ? 1 : 0); // bits -> bytes
        std::cout << "Memory used: " << size << "B / " << MAX_BYTES << "B" << std::endl;
        assert(size <= MAX_BYTES && "Predictor exceeds MAX_BYTES memory limit! Reduce HISTORY_LENGTH or MAX_TABLE_ENTRIES.");
    }

public:
    SampleCondPredictor() {
        use_alt_on_na = USE_ALT_THRESHOLD;
        reset_counter = 0;
        pred_cycle = 0;

        // Initialize base table
        base_table.resize(1 << BASE_INDEX_BITS, 0); // Initial state: weakly taken

        // Initialize RL weights
        weights.resize(NUM_TAGGED_TABLES);
        for (int i = 0; i < NUM_TAGGED_TABLES; i++) {
            weights[i].resize(MAX_HISTORY + 1, 0);
        }

        // Initialize tagged tables
        for (const auto& cfg : TAGGED_CONFIGS) {
            tagged_tables.emplace_back(cfg.entries(), TaggedEntry{0, 0, 0});
        }
    }

    void setup() {
        check_memory_budget();
    }

    void terminate() {
    }

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        return (seq_no << 4) | (piece & 0xF);
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, bool tage_pred) {
        (void)tage_pred;

        uint64_t id = get_unique_inst_id(seq_no, piece);
        SpeculativeState state;

        // Base prediction
        uint64_t base_idx = PC % base_table.size();
        bool base_pred = (base_table[base_idx] >= 0);

        // Find provider and altpred
        state.provider_table = -1;
        state.altpred = base_pred;
        state.final_pred = base_pred;

        // RL probabilities for each table to be the predictor (second most likely is the alt_predictor)
        // TODO: if we had floats, a softmax here would be neat...
        state.prediction = std::vector<int32_t>(NUM_TAGGED_TABLES, 0);
        for (int i = 0; i < NUM_TAGGED_TABLES; i++) {
            for (int j = 0; j < MAX_HISTORY && j < ghr.size(); j++) {
                state.prediction[i] += ghr[j] ? weights[i][j] : -weights[i][j];
            }
            state.prediction[i] += weights[i][weights[i].size() - 1];
        }
        auto top_pred_idxs = get_sorted_indices(state.prediction);
        // Once you have all table probabilities, go from most likely to less likely in order to pick the first table with a matching tag
        for (auto& i : top_pred_idxs) {
            const auto& cfg = TAGGED_CONFIGS[i];
            auto& table = tagged_tables[i];
            uint64_t idx = compute_hash(PC, ghr, cfg.hist_len, 0, cfg.index_bits) % table.size();
            uint16_t tag = compute_hash(PC, ghr, cfg.hist_len, 0, cfg.tag_bits);

            if (state.provider_table == -1) {
                if (table[idx].tag == tag) {
                    state.provider_table = i;
                    state.final_pred = (table[idx].ctr >= 0);
                }
            } else {
                if (table[idx].tag == tag) {
                    state.altpred = (table[idx].ctr >= 0);
                    break;
                }
            }
        }

        // Use altpred if weak and use_alt_on_na
        if (state.provider_table != -1) {
            auto& entry = tagged_tables[state.provider_table][
                compute_hash(PC, ghr, TAGGED_CONFIGS[state.provider_table].hist_len, 0,
                             TAGGED_CONFIGS[state.provider_table].index_bits) %
                tagged_tables[state.provider_table].size()];
            if (abs(entry.ctr) <= 1 && use_alt_on_na >= USE_ALT_THRESHOLD) {
                state.final_pred = state.altpred;
            }
        }

        // Save state
        speculative_states[id] = state;

        return state.final_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        (void)PC; (void)nextPC;
        pred_cycle += 1;
        
        // Update GHR
        ghr.push_back(taken);
        if (ghr.size() > MAX_HISTORY + MAX_HISTORY_BUFFER) ghr.pop_front();

        // Save current cycle to rollback on misprediction
        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& state = speculative_states[id];
        state.pred_cycle = pred_cycle;
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        (void)nextPC;

        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& state = speculative_states[id];

        // Rollback GHR on misprediction
        uint8_t delta_cycles = pred_cycle - state.pred_cycle;
        if (predDir != resolveDir) {
            ghr[ghr.size() - delta_cycles - 1] = resolveDir;
        }

        // Update usefulness and counters
        if (state.provider_table != -1) {
            auto& cfg = TAGGED_CONFIGS[state.provider_table];
            auto& table = tagged_tables[state.provider_table];
            uint64_t idx = compute_hash(PC, ghr, cfg.hist_len, delta_cycles, cfg.index_bits) % table.size();
            TaggedEntry& entry = table[idx];

            // Update prediction counter
            if (resolveDir) {
                entry.ctr = std::min(entry.ctr + 1, (1 << (TAGE_PRED_BITS - 1)) - 1);
            } else {
                entry.ctr = std::max(entry.ctr - 1, (-1 << (TAGE_PRED_BITS - 1)));
            }

            // Update usefulness counter
            if (state.altpred != predDir) {
                if (resolveDir == predDir) {
                    entry.u = std::min(entry.u + 1, (1 << TAGE_USEFUL_BITS) - 1);
                } else {
                    entry.u = std::max(entry.u - 1, 0);
                }
            }

            // Allocation on misprediction
            if (!resolveDir == predDir) {
                // Dumb allocation: find first table with u = 0
                int k;
                for (k = state.provider_table + 1; k < NUM_TAGGED_TABLES; ++k) {
                    auto& alloc_cfg = TAGGED_CONFIGS[k];
                    auto& alloc_table = tagged_tables[k];
                    uint64_t alloc_idx = compute_hash(PC, ghr, alloc_cfg.hist_len, delta_cycles, alloc_cfg.index_bits) % alloc_table.size();
                    if (alloc_table[alloc_idx].u == 0) {
                        alloc_table[alloc_idx].tag = compute_hash(PC, ghr, alloc_cfg.hist_len, delta_cycles, alloc_cfg.tag_bits);
                        alloc_table[alloc_idx].ctr = (resolveDir ? 0 : -1); // Weak correct
                        alloc_table[alloc_idx].u = 0;
                        break;
                    }
                }
                
                // If there is enough history to do the update safely
                if (ghr.size() >= delta_cycles + MAX_HISTORY + 1) {
                    // Update RL weights
                    for (int i = 0; i < NUM_TAGGED_TABLES; i++) {
                        bool target = i == k;
                        for (uint j = 0; j < MAX_HISTORY && j < ghr.size(); j++) {
                            int8_t update = (ghr[ghr.size() - delta_cycles - j - 1] ? 1 : -1) * target * LEARNING_RATE;
                            weights[i][j] = std::max(std::min((int8_t)(weights[i][j] + update), WEIGHT_MAX), WEIGHT_MIN);
                        }
                        weights[i][weights[i].size() - 1] = std::max(std::min((int8_t)(weights[i][weights[i].size() - 1] + LEARNING_RATE * target), WEIGHT_MAX), WEIGHT_MIN);
                    }
                }
            }
        } else {
            // Update base table
            uint64_t base_idx = PC % base_table.size();
            if (resolveDir) {
                base_table[base_idx] = std::min(base_table[base_idx] + 1, (1 << (BASE_PRED_BITS - 1)) - 1);
            } else {
                base_table[base_idx] = std::max(base_table[base_idx] - 1, (-1 << (BASE_PRED_BITS - 1)));
            }
        }

        // Update use_alt_on_na
        if (state.provider_table != -1) {
            if ((state.altpred == resolveDir) && (predDir != resolveDir)) {
                use_alt_on_na = std::min(use_alt_on_na + 1, 15);
            } else if ((state.altpred != resolveDir) && (predDir == resolveDir)) {
                use_alt_on_na = std::max(use_alt_on_na - 1, 0);
            }
        }

        // Periodically reset usefulness counters
        if (++reset_counter >= RESET_PERIOD) {
            reset_counter = 0;
            for (auto& table : tagged_tables) {
                for (auto& entry : table) {
                    entry.u >>= 1; // Dumb reset
                }
            }
        }

        speculative_states.erase(id);
    }
};

// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
