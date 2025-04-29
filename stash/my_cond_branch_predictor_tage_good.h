#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <vector>
#include <deque>
#include <map>
#include <cstdint>
#include <cmath>

// Configuration defines (values based on L-TAGE)
#define NUM_TAGGED_TABLES 12 // 12 TAGE tables
#define BASE_INDEX_BITS 14   // 16K entries for the base predictor (2^14)
#define BASE_PRED_BITS 2     // 2-bit counters for the base predictor [max is 8]
#define MAX_HISTORY 640      // 640 global history bits
#define TAGE_PRED_BITS 3     // 3-bit signed predictor (-4 to 3) [max is 8]
#define TAGE_USEFUL_BITS 2   // 2-bit usefulness [max is 8]
#define USE_ALT_THRESHOLD 8  // 4-bit counter, threshold 8 (mid-point)
#define MAX_BYTES 192 * 1024 // 192KB of memory budget

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
    // Base predictor: BASE_PRED_BITS-bit counters
    std::vector<int8_t> base_table;

    // Tagged tables
    struct TaggedEntry {
        uint16_t tag;
        int8_t ctr; // TAGE_PRED_BITS-bit signed (-4 to 3)
        uint8_t u;  // TAGE_USEFUL_BITS-bit usefulness
    };
    std::vector<std::vector<TaggedEntry>> tagged_tables;

    // Global History Register (GHR)
    std::deque<bool> ghr;
    int max_hist;

    // Alternate prediction counter
    uint8_t use_alt_on_na;

    // Reset counter for usefulness
    uint32_t reset_counter;
    static constexpr uint32_t RESET_PERIOD = 512 * 1024; // From L-TAGE

    // Speculative state tracking
    struct SpeculativeState {
        std::deque<bool> ghr_snapshot;
        int provider_table;
        bool altpred;
        bool final_pred;
    };
    std::map<uint64_t, SpeculativeState> speculative_states;

    // Helper functions
    uint64_t compute_hash(uint64_t pc, const std::deque<bool>& hist, int hist_len, int out_bits) {
        uint64_t hash = pc;
        int bits_used = 0;
        for (auto it = hist.rbegin(); it != hist.rend() && bits_used < hist_len; ++it) {
            hash ^= (*it << (bits_used % out_bits));
            bits_used++;
        }
        return hash & ((1ULL << out_bits) - 1);
    }

    // Estimate memory usage and check against budget
    void check_memory_budget() {
        size_t size = MAX_HISTORY; //GHR
        size += (1 << BASE_INDEX_BITS)*BASE_PRED_BITS; // base predictor
        for (const auto& cfg : TAGGED_CONFIGS) {
            size += cfg.entries()*(cfg.tag_bits + TAGE_PRED_BITS + TAGE_USEFUL_BITS); // tables
        }
        size = (size >> 3) + (size & 7 > 0 ? 1 : 0); // bits -> bytes
        std::cout << "Memory used: " << size << "B / " << MAX_BYTES << "B" << std::endl;
        assert(size <= MAX_BYTES && "Predictor exceeds MAX_BYTES memory limit! Reduce HISTORY_LENGTH or MAX_TABLE_ENTRIES.");
    }

public:
    SampleCondPredictor() : max_hist(MAX_HISTORY), use_alt_on_na(8), reset_counter(0) {
        // Initialize base table
        base_table.resize(1 << BASE_INDEX_BITS, 0); // Initial state: weakly taken

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

        for (int i = NUM_TAGGED_TABLES - 1; i >= 0; --i) {
            const auto& cfg = TAGGED_CONFIGS[i];
            auto& table = tagged_tables[i];
            uint64_t idx = compute_hash(PC, ghr, cfg.hist_len, cfg.index_bits) % table.size();
            uint16_t tag = compute_hash(PC, ghr, cfg.hist_len, cfg.tag_bits);

            if (table[idx].tag == tag) {
                state.provider_table = i;
                state.final_pred = (table[idx].ctr >= 0);

                // Find altpred from shorter tables
                for (int j = i - 1; j >= 0; --j) {
                    const auto& alt_cfg = TAGGED_CONFIGS[j];
                    auto& alt_table = tagged_tables[j];
                    uint64_t alt_idx = compute_hash(PC, ghr, alt_cfg.hist_len, alt_cfg.index_bits) % alt_table.size();
                    uint16_t alt_tag = compute_hash(PC, ghr, alt_cfg.hist_len, alt_cfg.tag_bits);
                    if (alt_table[alt_idx].tag == alt_tag) {
                        state.altpred = (alt_table[alt_idx].ctr >= 0);
                        break;
                    }
                }
                break;
            }
        }

        // Use altpred if weak and use_alt_on_na
        if (state.provider_table != -1) {
            auto& entry = tagged_tables[state.provider_table][
                compute_hash(PC, ghr, TAGGED_CONFIGS[state.provider_table].hist_len,
                             TAGGED_CONFIGS[state.provider_table].index_bits) %
                tagged_tables[state.provider_table].size()];
            if (abs(entry.ctr) <= 1 && use_alt_on_na >= USE_ALT_THRESHOLD) {
                state.final_pred = state.altpred;
            }
        }

        // Save state
        state.ghr_snapshot = ghr;
        speculative_states[id] = state;

        return state.final_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        (void)PC; (void)nextPC;
        
        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& state = speculative_states[id];

        // Speculatively update GHR with prediction
        state.ghr_snapshot.push_back(taken);
        if (state.ghr_snapshot.size() > MAX_HISTORY) {
            state.ghr_snapshot.pop_front();
        }
        ghr = state.ghr_snapshot;
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        (void)nextPC;

        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& state = speculative_states[id];

        // Rollback GHR on misprediction
        if (predDir != resolveDir) {
            ghr = state.ghr_snapshot;
            ghr.push_back(resolveDir);
            if (ghr.size() > MAX_HISTORY) ghr.pop_front();
        }

        // Update usefulness and counters
        if (state.provider_table != -1) {
            auto& cfg = TAGGED_CONFIGS[state.provider_table];
            auto& table = tagged_tables[state.provider_table];
            uint64_t idx = compute_hash(PC, state.ghr_snapshot, cfg.hist_len, cfg.index_bits) % table.size();
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
                for (int k = state.provider_table + 1; k < NUM_TAGGED_TABLES; ++k) {
                    auto& alloc_cfg = TAGGED_CONFIGS[k];
                    auto& alloc_table = tagged_tables[k];
                    uint64_t alloc_idx = compute_hash(PC, state.ghr_snapshot, alloc_cfg.hist_len, alloc_cfg.index_bits) % alloc_table.size();
                    if (alloc_table[alloc_idx].u == 0) {
                        alloc_table[alloc_idx].tag = compute_hash(PC, state.ghr_snapshot, alloc_cfg.hist_len, alloc_cfg.tag_bits);
                        alloc_table[alloc_idx].ctr = (resolveDir ? 0 : -1); // Weak correct
                        alloc_table[alloc_idx].u = 0;
                        break;
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
