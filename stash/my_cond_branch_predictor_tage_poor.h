#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <cstdint>
#include <vector>
#include <deque>
#include <unordered_map>
#include <cassert>
#include <algorithm>

// ---- PARAMETRIC DEFINES ----
#define NUM_TAGE_TABLES 7  // Number of tagged tables

// Per-table parameters:
const int TAGE_TABLE_SIZE[NUM_TAGE_TABLES] = { 1024, 512, 512, 256, 256, 128, 128 };
const int TAGE_TAG_BITS[NUM_TAGE_TABLES]   = { 12, 10, 10, 8, 8, 7, 7 };
const int TAGE_HIST_LEN[NUM_TAGE_TABLES]   = { 4, 8, 16, 32, 64, 96, 128 };

#define TAGE_COUNTER_BITS 2    // Bits for prediction counters
#define TAGE_USEFUL_BITS 1     // Bits for "useful" field
#define BIMODAL_SIZE 1024      // Size of simple bimodal predictor
#define MAX_HISTORY_LENGTH 128 // Longest history needed

class SampleCondPredictor
{
public:
    SampleCondPredictor() { setup(); }
    ~SampleCondPredictor() { terminate(); }

    void setup() {
        ghist_length = MAX_HISTORY_LENGTH;
        history.clear();
        tage_tables.resize(NUM_TAGE_TABLES);
        for (int i = 0; i < NUM_TAGE_TABLES; i++) {
            tage_tables[i].assign(TAGE_TABLE_SIZE[i], TageEntry());
        }
        bimodal.assign(BIMODAL_SIZE, 0);
        clock = 0;
        speculative_histories.clear();
    }

    void terminate() {
        speculative_histories.clear();
    }

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0xF);
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, bool tage_pred) {
        (void)tage_pred;
        // Find provider and alternate
        find_provider(PC);
        if (provider_table >= 0) {
            auto &e = tage_tables[provider_table][provider_idx];
            pred_taken = e.counter >= (1 << (TAGE_COUNTER_BITS - 1));
            if (altpred_table >= 0) {
                auto &ae = tage_tables[altpred_table][altpred_idx];
                alt_pred_taken = ae.counter >= (1 << (TAGE_COUNTER_BITS - 1));
            } else {
                alt_pred_taken = (bimodal[PC % BIMODAL_SIZE] >= 2);
            }
        } else {
            pred_taken = (bimodal[PC % BIMODAL_SIZE] >= 2);
            alt_pred_taken = pred_taken;
        }
        return pred_taken;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        (void)PC; (void)nextPC;
        // Save speculative snapshot
        uint64_t id = get_unique_inst_id(seq_no, piece);
        speculative_histories[id] = history;
        // Push predicted outcome
        history.push_back(taken);
        if (history.size() > ghist_length) history.pop_front();
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        (void)nextPC;
        uint64_t id = get_unique_inst_id(seq_no, piece);
        // Rollback speculative history
        auto it = speculative_histories.find(id);
        if (it != speculative_histories.end()) {
            history = it->second;
            speculative_histories.erase(it);
        }
        // Aging useful bits
        if (++clock % 1024 == 0) age_useful_bits();
        // Predict again provider for update
        find_provider(PC);
        if (provider_table >= 0) {
            auto &e = tage_tables[provider_table][provider_idx];
            // Update counter
            if (resolveDir) {
                if (e.counter < ( (1<<TAGE_COUNTER_BITS)-1)) e.counter++;
            } else {
                if (e.counter > 0) e.counter--;
            }
            bool alt_ok = (alt_pred_taken == resolveDir);
            if (predDir != resolveDir) {
                if (e.useful > 0) e.useful--;
                if (!alt_ok) allocate_new_entry(PC, resolveDir);
            } else {
                if (e.useful < ((1<<TAGE_USEFUL_BITS)-1)) e.useful++;
            }
        } else {
            // Bimodal update + allocation on miss
            uint8_t &b = bimodal[PC % BIMODAL_SIZE];
            if (resolveDir) { if (b < 3) b++; } else { if (b > 0) b--; }
            if (predDir != resolveDir) allocate_new_entry(PC, resolveDir);
        }
        // Apply real outcome
        history.push_back(resolveDir);
        if (history.size() > ghist_length) history.pop_front();
    }

private:
    struct TageEntry {
        bool valid = false;
        uint16_t tag = 0;
        uint8_t counter = (1 << (TAGE_COUNTER_BITS - 1));
        uint8_t useful = 0;
    };
    std::vector<std::vector<TageEntry>> tage_tables;
    std::vector<uint8_t> bimodal;
    std::deque<bool> history;
    int ghist_length;
    uint64_t clock;

    int provider_table, altpred_table;
    uint64_t provider_idx, altpred_idx;
    bool pred_taken, alt_pred_taken;
    std::unordered_map<uint64_t, std::deque<bool>> speculative_histories;

    void find_provider(uint64_t PC) {
        provider_table = altpred_table = -1;
        for (int i = NUM_TAGE_TABLES - 1; i >= 0; i--) {
            uint64_t idx = get_index(PC, i);
            uint64_t tag = get_tag(PC, i);
            auto &e = tage_tables[i][idx];
            if (e.valid && e.tag == tag) {
                if (provider_table < 0) {
                    provider_table = i;
                    provider_idx = idx;
                } else if (altpred_table < 0) {
                    altpred_table = i;
                    altpred_idx = idx;
                }
            }
        }
    }

    uint64_t get_index(uint64_t PC, int bank) const {
        uint64_t f = fold_history(TAGE_TABLE_SIZE[bank]);
        return (PC ^ f ^ (PC >> (bank+1))) & (TAGE_TABLE_SIZE[bank]-1);
    }

    uint64_t get_tag(uint64_t PC, int bank) const {
        uint64_t f = fold_history(1ULL << TAGE_TAG_BITS[bank]);
        return (PC ^ (f>>1) ^ (PC >> (bank+2))) & ((1ULL<<TAGE_TAG_BITS[bank]) - 1);
    }

    uint64_t fold_history(uint64_t mod) const {
        uint64_t res = 0;
        int i = 0;
        for (bool b : history) {
            res ^= (uint64_t(b) << (i & 15));
            i++;
        }
        return res % mod;
    }

    void allocate_new_entry(uint64_t PC, bool resolve) {
        for (int i = provider_table+1; i < NUM_TAGE_TABLES; i++) {
            uint64_t idx = get_index(PC, i);
            auto &e = tage_tables[i][idx];
            if (!e.valid || e.useful == 0) {
                e.valid = true;
                e.tag = get_tag(PC, i);
                e.counter = resolve ? (1<<(TAGE_COUNTER_BITS-1))+1
                                     : (1<<(TAGE_COUNTER_BITS-1))-1;
                e.useful = 0;
                break;
            }
        }
    }

    void age_useful_bits() {
        for (auto &tbl : tage_tables)
            for (auto &e : tbl)
                e.useful >>= 1;
    }
};

// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
