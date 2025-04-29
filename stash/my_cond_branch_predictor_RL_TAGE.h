#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>

#include <cstdint>
#include <vector>
#include <bitset>
#include <unordered_map>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <algorithm>

class SampleCondPredictor {
    // === Tunable Parameters ===
    static constexpr int NUM_TABLES = 4;
    static constexpr std::array<int, NUM_TABLES> HISTORY_LENGTHS = {16, 32, 64, 128};
    static constexpr int WEIGHT_BITS = 8;
    static constexpr int THETA = 20;  // Confidence threshold
    static constexpr int LEARNING_RATE = 1;
    static constexpr int TABLE_ENTRIES = 8192; // Must be power of 2
    static constexpr int GHR_MAX = 128;
    static constexpr size_t MAX_BYTES = 192 * 1024;

    static constexpr int8_t WEIGHT_MAX = (1 << (WEIGHT_BITS - 1)) - 1;
    static constexpr int8_t WEIGHT_MIN = -(1 << (WEIGHT_BITS - 1));

    using WeightTable = std::vector<int8_t>;

    struct FoldedHistory {
        std::bitset<GHR_MAX> full;
        std::array<uint32_t, NUM_TABLES> folded{};
    };

    std::array<std::vector<int8_t>, NUM_TABLES> weight_tables; // Shared hashed weight tables
    FoldedHistory GHR;
    std::unordered_map<uint64_t, FoldedHistory> speculative_GHRs;

    // Confidence counter per PC (simplified)
    std::unordered_map<uint64_t, int> confidence;

public:
    SampleCondPredictor() {}

    void setup() {
        for (auto& table : weight_tables) {
            table.resize(TABLE_ENTRIES, 0);
        }
        GHR.full.reset();
        check_memory_budget();
    }

    void terminate() {
        for (auto& table : weight_tables) table.clear();
        speculative_GHRs.clear();
        confidence.clear();
    }

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred) {
        (void)tage_pred;

        int total_score = 0;
        for (int i = 0; i < NUM_TABLES; ++i) {
            uint32_t idx = (PC ^ GHR.folded[i]) & (TABLE_ENTRIES - 1);
            total_score += weight_tables[i][idx];
        }

        speculative_GHRs[get_unique_inst_id(seq_no, piece)] = GHR;
        return total_score >= 0;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        GHR.full <<= 1;
        GHR.full.set(0, taken);
        for (int i = 0; i < NUM_TABLES; ++i) {
            int len = HISTORY_LENGTHS[i];
            GHR.folded[i] = fold_history(GHR.full, len);
        }
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        auto id = get_unique_inst_id(seq_no, piece);
        if (speculative_GHRs.count(id)) {
            if (resolveDir != predDir) {
                GHR = speculative_GHRs[id];
                GHR.full <<= 1;
                GHR.full.set(0, resolveDir);
                for (int i = 0; i < NUM_TABLES; ++i) {
                    GHR.folded[i] = fold_history(GHR.full, HISTORY_LENGTHS[i]);
                }
            }
            speculative_GHRs.erase(id);
        }

        int total_score = 0;
        std::array<uint32_t, NUM_TABLES> indices;
        for (int i = 0; i < NUM_TABLES; ++i) {
            indices[i] = (PC ^ GHR.folded[i]) & (TABLE_ENTRIES - 1);
            total_score += weight_tables[i][indices[i]];
        }

        // Confidence logic
        bool prediction = total_score >= 0;
        bool update_needed = (resolveDir != prediction) || std::abs(total_score) <= THETA;

        if (update_needed) {
            int target = resolveDir ? 1 : -1;
            for (int i = 0; i < NUM_TABLES; ++i) {
                int8_t& w = weight_tables[i][indices[i]];
                w = std::max(std::min((int8_t)(w + LEARNING_RATE * target), WEIGHT_MAX), WEIGHT_MIN);
            }
        }

        // Update confidence
        if (resolveDir == prediction) confidence[PC]++;
        else confidence[PC] = 0;
    }

private:
    uint32_t fold_history(const std::bitset<GHR_MAX>& hist, int len) {
        uint32_t result = 0;
        for (int i = 0; i < len; ++i) {
            result ^= (hist[i] << (i % 16));
        }
        return result;
    }

    void check_memory_budget() {
        size_t total = NUM_TABLES * TABLE_ENTRIES * sizeof(int8_t);
        std::cout << "Memory used: " << estimated_total << "B / " << MAX_BYTES << "B" << std::endl;
        assert(total <= MAX_BYTES && "Exceeded MAX_BYTES memory budget for predictor.");
    }
};

// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
