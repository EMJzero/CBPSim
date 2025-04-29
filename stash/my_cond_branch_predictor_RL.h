#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>

#include <cstdint>
#include <vector>
#include <bitset>
#include <unordered_map>
#include <cassert>
#include <cmath>
#include <iostream>

// === RL-Based Branch Predictor ===
// Implements a lightweight perceptron-like predictor using RL-style online updates
// - State: Global History Register (GHR)
// - Action: Predict taken/not taken
// - Reward: +1 for correct, -1 for incorrect
// - Model: Linear dot-product model with signed weights
// - Online TD-like learning with bounded weights
// ================================

class SampleCondPredictor
{
    // --- Tunable Parameters ---
    static constexpr int8_t HISTORY_LENGTH = 32;             // Number of recent branches tracked
    static constexpr int8_t WEIGHT_BITS = 8;                 // Weight resolution (in bits)
    static constexpr int8_t THETA = 20;                      // Confidence threshold for training
    static constexpr int8_t LEARNING_RATE = 1;               // Weight update step
    static constexpr int32_t MAX_TABLE_ENTRIES = 4096;       // Max number of PCs tracked
    static constexpr size_t MAX_BYTES = 192 * 1024;          // Memory budget: 192KB

    static constexpr int16_t NUM_FEATURES = HISTORY_LENGTH + 1; // +1 for bias
    static constexpr int8_t WEIGHT_MAX = (1 << (WEIGHT_BITS - 1)) - 1;
    static constexpr int8_t WEIGHT_MIN = -(1 << (WEIGHT_BITS - 1));

    // --- Internal State ---
    std::bitset<HISTORY_LENGTH> GHR;                                // Global History Register
    std::unordered_map<uint64_t, std::vector<int8_t>> weights;      // Per-PC weight vectors
    std::unordered_map<uint64_t, std::bitset<HISTORY_LENGTH>> speculative_GHRs; // For rollback

public:
    SampleCondPredictor(void) {}

    void setup() {
        GHR.reset();
        check_memory_budget(); // Ensure configuration is within 192KB
    }

    void terminate() {
        weights.clear();
        speculative_GHRs.clear();
    }

    // Create a unique instruction ID from seq_no and micro-op piece
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    // Predict using linear Q(s,a) = wᵀ·ϕ(s)
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred) {
        (void)tage_pred;

        auto& w = weights[PC];
        if (w.empty()) {
            w.resize(NUM_FEATURES, 0);
        }

        // Compute dot product between weights and GHR-based features
        int sum = w[0]; // bias
        for (int i = 0; i < HISTORY_LENGTH; ++i) {
            sum += GHR[i] ? w[i + 1] : -w[i + 1];
        }

        // Save current GHR in case we need to rollback on misprediction
        speculative_GHRs[get_unique_inst_id(seq_no, piece)] = GHR;

        return sum >= 0; // positive score → predict taken
    }

    // Speculative GHR update after prediction
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        GHR <<= 1;
        GHR.set(0, taken);
    }

    // Final update after branch resolution
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& w = weights[PC];
        if (w.empty()) w.resize(NUM_FEATURES, 0);

        // Restore speculative GHR if prediction was wrong
        if (resolveDir != predDir) {
            auto it = speculative_GHRs.find(id);
            if (it != speculative_GHRs.end()) {
                GHR = it->second;
                GHR <<= 1;
                GHR.set(0, resolveDir);
            }
        }

        speculative_GHRs.erase(id);

        // Recompute sum for learning decision
        int sum = w[0];
        for (int i = 0; i < HISTORY_LENGTH; ++i) {
            sum += GHR[i] ? w[i + 1] : -w[i + 1];
        }

        // Update weights if low confidence or incorrect prediction
        if ((resolveDir != (sum >= 0)) || std::abs(sum) <= THETA) {
            int8_t target = resolveDir ? 1 : -1;

            // Bias weight
            w[0] = std::max(std::min((int8_t)(w[0] + LEARNING_RATE * target), WEIGHT_MAX), WEIGHT_MIN);

            // History-based weights
            for (int i = 0; i < HISTORY_LENGTH; ++i) {
                int8_t update = (GHR[i] ? 1 : -1) * target * LEARNING_RATE;
                w[i + 1] = std::max(std::min((int8_t)(w[i + 1] + update), WEIGHT_MAX), WEIGHT_MIN);
            }
        }
    }

private:
    // Estimate memory usage and check against budget
    void check_memory_budget() {
        size_t bytes_per_entry = NUM_FEATURES * sizeof(int8_t);
        size_t estimated_total = MAX_TABLE_ENTRIES * bytes_per_entry;
        std::cout << "Memory used: " << estimated_total << "B / " << MAX_BYTES << "B" << std::endl;
        assert(estimated_total <= MAX_BYTES && "Predictor exceeds MAX_BYTES memory limit! Reduce HISTORY_LENGTH or MAX_TABLE_ENTRIES.");
    }
};
// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
