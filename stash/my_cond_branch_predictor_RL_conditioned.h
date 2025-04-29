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

/*
IDEA:
- allow branches to "collide" on the same entry more often
- try to ensure that colliding branches behave similary
- for each colliding branch consider an ID (e.g. 4-bit hash of the PC)
- use as state both the current history and the branch's ID
*/

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
    // HP: branches commit before 16 other branches are seen.
    // HP: instructions commit within 256 cycles.

    // --- Tunable Parameters ---
    static constexpr uint32_t HISTORY_LENGTH = 245; //501         // Number of recent branches tracked
    static constexpr uint32_t HISTORY_LENGTH_BUFFER = 64; //16    // Number of extra recent branches tracked
    static constexpr uint8_t ID_LENGTH = 10;                      // ID length for branches on the same entry (this amount of least significant bits of the PC is moved from indexing the table entry to a part of the state)
    static constexpr uint8_t WEIGHT_BITS = 8;                     // Weight resolution (in bits)
    static constexpr uint32_t THETA = 1.93 * HISTORY_LENGTH + 14; // Confidence threshold for training
    static constexpr uint8_t LEARNING_RATE = 1;                   // Weight update step
    static constexpr uint32_t MAX_TABLE_ENTRIES = 512; //1024     // Max number of PCs tracked
    static constexpr size_t MAX_BYTES = 192 * 1024;               // Memory budget: 192KB

    static constexpr uint16_t NUM_FEATURES = HISTORY_LENGTH + ID_LENGTH + 1; // +1 for bias
    static constexpr int8_t WEIGHT_MAX = (1 << (WEIGHT_BITS - 1)) - 1;
    static constexpr int8_t WEIGHT_MIN = -(1 << (WEIGHT_BITS - 1));

    // --- Internal State ---
    uint8_t pred_cycle;                                             // Cyclic counter of prediction
    std::bitset<HISTORY_LENGTH + HISTORY_LENGTH_BUFFER> GHR;        // Global History Register
    std::unordered_map<uint64_t, std::vector<int8_t>> weights;      // Per-PC weight vectors
    std::unordered_map<uint64_t, uint8_t> speculative_updates;      // For rollback, maps instruction ID to its absolute clock cycle
    std::unordered_map<uint64_t, int> past_predictions;             // For weights update, store raw past predictions

public:
    SampleCondPredictor(void) {}

    void setup() {
        pred_cycle = 0;
        GHR.reset();
        check_memory_budget(); // Ensure configuration is within 192KB
    }

    void terminate() {
        weights.clear();
        speculative_updates.clear();
    }

    // Create a unique instruction ID from seq_no and micro-op piece
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    // Compute idx of the weights for the present PC
    // TODO: be smarter than just a <mod>
    uint64_t get_weights_idx(uint64_t PC) const {
        return (PC >> ID_LENGTH) % MAX_TABLE_ENTRIES;
    }

    // Predict using linear Q(s,a) = wᵀ·ϕ(s)
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, bool tage_pred) {
        (void)tage_pred;

        auto& w = weights[get_weights_idx(PC)];
        if (w.empty()) {
            w.resize(NUM_FEATURES, 0);
        }

        // Compute dot product between weights and GHR-based features
        int sum = w[0]; // bias
        for (int i = 0; i < HISTORY_LENGTH; i++) {
            sum += GHR[i] ? w[i + 1] : -w[i + 1];
        }
        for (uint i = 1; i <= ID_LENGTH; i++) {
            bool bit = (PC >> i) & 1;
            sum += bit ? w[HISTORY_LENGTH + i] : -w[HISTORY_LENGTH + i];
        }

        // Save current prediction for future weights updates
        past_predictions[get_unique_inst_id(seq_no, piece)] = sum;

        return sum >= 0; // positive score → predict taken
    }

    // Speculative GHR update after prediction
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        pred_cycle += 1;
        GHR <<= 1;
        GHR.set(0, taken);

        // Save current cycle to rollback on misprediction
        speculative_updates[get_unique_inst_id(seq_no, piece)] = pred_cycle;
    }

    // Final update after branch resolution
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        uint64_t id = get_unique_inst_id(seq_no, piece);
        auto& w = weights[get_weights_idx(PC)];
        if (w.empty()) w.resize(NUM_FEATURES, 0);

        // Restore speculative GHR if prediction was wrong
        auto it_su = speculative_updates.find(id);
        if (it_su == speculative_updates.end()) {
            return;
        }
        uint8_t delta_cycles = pred_cycle - it_su->second;
        if (resolveDir != predDir) {
            GHR.set(delta_cycles, resolveDir);
        }
        speculative_updates.erase(id);

        // Recover past raw decision
        auto it_pp = past_predictions.find(id);
        past_predictions.erase(id);
        if (it_pp == past_predictions.end()) {
            return;
        }
        int sum = it_pp->second;

        // Not enough past GHR entries buffered (too many other branche seen between prediction and update), don't train
        if (delta_cycles > HISTORY_LENGTH_BUFFER) {
            return;
        }

        // Update weights if low confidence or incorrect prediction
        if ((resolveDir != (sum >= 0)) || std::abs(sum) <= THETA) {
            int8_t target = resolveDir ? 1 : -1;

            // Bias weight
            w[0] = std::max(std::min((int8_t)(w[0] + LEARNING_RATE * target), WEIGHT_MAX), WEIGHT_MIN);

            // History-based weights
            for (uint i = 0; i < HISTORY_LENGTH; i++) {
                int8_t update = (GHR[i + delta_cycles] ? 1 : -1) * target * LEARNING_RATE;
                w[i + 1] = std::max(std::min((int8_t)(w[i + 1] + update), WEIGHT_MAX), WEIGHT_MIN);
            }
            // PC-based weights
            for (uint i = 1; i <= ID_LENGTH; i++) {
                bool bit = (PC >> i) & 1;
                int8_t update = (bit ? 1 : -1) * target * LEARNING_RATE;
                w[HISTORY_LENGTH + i] = std::max(std::min((int8_t)(w[HISTORY_LENGTH + i] + update), WEIGHT_MAX), WEIGHT_MIN);
            }
        }
    }

private:
    // Estimate memory usage and check against budget
    void check_memory_budget() {
        size_t bits_per_entry = NUM_FEATURES * WEIGHT_BITS;
        size_t total_bits = MAX_TABLE_ENTRIES * bits_per_entry + HISTORY_LENGTH + HISTORY_LENGTH_BUFFER + 8;
        size_t total_bytes = (total_bits >> 3) + (total_bits & 7 > 0 ? 1 : 0);
        std::cout << "Memory used: " << total_bytes << "B / " << MAX_BYTES << "B" << std::endl;
        assert(total_bytes <= MAX_BYTES && "Predictor exceeds MAX_BYTES memory limit! Reduce HISTORY_LENGTH or MAX_TABLE_ENTRIES.");
    }
};
// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
