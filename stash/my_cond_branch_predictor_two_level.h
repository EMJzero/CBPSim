/*
    TWO LEVEL BRANCH PREDICTOR
*/

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cassert>

// ------------------------------------------------------------------------
// Predictor Configuration Parameters:
//
// HISTORY_LENGTH: Number of bits in the branch history register (k).
// PT_SETS: Number of sets in the pattern table. Branches are grouped based on PC.
// PT_ENTRIES: Number of entries per set, equal to 2^k.
//
// Storage required: PT_SETS * PT_ENTRIES
// ------------------------------------------------------------------------
#define HISTORY_LENGTH 4       // k: number of bits in the history register
#define PT_SETS 256            // Number of pattern table sets (grouping branches by PC)
#define PT_ENTRIES (1 << HISTORY_LENGTH)  // Number of entries per set (2^k)

struct SampleHist
{
    uint64_t ghist;   // Holds the branch history (only the lower k bits are used)
    bool tage_pred;   // Holds the prediction made at prediction time (unused here)
    
    SampleHist()
    {
        ghist = 0;
        tage_pred = false;
    }
};

class SampleCondPredictor
{
    // Map for storing speculative branch history checkpoints.
    // The simulator uses a unique identifier (built from seq_no and piece) to index
    // the history state that was used at prediction time.
    std::unordered_map<uint64_t, SampleHist> pred_time_histories;
    
    // Branch History Table (BHT): maps branch PC to its k-bit history register.
    // This implements the first level of the two-level predictor.
    std::unordered_map<uint64_t, uint32_t> branch_history;
    
    // Pattern Table: global storage for the two-bit saturating counters.
    // It is organized as PT_SETS sets, each with PT_ENTRIES entries.
    // Each entry is a two-bit counter stored in a uint8_t.
    std::vector<uint8_t> pattern_table;

public:

    SampleCondPredictor(void)
    {
    }

    // Called at the beginning of simulation. Initializes predictor state.
    void setup()
    {
        // Initialize the pattern table:
        // - Total size: PT_SETS * PT_ENTRIES.
        // - Default value: 1 (weakly not taken).
        pattern_table.resize(PT_SETS * PT_ENTRIES, 1);
        
        // Clear branch history and speculative checkpoints.
        branch_history.clear();
        pred_time_histories.clear();
    }

    // Called at the end of simulation. Can be used to dump state if needed.
    void terminate()
    {
    }

    // Returns a unique instruction id based on the sequence number and a piece.
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
    {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    // Called for each conditional branch to return a prediction.
    // Implements the two-level PA predictor:
    //   1. Uses the branch’s PC to retrieve its k-bit history (from the BHT).
    //   2. Uses (PC modulo PT_SETS) to select a group in the pattern table.
    //   3. Indexes the set with the branch’s history to select a 2-bit counter.
    //   4. Predicts taken if the counter is 2 or 3.
    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred)
    {
        // For the PA predictor, the external tage_pred input is ignored.
        
        // Retrieve the branch's history register for this PC. Default to 0 if unseen.
        uint32_t history = 0;
        auto it = branch_history.find(PC);
        if(it != branch_history.end()){
            history = it->second;
        }
        
        // Determine the pattern table set using the branch's PC.
        uint64_t set_index = PC % PT_SETS;
        // Compute the index into the pattern table: combine set and history.
        uint64_t table_index = set_index * PT_ENTRIES + (history & ((1 << HISTORY_LENGTH) - 1));
        
        // Read the two-bit saturating counter.
        uint8_t counter = pattern_table[table_index];
        
        // Prediction: taken if counter >= 2, not taken otherwise.
        bool pred_taken = (counter >= 2);
        
        // Save the current history in a speculative checkpoint.
        SampleHist cp;
        cp.ghist = history;         // save the branch history used for this prediction
        cp.tage_pred = pred_taken;  // record the prediction (for bookkeeping)
        uint64_t unique_inst_id = get_unique_inst_id(seq_no, piece);
        pred_time_histories.emplace(unique_inst_id, cp);
        
        return pred_taken;
    }

    // Called immediately after prediction, to speculatively update predictor history.
    // For the PA predictor, update the branch history register with the outcome.
    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC)
    {
        // Retrieve current history for the branch; default to 0.
        uint32_t history = 0;
        auto it = branch_history.find(PC);
        if(it != branch_history.end()){
            history = it->second;
        }
        // Shift in the new outcome (1 for taken, 0 for not taken)
        history = ((history << 1) | (taken ? 1 : 0)) & ((1 << HISTORY_LENGTH) - 1);
        branch_history[PC] = history;
    }

    // Called after branch resolution to update the predictor.
    // It uses the speculative checkpoint (saved at prediction time) to update the
    // corresponding pattern table entry.
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC)
    {
        uint64_t unique_inst_id = get_unique_inst_id(seq_no, piece);
        auto it = pred_time_histories.find(unique_inst_id);
        assert(it != pred_time_histories.end());  // Must find the checkpoint!
        const SampleHist& cp = it->second;
        update(PC, resolveDir, predDir, nextPC, cp);
        // Remove the checkpoint once update is complete.
        pred_time_histories.erase(unique_inst_id);
    }

    // Updates the pattern table (second-level) counter that was used for the prediction.
    // The counter is incremented on a taken outcome (if not already at max 3) and
    // decremented on a not-taken outcome (if not already at min 0).
    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use)
    {
        // Compute the index into the pattern table using the PC and the checkpointed history.
        uint64_t set_index = PC % PT_SETS;
        uint64_t index = set_index * PT_ENTRIES + (hist_to_use.ghist & ((1 << HISTORY_LENGTH) - 1));
        
        // Retrieve and update the 2-bit saturating counter.
        uint8_t counter = pattern_table[index];
        if(resolveDir) {
            // Branch was taken: increment counter (saturating at 3).
            if(counter < 3) {
                counter++;
            }
        }
        else {
            // Branch was not taken: decrement counter (saturating at 0).
            if(counter > 0) {
                counter--;
            }
        }
        pattern_table[index] = counter;
    }
};

// Global instance of the branch predictor implementation.
static SampleCondPredictor cond_predictor_impl;

#endif
