/*
    BACKWARD TAKEN FORWARD NOT TAKEN BRANCH PREDICTOR
*/

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>


class SampleCondPredictor
{
        std::unordered_map<uint64_t/*key*/, uint64_t/*val*/> next_pc_map;
    public:

        SampleCondPredictor (void)
        {
        }

        void setup()
        {
        }

        void terminate()
        {
        }

        // sample function to get unique instruction id
        uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
        {
            assert(piece < 16);
            return (seq_no << 4) | (piece & 0x000F);
        }

        bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred)
        {
            (void)tage_pred;  // Ignore cbp2016 prediction
            
            //const auto id = get_unique_inst_id(seq_no, piece);
            
            // Retrieve nextPC (default: assume forward branch)
            uint64_t nextPC = PC + 4;  // Default assumption
            auto it = next_pc_map.find(PC);
            if (it != next_pc_map.end()) {
                nextPC = it->second;
            }

            // Predict not taken for forward branches (PC < nextPC)
            // Predict taken for backward branches (PC >= nextPC)
            return PC >= nextPC;
        }

        // AKA: speculative update
        void history_update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC)
        {
            if(taken) {
                next_pc_map[PC] = nextPC;
            }
        }

        void update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC)
        {
            if(resolveDir) {
                next_pc_map[PC] = nextPC;
            }
        }
};
// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
