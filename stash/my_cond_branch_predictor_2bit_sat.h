/*
    TWO BIT SATURATING PREDICTOR
*/

#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>
#include <cstdint>
#include <iostream>

class SaturatingCounter
{
    private:
        uint8_t value;

    public:
        SaturatingCounter(uint8_t initial = 0) {
            value = initial & 0x03; // Mask to only use 2 bits
        }

        SaturatingCounter operator+(int) const {
            SaturatingCounter result(*this);
            if (result.value < 2)
                result.value++;
            return result;
        }

        SaturatingCounter operator-(int) const {
            SaturatingCounter result(*this);
            if (result.value > 0)
                result.value--;
            return result;
        }

        bool msb() const {
            return (value & 0x02) != 0;
        }

        uint8_t get() const {
            return value;
        }
};

class SampleCondPredictor
{
    private:
        SaturatingCounter counter;

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
            
            // The prediction is the most significant bit of the counter
            return counter.msb();
        }

        // AKA: speculative update
        void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC)
        {
        }

        void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC)
        {
            counter =+ taken:
        }
};
// =================
// Predictor End
// =================

#endif
static SampleCondPredictor cond_predictor_impl;
