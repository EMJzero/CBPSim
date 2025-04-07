- the current "my_branch_predictor" works by building a history of prediction outcomes for each instruction ID, by using the bits of "ghist" with a left shift;
- the current "predict_using_given_hist" method of "my_branch_predictor" is dump and just predicts w.r.t. the 2016 predictor's guess;

- look at the interface directly, you can skip the predictor class all together;
- in the interface, "pred_cycle" is the absolute cycle from the start, and can be used to infer how many cycles passed between two occurrences of the same instruction. Those may be helpful to predict;

- "spec_update" (spec = speculative) is called immediately after the prediction is made, so "resolve_dir" is a speculative, unreliable, branch resolution direction, while "pred_dir". Use it to update your state after a prediction w.r.t. ONLY to the prediction itself (e.g. increment a counter of how many times you predicted in a certain direction);
- "notify_instr_execute_resolve" calls the true "update" and lets you know the true outcome of a branch. Use it to store the correct history;
- you can use "spec_update" to update all your data structures temporarily, but then you must use "update" to backtrack and update your data correctly (use the instruction id to find where you have that speculatively updated data);
- the most useful thing that you get out of "spec_update" is the next PC depeding on what you did with the branch, then still, use it to update your state, don't do that while predicting (keep tasks conceptually separated);