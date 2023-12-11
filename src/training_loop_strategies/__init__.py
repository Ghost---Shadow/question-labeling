from training_loop_strategies import iterative_strategy

TRAINING_LOOP_STRATEGY_LUT = {
    "iterative_strategy": iterative_strategy.train_step,
    "iterative_strategy_full_precision": iterative_strategy.train_step_full_precision,
}
