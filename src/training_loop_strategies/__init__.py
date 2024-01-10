from training_loop_strategies import iterative_strategy, non_iterative_strategy

TRAINING_LOOP_STRATEGY_LUT = {
    "iterative_strategy": (iterative_strategy.train_step, iterative_strategy.eval_step),
    "non_iterative_strategy": (
        non_iterative_strategy.train_step,
        non_iterative_strategy.eval_step,
    ),
}
