import wandb


def calculate_train_loss(step):
    return 0.5 / (step + 1)


configs = [
    (2, 150, 100, 200),
    (5, 150 * 2 // 5, 100 * 2 // 5, 200 * 2 // 5),
]

for batch_size, interrupt_step, restart_step, end_step in configs:
    name = f"potato_{batch_size}"

    run = wandb.init(project="debug_wandb", name=name)

    for step in range(interrupt_step):
        train_loss = calculate_train_loss(step)
        wandb.log({"loss": train_loss}, step=step * batch_size)

    wandb.finish()

    wandb.init(project="debug_wandb", name=name)

    for step in range(restart_step, end_step):
        train_loss = calculate_train_loss(step)
        wandb.log({"loss": train_loss}, step=step * batch_size)

    wandb.finish()

run = wandb.init(project="debug_wandb", name="tomato")
for step in range(10):
    wandb.log({"loss_a": step}, step=step)
    wandb.log({"loss_b": step * 2}, step=step)
wandb.finish()
