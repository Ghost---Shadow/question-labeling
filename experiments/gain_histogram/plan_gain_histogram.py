import os
import glob


def find_latest_checkpoint(experiment_dir):
    """
    Finds the latest checkpoint for a given experiment and seed.
    """
    latest_checkpoints = []
    for seed_dir in glob.glob(f"{experiment_dir}seed_*/"):
        checkpoint_pattern = f"{seed_dir}epoch_*.pth"
        list_of_files = glob.glob(checkpoint_pattern)
        if list_of_files:
            latest_checkpoint = max(list_of_files, key=os.path.getmtime)
            latest_checkpoints.append(latest_checkpoint)
    return latest_checkpoints


# Base path for checkpoints
base_path = "checkpoints/"

# List of datasets
datasets = ["hotpot_qa_with_q", "wiki_multihop_qa_with_q"]

# Shell script file name
shell_script_file = "experiments/gain_histogram/compute_gain_histogram.sh"

with open(shell_script_file, "w") as script:
    script.write("#!/bin/bash\n")

    # Iterate over each experiment
    for dataset in datasets:
        script.write(f"\n# {dataset}\n")
        # Add baseline evaluation for each dataset
        script.write(
            f"python src/run_analysis_scripts/eval_checkpoint.py --dataset_name={dataset} --model_type=mpnet --checkpoint_path=baseline\n"
        )

        for experiment_dir in glob.glob(f"{base_path}*/"):
            experiment_name = os.path.basename(os.path.dirname(experiment_dir))

            # Find the latest checkpoint for each seed in the experiment
            latest_checkpoints = find_latest_checkpoint(experiment_dir)

            # Add evaluations for each checkpoint
            for checkpoint in latest_checkpoints:
                script.write(
                    f"python src/run_analysis_scripts/eval_checkpoint.py --dataset_name={dataset} --model_type=mpnet --checkpoint_path={checkpoint}\n"
                )

    # Additional commands
    script.write(f"\n")
    script.write("python src/run_analysis_scripts/plot_gain_histograms.py\n")
    script.write("source devops/upload_artifacts.sh\n")
    script.write("source devops/stop_current_gcp_instance.sh\n")

print(f"Shell script '{shell_script_file}' has been generated.")
