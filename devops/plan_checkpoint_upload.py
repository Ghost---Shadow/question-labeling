import os
import glob

base_path = "./checkpoints/"
bucket_name = "q-labeling"
shell_script_file = "./upload_checkpoints.sh"

with open(shell_script_file, "w") as script:
    script.write("#!/bin/bash\n\n")
    for experiment_dir in glob.glob(f"{base_path}*/"):
        experiment_name = os.path.basename(os.path.dirname(experiment_dir))
        for seed_dir in glob.glob(f"{experiment_dir}seed_*/"):
            seed = os.path.basename(os.path.dirname(seed_dir))
            checkpoint_pattern = f"{seed_dir}epoch_*.pth"
            list_of_files = glob.glob(checkpoint_pattern)
            if list_of_files:
                latest_checkpoint = max(list_of_files, key=os.path.getctime)
                destination_blob_name = f"{bucket_name}/checkpoints/{experiment_name}/{seed}/{os.path.basename(latest_checkpoint)}"
                script.write(
                    f'gsutil cp "{latest_checkpoint}" "gs://{destination_blob_name}"\n'
                )

print(f"Written to {shell_script_file}")
