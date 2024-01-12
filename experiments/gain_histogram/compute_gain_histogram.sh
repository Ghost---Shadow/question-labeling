python src/one_off_experiments/eval_checkpoint.py --dataset_name=wiki_multihop_qa_with_q --model_type=mpnet --checkpoint_path=baseline
python src/one_off_experiments/eval_checkpoint.py --dataset_name=wiki_multihop_qa_with_q --model_type=mpnet --checkpoint_path=./checkpoints/gpt35_mpnet_kldiv_qd_8a5d1/seed_42/epoch_6.pth
python src/one_off_experiments/eval_checkpoint.py --dataset_name=wiki_multihop_qa_with_q --model_type=mpnet --checkpoint_path=./checkpoints/gpt35_mpnet_ni_kldiv_9de1b/seed_42/epoch_6.pth
python src/one_off_experiments/eval_checkpoint.py --dataset_name=wiki_multihop_qa_with_q --model_type=mpnet --checkpoint_path=./checkpoints/gpt35_mpnet_kldiv_only_d_ed701/seed_42/epoch_6.pth
python src/one_off_experiments/eval_checkpoint.py --dataset_name=wiki_multihop_qa_with_q --model_type=mpnet --checkpoint_path=./checkpoints/gpt35_mpnet_kldiv_only_q_a1ba2/seed_42/epoch_6.pth

python src/one_off_experiments/plot_gain_histograms.py

source devops/upload_artifacts.sh

source devops/stop_current_gcp_instance.sh
