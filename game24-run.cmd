conda activate ToT
cd C:\Users\hua_j\researchToT\tot\tree-of-thought-llm
python run_usingLLM.py --task game24 --task_start_index 2 --task_end_index 3 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5

python run.py --task game24 --task_start_index 2 --task_end_index 3 --method_generate propose --method_evaluate value --method_select greedy --n_evaluate_sample 3 --n_select_sample 5

