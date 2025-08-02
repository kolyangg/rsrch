# testing our upgrade (in PhotoMaker dir)
pip uninstall -y photomaker
pip install -e .

python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 --use_branched_attention --save_heatmaps
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 --no_branched_attention --save_heatmaps