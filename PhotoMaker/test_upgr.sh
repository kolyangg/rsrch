# testing our upgrade (in PhotoMaker dir)
pip uninstall -y photomaker
pip install -e .

python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 --face_embed_strategy id_embeds --use_branched_attention --save_heatmaps
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 --face_embed_strategy face --use_branched_attention --save_heatmaps
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 --no_branched_attention --save_heatmaps

## ref3 (eddie)
# pm
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 --no_branched_attention --save_heatmaps

# id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 --face_embed_strategy id_embeds --use_branched_attention --save_heatmaps

# face
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 --face_embed_strategy id_embeds --use_branched_attention --save_heatmaps
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 --face_embed_strategy face --use_branched_attention --save_heatmaps
