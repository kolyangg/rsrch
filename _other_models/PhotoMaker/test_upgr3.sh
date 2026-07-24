# testing our upgrade (in PhotoMaker dir)
pip uninstall -y photomaker
pip install -e .

# # No ID, then PhotoMaker from step 10 and branched attn from step 15

# Keanu

# # # id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/new1 \
#  --face_embed_strategy id_embeds \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.3 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0


 # # start_merge_step > branched_attn_start_step
python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
 --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompt_one2.txt \
 --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/new1 \
 --face_embed_strategy face \
 --start_merge_step 15 \
 --branched_attn_start_step 10 \
 --use_branched_attention \
 --auto_mask_ref \
 --pose_adapt_ratio 0.3 \
 --ca_mixing_for_face 0 \
 --use_id_embeds 0