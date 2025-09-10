# testing our upgrade (in PhotoMaker dir)
pip uninstall -y photomaker
pip install -e .

# # No ID, then PhotoMaker from step 10 and branched attn from step 15

# Keanu

# # id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.0 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0 \

 # id_embeds and ca_mixing_for_face = 1
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.0 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0 \


# # face
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref

# # branched_attn_start_step < start_merge_step
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 20 \
#  --branched_attn_start_step 15 \
#  --branched_start_mode branched \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.0 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0 \

# branched_attn_start_step < start_merge_step NEW
python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
 --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
 --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
 --face_embed_strategy face --save_heatmaps \
 --branched_attn_start_step 15 \
 --photomaker_start_step 20 \
 --merge_start_step 15 \
 --branched_start_mode both \
 --use_branched_attention \
 --auto_mask_ref \
 --pose_adapt_ratio 0.0 \
 --ca_mixing_for_face 0 \
 --use_id_embeds 0 \
 --force_par_before_pm 1 \

# # face + pose_adapt_ratio = 0.1
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.10 \
#  --ca_mixing_for_face 1 \
#  --use_id_embeds 1 \


# Eddie
# # id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.0 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0 \

# # face + pose_adapt_ratio = 0.1
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.10 \
#  --ca_mixing_for_face 1 \
#  --use_id_embeds 1 \


#  # face
#  python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref3 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade3 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref


# # No ID, then branched attn from step 10 and PhotoMaker from step 15
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref2 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 15 \
#  --branched_attn_start_step 10 \
#  --use_branched_attention


## Sydney

# # No ID, then PhotoMaker from step 10 and branched attn from step 15

# id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref \
#  --pose_adapt_ratio 0.5 \
#  --ca_mixing_for_face 0 \
#  --use_id_embeds 0 \




# face
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --face_embed_strategy face --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref


# ## Sydney - PM
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1 \
#  --no_branched_attention \


# ## Jisoo

# # No ID, then PhotoMaker from step 10 and branched attn from step 15
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref4 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade4 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref


## Marion

# No ID, then PhotoMaker from step 10 and branched attn from step 15

# # # face_embed_strategy = id_embeds
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref5 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade5 \
#  --face_embed_strategy id_embeds \
#  --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref

# # # face_embed_strategy = face
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref5 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade5 \
#  --face_embed_strategy face \
#  --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
#  --auto_mask_ref


#  # No ID, then PhotoMaker from step 10 and branched attn from step 15
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref5 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade5 \
#  --face_embed_strategy id_embeds --save_heatmaps \
#  --start_merge_step 10 \
#  --branched_attn_start_step 15 \
#  --use_branched_attention \
# #  --auto_mask_ref



# # No ID, then PhotoMaker from step 10 and branched attn from step 15
# python3 inference_scripts/inference_pmv2_seed_NS4_upd2.py \
#  --image_folder ../compare/testing/ref5 --prompt_file ../compare/testing/prompt_one2.txt \
#  --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade5 \
#  --save_heatmaps \
#  --no_branched_attention \