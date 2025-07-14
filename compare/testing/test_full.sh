
#!/bin/bash
set -euo pipefail                             # optional, but helpful

# --- make conda functions available in this script ------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
# --------------------------------------------------------------------

echo "start testing"

# PhotoMaker
echo "Start PhotoMaker"
cd ../../PhotoMaker
echo "PhotoMaker dir: $(pwd)"
conda deactivate
conda activate photomaker
python3 inference_scripts/inference_pmv2_seed_NS3.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_full

echo "PhotoMaker done"


# PuLID
echo "Start Pulid"
cd ../PuLID
echo "Pulid dir: $(pwd)"
conda deactivate
conda activate pulid

python3 pulid_generate3.py  --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PL_full

sleep 5 # wait 5 seconds
echo "PhotoMaker done"