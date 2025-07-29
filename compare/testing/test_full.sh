
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
python3 inference_scripts/inference_pmv2_seed_NS4.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_full5

echo "PhotoMaker done"


# PuLID
echo "Start Pulid"
cd ../PuLID
echo "Pulid dir: $(pwd)"
conda deactivate
conda activate pulid

python3 pulid_generate3.py  --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PL_full5

sleep 5 # wait 5 seconds
echo "PuLID done"

# Metrics
cd ../persongen
conda deactivate
conda activate metrics
python3 src/metrics/eval_NS2.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --new_images ../compare/results/PM_full5 --class_file ../compare/testing/classes_ref.json  --out ../compare/results/metrics_PM_full5.csv
python3 src/metrics/eval_NS2.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --new_images ../compare/results/PL_full5 --class_file ../compare/testing/classes_ref.json  --out ../compare/results/metrics_PL_full5.csv

sleep 5 # wait 5 seconds
echo "Metrics done"

# Create PDF reports
python3 ../compare/testing/pdf_output4.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --new_images ../compare/results/PM_full5 --metrics_file ../compare/results/metrics_PM_full5.csv --output_pdf ../compare/testing/output_PM_full5.pdf
python3 ../compare/testing/pdf_output4.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts7.txt --new_images ../compare/results/PL_full5 --metrics_file ../compare/results/metrics_PL_full5.csv --output_pdf ../compare/testing/output_PL_full5.pdf
echo "PDF reports done"