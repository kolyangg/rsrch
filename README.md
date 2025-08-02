## Photomaker

```bash
cd PhotoMaker
conda create --name photomaker python=3.10
conda activate photomaker
pip install -U pip

# Install requirements
cd ..
pip install -r pm_requirements.txt

# Install photomaker
pip install git+https://github.com/TencentARC/PhotoMaker.git

# Run inference
python3 inference_scripts/inference_pmv2.py

python3 inference_scripts/inference_pmv2_seed_NS2.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --output_dir ../compare/results/PM

# new
python3 inference_scripts/inference_pmv2_seed_NS2.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3.txt --output_dir ../compare/results/PM3

# new2
python3 inference_scripts/inference_pmv2_seed_NS3.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3pm.txt --class_file ../compare/testing/classes.json --output_dir ../compare/results/PM3b

# new_full
python3 inference_scripts/inference_pmv2_seed_NS3.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_full

# testing our upgrade (in PhotoMaker dir)
pip uninstall -y photomaker
pip install -e .

python3 inference_scripts/inference_pmv2_seed_NS4.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade0

python3 inference_scripts/inference_pmv2_seed_NS4_upd.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompt_one.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PM_upgrade1


```



## PuLID

```bash
cd PuLID
conda create --name pulid python=3.10
conda activate pulid

# Install requirements
cd ..
pip install -r pl_requirements.txt

# Run inference
python3 pulid_generate2.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --output_dir ../compare/results/PL

python3 pulid_generate3.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3pm.txt --class_file ../compare/testing/classes.json --output_dir ../compare/results/PL3b


# new_full
python3 pulid_generate3.py  --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --class_file ../compare/testing/classes_ref.json --output_dir ../compare/results/PL_full

```


## Metrics

```bash
cd persongen
conda create --name metrics python=3.10
conda activate metrics

# Install requirements
pip install -r requirements.txt

# Run eval
cd persongen
python3 src/metrics/eval_NS.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PM  --out ../compare/results/metrics_PM.csv

python3 src/metrics/eval_NS.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PL  --out ../compare/results/metrics_PL.csv

# new
cd persongen
python3 src/metrics/eval_NS.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3pm.txt --new_images ../compare/results/PM3  --out ../compare/results/metrics_PM3.csv

python3 src/metrics/eval_NS.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3.txt --new_images ../compare/results/PL3  --out ../compare/results/metrics_PL3.csv

python3 src/metrics/eval_NS.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PL_new_one  --out ../compare/results/metrics_PL_new_one.csv

# new_full
cd persongen
python3 src/metrics/eval_NS2.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PM_full --class_file ../compare/testing/classes_ref.json  --out ../compare/results/metrics_PM_full.csv

python3 src/metrics/eval_NS2.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PL_full --class_file ../compare/testing/classes_ref.json  --out ../compare/results/metrics_PL_full.csv


# Create a pdf output
cd ..
python3 ../compare/testing/pdf_output.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PM  --metrics_file ../compare/results/metrics_PM.csv --output_pdf ../compare/testing/output_PM.pdf

python3 ../compare/testing/pdf_output.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PL  --metrics_file ../compare/results/metrics_PL.csv --output_pdf ../compare/testing/output_PL.pdf

# new
python3 ../compare/testing/pdf_output3.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3pm.txt --new_images ../compare/results/PM3  --metrics_file ../compare/results/metrics_PM3.csv --output_pdf ../compare/testing/output_PM3.pdf

python3 ../compare/testing/pdf_output3.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts3.txt --new_images ../compare/results/PL3  --metrics_file ../compare/results/metrics_PL3.csv --output_pdf ../compare/testing/output_PL3.pdf

python3 ../compare/testing/pdf_output3.py --image_folder ../compare/testing/ref1 --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PL_new_one --metrics_file ../compare/results/metrics_PL_new_one.csv --output_pdf ../compare/testing/output_PL_new_one.pdf

# new_full

python3 ../compare/testing/pdf_output4.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PM_full --metrics_file ../compare/results/metrics_PM_full.csv --output_pdf ../compare/testing/output_PM_full.pdf

python3 ../compare/testing/pdf_output4.py --image_folder ../compare/testing/references --prompt_file ../compare/testing/prompts4.txt --new_images ../compare/results/PL_full --metrics_file ../compare/results/metrics_PL_full.csv --output_pdf ../compare/testing/output_PL_full.pdf


```