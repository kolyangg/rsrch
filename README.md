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

# Create a pdf output
cd ..
python3 ../compare/testing/pdf_output.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PM  --metrics_file ../compare/results/metrics_PM.csv --output_pdf ../compare/testing/output_PM.pdf

python3 ../compare/testing/pdf_output.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --new_images ../compare/results/PL  --metrics_file ../compare/results/metrics_PL.csv --output_pdf ../compare/testing/output_PL.pdf

```