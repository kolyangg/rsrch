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
python3 pulid_generate.py
```


## Metrics

```bash
cd persongen
conda create --name metrics python=3.10
conda activate metrics

# Install requirements
pip install -r requirements.txt

# Run inference
python3 pulid_generate2.py --image_folder ../compare/testing/images --prompt_file ../compare/testing/prompts.txt --output_dir ../compare/results/PL
```