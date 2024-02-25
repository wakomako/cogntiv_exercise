# Implicit Neural Representation of Images:


## Installation

install python 3.9.5

install requirements specified in requirements.txt
```bash
pip install -r requirements.txt
```


## Usage

The full report can be found here - [here](https://docs.google.com/document/d/1ZmYBut54J3MAaKl7c7gO465Isv4-5qu_Cm00xHpEEZ4/edit?usp=sharing)

Fit the baseline model on all images (Q1) [defalut arguments are as specified in the report]
```bash
python siren/scripts/represent_imgs.py --image_encoding=<image_encoding_version> --side_length=<image_side_length> 
```

Evaluate the baseline model on upsampling (Q2.a)  - generates 3 random samples
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/upsample.py
```

Evaluate the baseline model on interpolation (Q2.b) - generates 3 random samples 
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/interpolate_baseline.py
```


(Optional) Fit the OLS model on all images [defalut arguments are as specified in the report]
```bash
siren/ols/optimized_latent_siren.py
```

Evaluate the improved OLS model on interpolation (Q2.b)  - generates 3 random samples
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/interpolate_ols.py
```
