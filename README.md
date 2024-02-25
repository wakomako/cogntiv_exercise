# Implicit Neural Representation of Images:


## Installation

install python 3.9.5

install requirements specified in requirements.txt
```bash
pip install -r requirements.txt
```


## Usage

Fit the baseline model on all images (Q1) [defalut arguments are as specified in the report]
```bash
python siren/scripts/represent_imgs.py --image_encoding=<image_encoding_version> --side_length=<image_side_length> 
```

Evaluate the baseline model on upsampling (Q2.a)
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/upsample.py
```

Evaluate the baseline model on interpolation (Q2.b)
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/interpolate_baseline.py
```

Evaluate the improved OLS model on interpolation (Q2.b)
[runs on a pretrained model as specified in the report]
```bash
python siren/scripts/interpolate_ols.py
```
