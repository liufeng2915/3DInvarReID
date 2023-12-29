# Stage1: Pretraining


Install environment:
```
python setup.py develop
```
Please download the [SMPL models](https://smpl.is.tue.mpg.de) (version 1.0.0 compatible with Python 2.7, featuring 10 shape PCs) and relocate them to the corresponding locations:
```
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl lib/smpl/smpl_model/SMPL_NEUTRAL.pkl
```

## Prepare Training Data
We combine THuman2.0 (526 scans) and CAPE (3, 000 scans) to train our joint two-layer implicit model.

***1. THuman 2.0 dataset***

i) Download [THuman2.0 dataset](https://github.com/ytrock/THuman2.0-Dataset) following their instructions.

ii) Also download the corresponding SMPL parameters:
```
wget https://dataset.ait.ethz.ch/downloads/gdna/THuman2.0_smpl.zip
unzip THuman2.0_smpl.zip -d data/
```
iii) Run the pre-processing script to get ground truth occupancy:
```
python preprocess_thuman_data.py 
```

***2. CAPE dataset***

i) Download [CAPES dataset](https://cape.is.tue.mpg.de/) following their instructions.

ii) Run the pre-processing script to get ground truth occupancy:
```
python preprocess_cape_data.py 
```

## Training

```bash
python train.py expname=pretrain