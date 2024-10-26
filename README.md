# Semi-supervised medical image segmentation via hidden feature discrepancy learning across heterogeneous encoders

**Authors:**
Mengyi Ju,Shiyin Zhang,Zutong Zhao,Bing Wang,Shuo Yang,Zhihong Wei
# Usage
We provide code, models, data_split and training weights for three datasets.{ [ISIC2017](https://challenge.isic-archive.com/data/),[ISIC2018](https://challenge.isic-archive.com/data/),and PH2}.

#### 1. Clone the repo.;

```
https://github.com/joey-AI-medical-learning/IHE-Net.git
```
#### 2. Put the data in './DD-Net/data/';
#### 3. Training;
```
cd DD-Net/code
python train_ddnet_acdc.py    #for ACDC training
python train_ddnet_2D.py    #for BrainMRI, COVID-19 and LUNA16 training
```
#### 4. Testing;
```
python test_ACDC.py    #for ACDC testing
python test_2D.py    #for BrainMRI, COVID-19 and LUNA16 testing
```
# Citation
# Acknowledgements:
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net/blob/main/README.md). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.
