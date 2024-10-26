# Semi-supervised medical image segmentation via hidden feature discrepancy learning across heterogeneous encoders

**Authors:**
Mengyi Ju,Shiyin Zhang,Zutong Zhao,Bing Wang,Shuo Yang,Zhihong Wei
# Usage
We provide code, models, data_split and training weights for three datasets.{ [ISIC2017](https://challenge.isic-archive.com/data/),[ISIC2018](https://challenge.isic-archive.com/data/),and PH2}.

#### 1. Clone the repo.;

```
https://github.com/joey-AI-medical-learning/IHE-Net.git
```
#### 2. Put the data in './IHE-Net/data/';
#### 3. Training;
```
python train_ISIC.py    #for ISIC training
``` 
#### 4. Testing;
```
python test_ISIC.py    #for ISIC testing
```
# Citation
# Acknowledgements:
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [Laplacian-Former](https://github.com/xmindflow/Laplacian-Former.git), [VM-UNet](https://github.com/JCruan519/VM-UNet.git). Thanks for these authors for their valuable works.
