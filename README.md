# IHE-Net:Hidden feature discrepancy fusion and triple consistency training for semi-supervised medical image segmentation

**Authors:**
Mengyi Ju,Bing Wang,Zutong Zhao,Shiyin Zhang,Shuo Yang,Zhihong Wei
# Usage
We provide code, models, data_split and training weights for three datasets.{ [ISIC2017](https://challenge.isic-archive.com/data/),[ISIC2018](https://challenge.isic-archive.com/data/),and PH2}.


#### 1. Create conda environment and activate it:
```
conda create -n joey python=3.8
conda activate joey
```

#### 2. Clone the repo.;

```
https://github.com/joey-AI-medical-learning/IHE-Net.git
```
#### 3. Put the data in './IHE-Net/data/';
#### 4. Training;
```
python train_ISIC.py    #for ISIC training
``` 
#### 5. Testing;
```
python test_ISIC.py    #for ISIC testing
```
# Citation
# Acknowledgements:
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [Laplacian-Former](https://github.com/xmindflow/Laplacian-Former.git), [VM-UNet](https://github.com/JCruan519/VM-UNet.git). Thanks for these authors for their valuable works.
