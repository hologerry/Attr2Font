# Attr2Font
## Introduction

This is the official PyTorch implementation of the Attribute2Font: Creating Fonts You Want From Attributes.

![Teaser](img/teaser.png)

## Abstract
Font design is now still considered as an exclusive privilege of professional designers, whose creativity is not possessed by existing software systems. Nevertheless, we also notice that most commercial font products are in fact manually designed by following specific requirements on some attributes of glyphs, such as italic, serif, cursive, width, angularity, etc. Inspired by this fact, we propose a novel model, Attribute2Font, to automatically create fonts by synthesizing visually pleasing glyph images according to user-specified attributes and their corresponding values. To the best of our knowledge, our model is the first one in the literature which is capable of generating glyph images in new font styles, instead of retrieving existing fonts, according to given values of specified font attributes. Specifically, Attribute2Font is trained to perform font style transfer between any two fonts conditioned on their attribute values. After training, our model can generate glyph images in accordance with an arbitrary set of font attribute values. Furthermore, a novel unit named Attribute Attention Module is designed to make those generated glyph images better embody the prominent font attributes. Considering that the annotations of font attribute values are extremely expensive to obtain, a semi-supervised learning scheme is also introduced to exploit a large number of unlabeled fonts. Experimental results demonstrate that our model achieves impressive performance on many tasks, such as creating glyph images in new font styles, editing existing fonts, interpolation among different fonts, etc.

## Model Architecture
![Architecture](img/att_arch.png)


## Prerequisites

* Linux
* CPU or NVIDIA GPU + CUDA cuDNN
* Python 3
* PyTorch 1.0+

## Get Started

### Installation
1. Install PyTorch, torchvison and dependencies from [https://pytorch.org](https://pytorch.org)
2. Clone this repo:
   ```shell
   git clone -b master --single-branch https://github.com/hologerry/Attr2Font
   cd Attr2Font
   ```
3. Download the offical pre-trained vgg19 model: [vgg19-dcbb9e9d.pth]('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'), and put it under this project root folder


### Datasets

[Download]() the datasets and put it into the `data/`:
```
data/
    explor_all/
        images/
        attributes.txt/
```
| This dataset is constructed by O’Donovan et al. *Exploratory Font Selection Using Crowdsourced Attributes*. TOG 2014



### Model Training
```
python main.py --phase train
```

### Model Testing
```
python main.py --phase test
```


### Model Interpolation
```
python main.py --phase test_interp --test_epoch EPOCH
```


## Citation:

If you use this code or find our work is helpful, please consider citing our work:
```
@article{WangSIGGRAPH2020, 
  author = {Yizhi Wang*, Yue Gao*, Zhouhui Lian}, 
  title = {Attribute2Font: Creating Fonts You Want From Attributes}, 
  journal = {ACM Trans. Graph.},
  year = {2020}
} 
```