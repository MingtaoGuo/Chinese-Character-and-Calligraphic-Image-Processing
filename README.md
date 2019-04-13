# Chinese-Character-and-Calligraphic-Image-Processing
Some interesting method like style transfer, GAN, deep neural networks for Chinese character and calligraphic image processing

# 1. Classification for 30 different Fonts 
### Dataset: https://pan.baidu.com/s/1LVcfD_M-pI3Vkscsb6hlow  Extract code: lqp2
### Part of the dataset
![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/samefont.jpg)

### Fonts classification by GoogLeNet
|Loss|Test accuracy|Confusion matrix|
|-|-|-|
|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/loss.png)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/acc.png)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/confusion_matrix.jpg)|

### Feature visualizing
![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/tsne.jpg)

# 2. Style transfer for calligraphic image
![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/n_style_transfer.jpg)
Content image dataset: http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

### Style fusion
|||||||
|-|-|-|-|-|-|
|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result1.gif)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result2.gif)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result3.gif)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result11.gif)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result22.gif)|![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/result33.gif)|

### zi2zi
The method of this application, we just simply use pix2pix to generate another style of Chinese character.

dataset: https://pan.baidu.com/s/1JagVbA8p-Bn5OnoOErJAyQ extract code: 2vku 

![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/zi2zi.jpg)

# 3. Calligraphic image denoising
![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/denoise.jpg)

# 4. Chinese character inpainting
![](https://github.com/MingtaoGuo/Chinese-Character-and-Calligraphic-Image-Processing/blob/master/IMGS/inpainting.jpg)
# Acknowledgement
These great calligraphy works are written by my teacher Prof. Zhang.

# Author
1. Mingtao Guo 2. Xinran Wen

# Reference
[1]. Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.

[2]. Dumoulin V, Shlens J, Kudlur M. A learned representation for artistic style[J]. Proc. of ICLR, 2017, 2.

[3]. Isola P, Zhu J Y, Zhou T, et al. Image-to-image translation with conditional adversarial networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1125-1134.
