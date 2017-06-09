# Depthwise Convolutional Layer 

### Introduction
This is a personal caffe implementation of mobile convolution layer. For details, please read the original paper:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)

### How to build
1. Merge the caffe folder in the repo with your own caffe.
$ cp -r $REPO/caffe/* $YOURCAFFE/
2. Then make. 
$ cd $YOURCAFFE && make

### Usage
Replacing the type of mobile convolution layer with "DepthwiseConvolution" is all.
Please refer to the example/Withdw_MN_train_128_1_train.prototxt, which is altered from
- [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)



