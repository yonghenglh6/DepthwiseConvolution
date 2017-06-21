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


### GPUPerformance on example net

| GPUPerformance      | Origin[^nocudnn]   | Mine      |
| ------------------- |:------------------:| ---------:|
| forward_batch1      |        40 ms       |    3 ms   |
| backward_batch1     |        87 ms       |    8 ms   |
| forward_batch16     |       512 ms       |   13 ms   |
| backward_batch16    |      1155 ms       |   68 ms   |


[^nocudnn]: When turn on cudnn, the memory consuming of mobilenet would increase to unbelievable level. You may try.


### Transfer normal net to mobilenet
I write a script[transfer2mobilenet.py] to convert normal net to mobilenet format. You may try too.

