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
| forward_batch1      |        41 ms       |    8 ms   |
| backward_batch1     |        51 ms       |   11 ms   |
| forward_batch16     |       532 ms       |   36 ms   |
| backward_batch16    |       695 ms       |   96 ms   |


[^nocudnn]: When turn on cudnn, the memory consuming of mobilenet would increase to unbelievable level. You may try.


### Transfer normal net to mobilenet

    I write a script [transfer2Mobilenet.py] to convert normal net to mobilenet format. You may try too.
    usage: python ./transfer2Mobilenet.py sourceprototxt targetprototxt [--midbn nobn --weight_filler msra --activation ReLU]    ["--origin_type" means the depthwise convolution layer's type will be "Convolution" instead of "DepthwiseConvolution"]

    The "transferTypeToDepthwiseConvolution.py" will be used for changing the depthwise convolution layer's type from "Convolution" to "DepthwiseConvolution".
