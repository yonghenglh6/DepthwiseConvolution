import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf.text_format import Merge
import argparse


def get_input_channel(net):
    layers = net.layer
    input_channel_map = {}
    owner = {}  # which layer own the special blob

    for i_layer, layer in enumerate(layers):
        bottoms = layer.bottom
        tops = layer.top
        for top in tops:
            find_in_bottom = False
            for bottom in bottoms:
                if bottom == top:
                    find_in_bottom = True
                    break
            if not find_in_bottom:
                owner[top] = layer

        if layer.type == "Convolution":
            assert len(bottoms) == 1, 'Is there a mistake? Convolution layer normally do not has multi-input.'
            bottom = bottoms[0]
            input_channel = -1
            to_find_root = [bottom]
            while len(to_find_root) > 0:
                root = to_find_root.pop(0)
                if root not in owner:
                    continue
                if owner[root].type == "Convolution":
                    input_channel = owner[root].convolution_param.num_output
                    break
                for r_b in owner[root].bottom:
                    to_find_root.append(r_b)
            if input_channel == -1:
                print '{} cannot find the input channel,please pay attention to it.'.format(layer.name)
            input_channel_map[layer.name] = input_channel

    return input_channel_map


def add_mobilenet_block(net, originlayer, input_channel, args):
    last_num_output = originlayer.convolution_param.num_output
    last_top_name = originlayer.top[0]
    baselayername = originlayer.name

    def clearedblobs(blobs):
        for b in blobs:
            blobs.remove(b)
        return blobs

    # convx layer
    convx_layer = net.layer.add()
    convx_layer.CopyFrom(originlayer)
    convx_layer.name = baselayername + "_depthwise"
    if not args.origin_type:
        convx_layer.type = "DepthwiseConvolution"
    clearedblobs(convx_layer.top).append(baselayername + "_3x3")
    convx_layer.convolution_param.group = input_channel
    convx_layer.convolution_param.num_output = input_channel

    # mid bn layer
    if args.midbn != 'nobn':
        convx_bn_layer = net.layer.add()
        convx_bn_layer.name = baselayername + "_3x3_bn"
        convx_bn_layer.type = "BatchNorm"
        convx_bn_layer.bottom.append(baselayername + "_3x3")
        convx_bn_layer.top.append(baselayername + "_3x3")
        if args.midbn == 'bn_nouseglobalstats':
            convx_bn_layer.batch_norm_param.use_global_stats = False
        elif args.midbn == 'bn_useglobalstats':
            convx_bn_layer.batch_norm_param.use_global_stats = True
        else:
            assert False, 'midbn is not configured right'
        # scale
        convx_scale_layer = net.layer.add()
        convx_scale_layer.name = baselayername + "_3x3_scale"
        convx_scale_layer.type = "Scale"
        convx_scale_layer.bottom.append(baselayername + "_3x3")
        convx_scale_layer.top.append(baselayername + "_3x3")
        convx_scale_layer.scale_param.bias_term = True

    # activation layer
    activation = args.activation
    convx_relu_layer = net.layer.add()
    convx_relu_layer.name = baselayername + "_3x3_" + activation.lower()
    convx_relu_layer.type = activation
    convx_relu_layer.bottom.append(baselayername + "_3x3")
    convx_relu_layer.top.append(baselayername + "_3x3")

    # 1x1 layer
    conv1_layer = net.layer.add()
    conv1_layer.CopyFrom(originlayer)
    conv1_layer.name = baselayername + "_1x1"
    clearedblobs(conv1_layer.bottom).append(baselayername + "_3x3")
    clearedblobs(conv1_layer.top).append(last_top_name)
    conv1_param = conv1_layer.convolution_param
    conv1_param.num_output = last_num_output
    clearedblobs(conv1_param.pad).append(0)
    clearedblobs(conv1_param.kernel_size).append(1)
    clearedblobs(conv1_param.stride).append(1)

    # fix the weight_filler
    if args.weight_filler != 'origin':
        convx_layer.convolution_param.weight_filler.type = args.weight_filler
        conv1_layer.convolution_param.weight_filler.type = args.weight_filler


def create_mobile_net(net, args):
    input_channel_map = get_input_channel(net)
    mobile_net = caffe_pb2.NetParameter()
    for layer in net.layer:
        if layer.type == "Convolution":
            conv_param = layer.convolution_param

            # cannot find the input channel
            if input_channel_map[layer.name] == -1:
                mobile_net.layer.add().CopyFrom(layer)
                continue

            # origin 1x1 layer, we should skip it.
            if (len(conv_param.kernel_size) == 1 and conv_param.kernel_size[0] == 1) or (
                                len(conv_param.kernel_size) == 2 and conv_param.kernel_size[0] == 1 and
                            conv_param.kernel_size[
                                1] == 1) or len(conv_param.kernel_size) == 0:
                mobile_net.layer.add().CopyFrom(layer)
                continue

            # now this layer can be transferred legally.
            add_mobilenet_block(mobile_net, layer, input_channel_map[layer.name], args)
        else:
            mobile_net.layer.add().CopyFrom(layer)
    return mobile_net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source_prototxt')
    parser.add_argument('target_prototxt')

    # midbn:  nobn  bn_nouseglobalstats  bn_useglobalstats
    parser.add_argument('--midbn', type=str, default='nobn')
    parser.add_argument('--weight_filler', type=str, default='origin')
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--origin_type', action='store_true')
    args = parser.parse_args()

    net = caffe_pb2.NetParameter()
    Merge(open(args.source_prototxt, 'r').read(), net)

    mobile_net = create_mobile_net(net, args)

    with open(args.target_prototxt, 'w') as tf:
        tf.write(str(mobile_net))

    print 'If you use the depressed input format,\'input: "data"\', it will be lost in new net, please check it.'
