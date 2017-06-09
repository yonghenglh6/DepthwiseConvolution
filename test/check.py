import caffe
import numpy as np
import argparse



parser=argparse.ArgumentParser()

parser.add_argument('net1')
parser.add_argument('net2')
parser.add_argument('--method',type=int,default=1)
args=parser.parse_args()

# caffeutil.caffe_switch('/home/liuhao/tool/caffe_withoutcudnn/python')
#args.net1='fb_dw.pt'
#args.net2='fb_cv.pt'




caffe.set_mode_gpu()


net0 = caffe.Net(args.net1,caffe.TEST)
net1 = caffe.Net(args.net2,caffe.TEST)


# def generate_data(shape):
#     mul=1
#     for s in shape:
#         mul*=s
#     return np.arange(mul).reshape(shape)

def generate_data(shape):
    return np.random.normal(size=shape)


# init_param
for pmkey in net0.params:
    if pmkey.find('bn')==-1:
        for index_pm, pmblob in enumerate(net0.params[pmkey]):
            shape = pmblob.data.shape
            gparams = generate_data(shape=shape)
            net0.params[pmkey][index_pm].data[...] = gparams
            net1.params[pmkey][index_pm].data[...] = gparams

# init data and label
datashape = net0.blobs['data'].data.shape
labelshape = net0.blobs['label'].data.shape
gdata = generate_data(shape=datashape)

net0.blobs['data'].data[...] = gdata
net1.blobs['data'].data[...] = gdata

# let label is not true
net0.forward()
net1.forward()
glabel = np.where(np.argmax(net0.blobs['prob'].data[...],axis=1) == 0, 1, 0)
glabel=glabel.reshape(glabel.shape[0],1,1,1)
net0.blobs['label'].data[...] = glabel
net1.blobs['label'].data[...] = glabel

# begin to test
net0.forward()
net1.forward()
net0.backward()
net1.backward()


def almost_same(arr1, arr2,method=args.method):
    arr1=np.nan_to_num(arr1)
    arr2 = np.nan_to_num(arr2)
    if method==0:
        arr1=np.abs(arr1)
        arr2=np.abs(arr2)
        error=np.abs(arr1-arr2)
        return np.all(error<np.where(arr1>arr2,arr1*0.1,arr2*0.1))
    elif method==1:
        return np.all(np.abs(arr1-arr2)<0.1)
    elif method==2:
        if np.abs(np.mean(arr1)-np.mean(arr2))<np.abs(np.mean(arr1))*0.1:
            return False
        if np.abs(np.max(arr1)-np.max(arr2))<np.abs(np.max(arr1))*0.1:
            return False
        if np.abs(np.var(arr1)-np.var(arr2))<np.abs(np.var(arr1))*0.1:
            return False
        return True


# test blob
for bbkey in net0.blobs:
    bb0 = net0.blobs[bbkey]
    bb1 = net1.blobs[bbkey]
    assert almost_same(bb0.data, bb1.data), 'blob data [%s] is not consistent' % (bbkey)
    print 'blob data [%s] pass' % (bbkey)

# test blob_diff
bkeys=net0.blobs.keys()
bkeys.reverse()
for bbkey in bkeys:
    bb0 = net0.blobs[bbkey]
    bb1 = net1.blobs[bbkey]
    assert almost_same(bb0.diff, bb1.diff), 'blob diff [%s] is not consistent' % (bbkey)
    print 'blob diff [%s] pass.' % (bbkey)

pkeys=net0.params.keys()
pkeys.reverse()
# test weight_diff
for pmkey in pkeys:
    for index_pm, _ in enumerate(net0.params[pmkey]):
        pm0 = net0.params[pmkey][index_pm]
        pm1 = net1.params[pmkey][index_pm]
        assert almost_same(pm0.diff, pm1.diff), 'weight_diff [%s %d] is not consistent' % (pmkey,index_pm)
        print 'weight_diff [%s %d] pass' % (pmkey,index_pm)

print 'The output is:'
print net0.blobs['prob'].data[...]
print net1.blobs['prob'].data[...]
print 'All passed. But please assure no nan in the output.'