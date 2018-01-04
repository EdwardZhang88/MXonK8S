import os
os.system('sudo ln /dev/null /dev/raw1394')
os.environ.update({"PS_VERBOSE": "1"})
import mxnet as mx
import logging
import numpy as np
import urllib
import gzip
import struct

role = os.environ.get("DMLC_ROLE", "")

def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

if role == 'scheduler':
    #scheduler
    print("scheduler is running...")
elif role == 'server':
    #sever
    print("server is running...")
elif role == 'worker':
    #worker
    print("worker is running...")
    
    # Load data
    path='/workdir/data/'
    (train_lbl, train_img) = read_data(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
    
    print("Data Loaded")
    kv_store = mx.kv.create('dist_async')
    
    print("Synced with scheduler")
    batch_size = 100
    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)


    # Create a place holder variable for the input data
    data = mx.sym.Variable('data')
    # Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
    # into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer
    fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    # Apply relu to the output of the first fully-connnected layer
    act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

    # The second fully-connected layer and the according activation function
    fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
    act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

    # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
    # The softmax and loss layer
    mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')


    logging.getLogger().setLevel(logging.DEBUG)

    model = mx.model.FeedForward(
    symbol = mlp,       # network structure
    num_epoch = 10,     # number of data passes for training 
    learning_rate = 0.1, # learning rate of SGD
    ctx = mx.gpu() # train on gpu 
    )

    model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200),# output progress for each 200 data batches
    kvstore=kv_store 
    )
else:
    raise ValueError("invalid role %s" % (role))

