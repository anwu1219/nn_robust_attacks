## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import pickle
import gzip
import urllib.request

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

import onnx
from onnx import numpy_helper

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    def __init__(self, restore, layers, neurons, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        #model_onnx = onnx.load(restore)
        #weights = model_onnx.graph.initializer
        #isBias = numpy_helper.to_array(weights[0]).shape == (neurons,)
        #allWeights = []
        #allBiases = []
        #for weight in weights:
        #    if isBias:
        #        allBiases.append(numpy_helper.to_array(weight))
        #        isBias = False
        #    else:
        #        allWeights.append(numpy_helper.to_array(weight).T)
        #        isBias = True

        #allLayers = []
        #for i in range(len(allBiases)):
        #    if i == 0:
        #        allLayers.append((np.eye(784), np.ones(784) - 0.5))
        #    allLayers.append((allWeights[i], allBiases[i]))

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        for i in range(layers):
            model.add(Dense(neurons, use_bias=True))
            model.add(Activation('relu'))
        model.add(Dense(10))


        #index = 0
        #weights = model.get_weights()
        #for j, layer in enumerate(model.layers):
        #    if type(layer) == Dense:
        #        layer.set_weights(allLayers[index])
        #        index += 1
        model.load_weights(restore)

        self.model = model

        '''
        #m = MNIST()
        #print(model.predict(m.test_data[0].reshape(1,28,28,1)))
        weights = []
        biases = []
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                weights.append(layer.get_weights()[0])
                biases.append(layer.get_weights()[1])

        with open("temp.nnet", 'w') as out_file:
            out_file.write("{},784,10,784,\n".format(layers+1))
            out_file.write("784,{},10,\n".format(",".join(layers * ["256"])))
            out_file.write('0\n')
            for i in range(784):
                out_file.write("-0.5,")
            out_file.write("\n")
            for i in range(784):
                out_file.write("0.5,")
            out_file.write("\n")
            for i in range(785):
                out_file.write("0.0,")
            out_file.write("\n")
            for i in range(785):
                out_file.write("1.0,")
            out_file.write("\n")
            for i in range(len(biases)):
                weight = list(weights[i].T)
                for line in weight:
                    out_file.write(",".join(list(map(str, line))))
                    out_file.write(",\n")
                for ele in list(biases[i].flatten()):
                    out_file.write("{},".format(ele))
                    out_file.write('\n')
        '''

    def predict(self, data):
        return self.model(data)
