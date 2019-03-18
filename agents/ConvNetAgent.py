import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.LearningAgent import *
from agents.DenseNetAgent import *
import random


class ConvNetAgent(DenseNetAgent):
    def __init__(self, config):
        super(DenseNetAgent, self).__init__(config)

    def addEvalOp(self):
        with tf.variable_scope(self.config.name):
            out = tf.reshape(self.sa, [-1, 5, 60, 1])
            convCount = sum(type(layer) == tuple for layer in self.config.layers)
            for i, layer in enumerate(self.config.layers[:min(convCount, 2)]):
                out = tf.layers.conv2d(out, layer[1], layer[0], padding='valid', activation=self.config.activations[i],
                                       reuse=tf.AUTO_REUSE, name="Conv%d" % (i+1))
            if convCount > 2:
                out = tf.squeeze(out, axis=1)
            for i, layer in enumerate(self.config.layers[2:convCount]):
                out = tf.layers.conv1d(out, layer[1], layer[0], padding='valid', activation=self.config.activations[i],
                                       reuse=tf.AUTO_REUSE, name="Conv%d" % (i + 3))
            out = tf.layers.flatten(out)
            for i, layer in enumerate(self.config.layers[convCount:]):
                out = tf.layers.dense(out, layer, activation=self.config.activations[convCount+i],
                                      reuse=tf.AUTO_REUSE, name="Dense%d" % (i+1))
            out = tf.layers.dense(out, 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name="output")
            self.out = out





