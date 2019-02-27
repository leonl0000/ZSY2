import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.deckops as dc
import random
import numpy as np
import tensorflow as tf
import pickle
from agents.staticAgents import Agent
import agents.utils.SAConverters as SAConverters
from multiprocessing import Pool


class LearnedAgent(Agent):
    def __init__(self):
        self.model_path = os.path.join("Models", self.name)
        self.results_path = os.path.join("Results", self.name)
        self.addPlaceHolders()
        self.out = self.get_eval_SA_OP()
        self.addLossAndOptimizer()
        self.initialize()
        self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

    def saveModel(self, step):
        self.saver.save(self.sess, os.path.join(self.model_path, self.name), global_step=step)

    def loadModel(self, step=None):
        if step is None:
            f = tf.train.latest_checkpoint(self.model_path)
        else:
            f = os.path.join(self.model_path, self.name + '-%d'%step)
        self.saver.restore(self.sess, f)

    def addPlaceHolders(self):
        raise NotImplementedError

    # While playing the game, use g to generate
    # the appropriate S, A from parts of the game state
    def SAFromGamestate(self, history, hand, actions):
        raise NotImplementedError

    # While learning, use the buffer and indices to generate
    # the appropriate SA and targets for learning
    def SATargetsFromSample(self, sample):
        raise NotImplementedError

    def addLossAndOptimizer(self):
        raise NotImplementedError

    # Use self.name for scope
    def get_eval_SA_OP(self):
        raise NotImplementedError

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        # self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)



class BasicDenseNet(LearnedAgent):
    def __init__(self, name="BasicDenseNet"):
        self.name = name
        self.exploration_prob = .1

        super().__init__()

    def getMove(self, g):
        actions = dc.getMovesFromGameState(g)
        SA = self.SAFromGameState(g.history, g.A_Hand if g.turn else g.B_Hand, actions)
        scores = self.sess.run(self.out, feed_dict={self.sa: SA})
        if random.random() < self.exploration_prob:
            return random.choice(actions)
        else:
            return actions[np.argmax(scores)]

    def addPlaceHolders(self):
        self.sa = tf.placeholder(shape=[None, 300], dtype=tf.float32)
        self.r = tf.placeholder(shape=[None], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

    def SAFromGameState(self, history, hand, actions):
        return SAConverters.DenseNetSAFromGameState(history, hand, actions)

    def getFunction_SAFromGameState(self):
        return SAConverters.DenseNetSAFromGameState

    def SATargetsFromSample(self, sample):
        return SAConverters.DenseNetSAFromSample(sample)

    def getFunction_SATargetsFromSample(self):
        return SAConverters.DenseNetSAFromSample

    def train(self, buffer, num_iters=1, minibatch_size=2048, increment_sample=True):
        for _ in range(num_iters):
            sample, ind, buf_size = buffer.getSample(sample_size=minibatch_size, increment_sample=increment_sample)
            SA, targets = self.SATargetsFromSample(sample)
            progbar = tf.keras.utils.Progbar(int(buf_size/minibatch_size), 50, 1, 1, ['loss'])
            i = 0
            while True:
                i+=1
                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.sa:SA, self.r:targets})
                progbar.update(i, [('loss', loss)])
                if ind + minibatch_size > buf_size:
                    break
                sample, ind, buf_size = buffer.getSample(sample_size=minibatch_size, increment_sample=increment_sample)
                SA, targets = self.SATargetsFromSample(sample)

    def trainWithPool(self, buffer, num_iters=1, minibatch_size=2048, increment_sample=True, poolsize=5, set_size=128):
        for it in range(num_iters):
            buf_size = len(buffer.isWinner)
            set_sizes = [set_size for _ in range(int(buf_size/minibatch_size/set_size))] + [buf_size % (minibatch_size*set_size)]
            progbar = tf.keras.utils.Progbar(len(set_sizes), 50, 1, 1, ['loss'])
            for i, _set_size in enumerate(set_sizes):
                sample_set = [buffer.getSample(sample_size=minibatch_size, increment_sample=increment_sample)[0] for _ in range(_set_size)]
                p = Pool(poolsize)
                SAT = p.map(self.getFunction_SATargetsFromSample(), sample_set)
                p.close()
                p.join()
                for SA, targets in SAT:
                    loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.sa:SA, self.r:targets})
                    progbar.update(i, [('loss', loss)])

    def get_eval_SA_OP(self):
        with tf.variable_scope(self.name):
            out = tf.layers.dense(self.sa, 300, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="H1")
            out = tf.layers.dense(out, 40, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="H3")
            out = tf.layers.dense(out, 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name="output")
            return out

    def addLossAndOptimizer(self):
        with tf.variable_scope(self.name):
            self.loss = tf.losses.log_loss(tf.squeeze(self.out), self.r)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)





