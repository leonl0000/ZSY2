import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.newTestAgent import *


class ConvNetAgent(LearnedAgent):
    def __init__(self, name="ConvNetAgent"):
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
        self.sa = tf.placeholder(shape=[None, 5, 60, 1], dtype=tf.float32)
        self.r = tf.placeholder(shape=[None], dtype=tf.float32)
        self.lr = tf.placeholder(shape=[], dtype=tf.float32)

    def SAFromGameState(self, history, hand, actions):
        return SAConverters.ConvNetSAFromGameState(history, hand, actions)

    def getFunction_SAFromGameState(self):
        return SAConverters.ConvNetSAFromGameState

    def SATargetsFromSample(self, sample):
        return SAConverters.ConvNetSAFromSample(sample)

    def getFunction_SATargetsFromSample(self):
        return SAConverters.ConvNetSAFromSample

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


    def get_eval_SA_OP(self):
        with tf.variable_scope(self.name):
            out = tf.layers.conv2d(self.sa, 32, 3, padding='valid', activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="Conv1")
            out = tf.layers.conv2d(out, 64, 3, padding='valid', activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="Conv2")
            out = tf.layers.dense(tf.layers.flatten(out), 100, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name="Dense1")
            out = tf.layers.dense(tf.layers.flatten(out), 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name="output")
            return out

    def addLossAndOptimizer(self):
        with tf.variable_scope(self.name):
            self.loss = tf.losses.log_loss(tf.squeeze(self.out), self.r)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)





