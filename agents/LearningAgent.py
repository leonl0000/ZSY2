import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.deckops as dc
import numpy as np
import tensorflow as tf
from agents.staticAgents import Agent
tf_sess = tf.Session()

class LearningAgent(Agent):
    def __init__(self, config):
        self.config = config
        self.name = config.name
        self.addPlaceHolders()
        self.addEvalOp()
        self.addLossAndOptimizer()
        self.initialize()
        with tf.variable_scope(self.config.name):
            self.globalStep = tf.get_variable("global_step", [], dtype=tf.int64)
            self.sess.run(self.globalStep.assign(-1))
        self.saver = tf.train.Saver(var_list=self.getVariables(), max_to_keep=1000)
        self.exploration_prob = .1
        self.exploration_prob_holder = 0.1

    def gameStateConverter(self, g):
        hand = g.A_Hand if g.turn else g.B_Hand
        actions = np.vstack(dc.getMovesFromGameState(g))
        histAgent = np.concatenate([hist[0] for hist in g.history[-2::-2]], axis=0) if len(g.history) >= 2 \
            else np.zeros((0, 15))
        histOpp = np.concatenate([hist[0] for hist in g.history[-1::-2]], axis=0) if len(g.history) >= 1 \
            else np.zeros((0, 15))
        return histAgent, histOpp, hand, actions

    def setTest(self):
        self.exploration_prob_holder = self.exploration_prob
        self.exploration_prob = 0

    def setTrain(self):
        self.exploration_prob = self.exploration_prob_holder

    def getMove(self, g):
        raise NotImplementedError

    def getVariables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.config.name)

    def saveModel(self, step=None):
        ops = [
            tf.assign(self.v_Random_saveVar, self.v_Random),
            tf.assign(self.v_Greedy_saveVar, self.v_Greedy),
            tf.assign(self.v_OldAg_saveVar, self.v_OldAg),
            tf.assign(self.vs_saveVar, self.vs)
        ]
        _ = self.sess.run(ops)
        if step is None:
            self.globalStep = self.globalStep + 1
            step = self.sess.run(self.globalStep)
            self.saver.save(self.sess, os.path.join(self.config.model_path, self.config.name), global_step=step)
        else:
            self.sess.run(self.globalStep.assign(step))
            self.saver.save(self.sess, os.path.join(self.config.model_path, self.config.name), global_step=step)

    def loadModel(self, step=None):
        if step is None:
            f = tf.train.latest_checkpoint(self.config.model_path)
            if f is None:
                return False
        else:
            f = os.path.join(self.config.model_path, self.config.name + '-%d'%step)
            if not os.path.isfile(f):
                return False
        self.saver.restore(self.sess, f)
        self.v_Random, self.v_Greedy, self.v_OldAg, self.vs = self.sess.run([
            self.v_Random_saveVar,
            self.v_Greedy_saveVar,
            self.v_OldAg_saveVar,
            self.vs_saveVar,
        ])
        return True

    def addPlaceHolders(self):
        with tf.variable_scope(self.config.name):
            self.hands = tf.placeholder(shape=[None, 5, 15], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None, 5, 15], dtype=tf.float32)
            self.r = tf.placeholder(shape=[None], dtype=tf.float32)
            self.lr = tf.placeholder(shape=[], dtype=tf.float32)

    def addLossAndOptimizer(self):
        with tf.variable_scope(self.config.name):
            self.loss = self.config.lossfn(tf.squeeze(self.out), self.r)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    # Use self.config.name for scope
    def addEvalOp(self):
        self.out = None
        raise NotImplementedError

    def initWinRates(self):
        self.v_Random = 0.5
        self.v_Greedy = 0.5
        self.v_OldAg = 0.5
        self.vs = 0.5

        self.v_Random_placeholder = tf.placeholder(tf.float32, shape=(), name="v_Random")
        self.v_Greedy_placeholder = tf.placeholder(tf.float32, shape=(), name="v_Greedy")
        self.v_OldAg_placeholder = tf.placeholder(tf.float32, shape=(), name="v_OldAg")
        self.vs_placeholder = tf.placeholder(tf.float32, shape=(), name="vs")

        with tf.variable_scope(self.config.name):
            self.v_Random_saveVar = tf.get_variable("v_Random", (), tf.float32, trainable=True)
            self.v_Greedy_saveVar = tf.get_variable("v_Greedy", (), tf.float32, trainable=True)
            self.v_OldAg_saveVar = tf.get_variable("v_OldAg", (), tf.float32, trainable=True)
            self.vs_saveVar = tf.get_variable("vs", (), tf.float32, trainable=True)


    def add_summary(self):
        """
        Tensorboard stuff
        """
        # add placeholders from the graph
        # logging

        self.merged = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("v_Random", self.v_Random_placeholder),
            tf.summary.scalar("v_Greedy", self.v_Greedy_placeholder),
            tf.summary.scalar("v_OldAg", self.v_OldAg_placeholder),
            tf.summary.scalar("vs", self.vs_placeholder)])
        self.file_writer = tf.summary.FileWriter(self.config.results_path)

    def recordSummary(self, summary, ind):
        if ind < 0:
            return
        summary = tf.Summary().FromString(summary)
        for entry in summary.value:
            if entry.tag[-1].isdigit():
                if entry.tag[-2] == '_':
                    entry.tag = entry.tag[:-2]
                if entry.tag[-2].isdigit() and entry.tag[-3] == '_':
                    entry.tag = entry.tag[:-3]
        self.file_writer.add_summary(summary, ind)

    def initialize(self):
        self.sess = tf_sess
        self.initWinRates()
        self.add_summary()
        vars_list = tf.global_variables()
        excludes = ['global_step']

        for ex in excludes:
            vars_list = [v for v in vars_list if ex not in v.name]
        init = tf.initializers.variables(vars_list)
        self.sess.run(init)

