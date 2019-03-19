import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.LearningAgent import *
import random
import tensorflow as tf


def getMoveMean(scores, actions):
    scores = sum(scores)
    return actions[np.argmax(scores)]

class ComboAgent(Agent):
    def __init__(self, agents, kind="Mean", name="ComboAgent"):
        self.name = "_".join([name, kind, str(len(agents))])
        self.agents = agents
        self.sess = agents[0].sess
        self.kind = kind
        cc = tf.concat([agent.out for agent in agents], axis=1)
        if kind == "Mean":
            self.out = tf.reduce_mean(cc, axis=1)
        elif kind == "Max":
            self.out = tf.reduce_max(cc, axis=1)
        elif kind == "Min":
            self.out = tf.reduce_min(cc, axis=1)
        elif kind == 'Plurality' or kind == "STV":
            self.out = tf.reduce_mean(cc, axis=1), cc

    def gameStateConverter(self, g):
        hand = g.A_Hand if g.turn else g.B_Hand
        actions = np.vstack(dc.getMovesFromGameState(g))
        histAgent = np.concatenate([hist[0] for hist in g.history[-2::-2]], axis=0) if len(g.history) >= 2 \
            else np.zeros((0, 15))
        histOpp = np.concatenate([hist[0] for hist in g.history[-1::-2]], axis=0) if len(g.history) >= 1 \
            else np.zeros((0, 15))
        return histAgent, histOpp, hand, actions


    def getMove(self, g):
        histAg, histOp, hand, actions = self.gameStateConverter(g)
        histAg = dc.handToExpanded(np.sum(histAg, axis=0))[None,:].repeat(len(actions), axis=0)
        histOp = dc.handToExpanded(np.sum(histOp, axis=0))[None,:].repeat(len(actions), axis=0)
        hands = dc.handToExpandedBatch(hand-actions)
        _actions = dc.handToExpandedBatch(actions)
        SA = np.concatenate([histAg, histOp, hands, _actions], axis=2)
        feed_dict = {}
        for agent in self.agents:
            feed_dict[agent.sa] = SA
        out = self.sess.run(self.out, feed_dict=feed_dict)
        if self.kind == "Max" or self.kind == "Mean" or self.kind == "Min":
            return actions[np.argmax(out)]
        elif self.kind == "Plurality" or self.kind == "STV":
            means, scores = out
            return actions[np.argmax(means)]


    def getManyMoves(self, gs):
        converted = [self.gameStateConverter(g) for g in gs]
        histsAg, histsOp, hands, actions = [[c[i] for c in converted] for i in range(4)]
        splits = [0] * (len(gs)-1)
        for i, acts in enumerate(actions[:-1]):
            splits[i] = splits[i-1] + len(acts)
        histsAg = np.vstack([dc.handToExpanded(np.sum(histsAg[i], axis=0))[None,:].repeat(len(actions[i]), axis=0)
                             for i in range(len(actions))])
        histsOp = np.vstack([dc.handToExpanded(np.sum(histsOp[i], axis=0))[None,:].repeat(len(actions[i]), axis=0)
                             for i in range(len(actions))])
        hands = np.vstack([dc.handToExpandedBatch(hands[i]-actions[i]) for i in range(len(gs))])
        _actions = dc.handToExpandedBatch(np.vstack(actions))
        SA = np.concatenate([histsAg, histsOp, hands, _actions], axis=2)
        feed_dict = {}
        for agent in self.agents:
            feed_dict[agent.sa] = SA
        out = self.sess.run(self.out, feed_dict=feed_dict)
        if self.kind == "Max" or self.kind == "Mean" or self.kind == "Min":
            scores = np.split(out, splits)
            manyMoves = [actions[i][np.argmax(scores[i])].reshape(1, 15) for i in range(len(gs))]
            return manyMoves
        elif self.kind == "Plurality" or self.kind == "STV":
            pass

    def setTest(self):
        pass
    def setTrain(self):
        pass

