import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.LearningAgent import *
import random


class DenseNetAgent(LearningAgent):
    def __init__(self, config):
        super().__init__(config)

    def getMove(self, g):
        histAg, histOp, hand, actions = self.gameStateConverter(g)
        histAg = dc.handToExpanded(np.sum(histAg, axis=0))[None,:].repeat(len(actions), axis=0)
        histOp = dc.handToExpanded(np.sum(histOp, axis=0))[None,:].repeat(len(actions), axis=0)
        hands = dc.handToExpandedBatch(hand-actions)
        _actions = dc.handToExpandedBatch(actions)
        SA = np.concatenate([histAg, histOp, hands, _actions], axis=2)
        scores = self.sess.run(self.out, feed_dict={self.sa: SA})
        actions = actions.reshape(-1, 1, 15)
        if random.random() < self.exploration_prob:
            return random.choice(actions)
        else:
            return actions[np.argmax(scores)]

    def getScores(self, g):
        histAg, histOp, hand, actions = self.gameStateConverter(g)
        histAg = dc.handToExpanded(np.sum(histAg, axis=0))[None, :].repeat(len(actions), axis=0)
        histOp = dc.handToExpanded(np.sum(histOp, axis=0))[None, :].repeat(len(actions), axis=0)
        hands = dc.handToExpandedBatch(hand - actions)
        _actions = dc.handToExpandedBatch(actions)
        SA = np.concatenate([histAg, histOp, hands, _actions], axis=2)
        scores = self.sess.run(self.out, feed_dict={self.sa: SA})
        actions = actions.reshape(-1, 1, 15)
        return scores.squeeze(axis=1), actions

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
        scores = self.sess.run(self.out, feed_dict={self.sa: SA})
        scores = np.split(scores, splits)
        explores = [random.random() < self.exploration_prob for _ in range(len(gs))]
        manyMoves = [None] * len(gs)
        for i, explore in enumerate(explores):
            if explore:
                manyMoves[i] = random.choice(actions[i]).reshape(1, 15)
            else:
                manyMoves[i] = actions[i][np.argmax(scores[i])].reshape(1, 15)
        return manyMoves

    def getManyScores(self, gs):
        converted = [self.gameStateConverter(g) for g in gs]
        histsAg, histsOp, hands, actions = [[c[i] for c in converted] for i in range(4)]
        splits = [0] * (len(gs) - 1)
        for i, acts in enumerate(actions[:-1]):
            splits[i] = splits[i-1] + len(acts)
        histsAg = np.vstack([dc.handToExpanded(np.sum(histsAg[i], axis=0))[None, :].repeat(len(actions[i]), axis=0)
                             for i in range(len(actions))])
        histsOp = np.vstack([dc.handToExpanded(np.sum(histsOp[i], axis=0))[None, :].repeat(len(actions[i]), axis=0)
                             for i in range(len(actions))])
        hands = np.vstack([dc.handToExpandedBatch(hands[i] - actions[i]) for i in range(len(gs))])
        _actions = dc.handToExpandedBatch(np.vstack(actions))
        SA = np.concatenate([histsAg, histsOp, hands, _actions], axis=2)
        scores = self.sess.run(self.out, feed_dict={self.sa: SA})
        scores = np.split(scores, splits)
        scores = [s.squeeze(axis=1) for s in scores]
        return scores, actions

    def getScoresFromSample(self, sample):
        buf_expanded_states, buf_expanded_actions, _, buf_history_ag, buf_history_op, _, buf_isWinner = sample
        hist_ag = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_ag]))
        hist_op = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_op]))
        prev_move = [ho[-1] if len(ho) > 0 else np.zeros((1,15)) for ho in buf_history_op]
        moves = [dc.getMoves(dc.expandedToHand(buf_expanded_states[i]), prev_move[i]) for i in range(len(prev_move))]
        return moves



    def addPlaceHolders(self):
        with tf.variable_scope(self.config.name):
            self.sa = tf.placeholder(shape=[None, 5, 60], dtype=tf.float32)
        super().addPlaceHolders()






    def trainOnSample(self, sample, summaryStepsIndex=-1):
        buf_expanded_states, buf_expanded_actions, _, buf_history_ag, buf_history_op, _, buf_isWinner = sample
        hist_ag = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_ag]))
        hist_op = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_op]))
        SA = np.concatenate([hist_ag, hist_op, buf_expanded_states, buf_expanded_actions], axis=2)
        fd = {self.sa: SA,
              self.r: buf_isWinner,
              self.v_Random_placeholder: self.v_Random,
              self.v_Greedy_placeholder: self.v_Greedy,
              self.v_OldAg_placeholder: self.v_OldAg,
              self.vs_placeholder: self.vs}
        loss, _, summary = self.sess.run([self.loss, self.optimizer, self.merged], feed_dict=fd)
        self.recordSummary(summary, summaryStepsIndex)
        return loss

    def train(self, buffer, epochs=1, minibatch_size=4096, increment_sample=True):
        for _ in range(epochs):
            numBatches = buffer.numBatchs(minibatch_size)
            summarySteps = (np.arange(100) * numBatches/100).astype(np.int)
            summaryStepsIndex = 0
            progbar = tf.keras.utils.Progbar(numBatches, 50, 1, 1, ['loss'])
            for i in range(numBatches):
                sample = buffer.getSample(sample_size=minibatch_size, increment_sample=increment_sample)
                summarize = i == summarySteps[summaryStepsIndex]
                loss = self.trainOnSample(sample, summaryStepsIndex if summarize else -1)
                summaryStepsIndex += summarize
                progbar.update(i+i, [('loss', loss)])

    def addEvalOp(self):
        with tf.variable_scope(self.config.name):
            out = tf.layers.flatten(self.sa)
            for i, layer in enumerate(self.config.layers):
                out = tf.layers.dense(out, layer, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="H%d"%(i+1))
            out = tf.layers.dense(out, 1, activation=tf.nn.sigmoid, reuse=tf.AUTO_REUSE, name="output")
            self.out = out

    def addLossAndOptimizer(self):
        with tf.variable_scope(self.config.name):
            self.loss = tf.losses.log_loss(tf.squeeze(self.out), self.r)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)