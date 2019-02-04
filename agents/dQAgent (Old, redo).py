import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.deckops as dc
import random
import tensorflow as tf
import pickle

def convertToTensor(parameters):
    params = {}
    for key in parameters:
        params[key] = tf.convert_to_tensor(parameters[key])
    return params

f = open("Parameters_100epochs_dropout.pkl", 'rb')
params_100e_dropout = pickle.load(f)
params_100e_dropout = convertToTensor(params_100e_dropout)
f.close()



globalSess = tf.Session()
initializer = tf.global_variables_initializer()


def flattenSAGenerator(g, moves):
    if len(g.history) >= 2:
        histories = np.concatenate([
                        dc.handToExpanded(np.sum(g.history[-2::-2], axis=0)).reshape(75),
                        dc.handToExpanded(np.sum(g.history[-1::-2], axis=0)).reshape(75)])
    elif len(g.history) == 1:
        histories = np.concatenate([
                        np.zeros(75,),
                        dc.handToExpanded(g.history[0]).reshape(75)])
    else:
        histories = np.zeros((150,))
    hand = g.A_Hand if len(g.history)%2 == 0 else g.B_Hand
    SA = [np.concatenate([histories,
                     dc.handToExpanded(hand-move).reshape(75),
                     dc.handToExpanded(move).reshape(75)]) for move in moves]
    SA = np.stack(SA).T
    return SA

class dQParameterSetInstance:
    def __init__(self, paramFileName, tfSession = globalSess):
        params = pickle.load(open(paramFileName, 'rb'))

        self.n_x = params['W1'].shape[1]
        self.params = convertToTensor(params)
        self.n_layers = int(len(self.params)/2)

        self.params['X'] = tf.placeholder("float", [self.n_x, None])
        self.params['A0'] = self.params['X']

        for i in range(1, 1+self.n_layers):
            self.params['Z' + str(i)] = tf.add(
                tf.matmul(self.params['W' + str(i)], self.params['A'+str(i-1)]), self.params['b' + str(i)])
            self.params['A' + str(i)] = tf.nn.relu(self.params['Z' + str(i)]) if i < self.n_layers else \
                tf.nn.sigmoid(self.params['Z' + str(i)])

        self.sess = tfSession

    def predict(self, x):
        return self.sess.run(self.params['A'+str(self.n_layers)], feed_dict = {self.params['X']: x})


globalDQParamSetInstance = dQParameterSetInstance("Parameters_100epochs_dropout.pkl", globalSess)
dQ2params = dQParameterSetInstance("Parameters_M1_100epochs.pkl", globalSess)

def getDQA(paramFilename, exploration_prob=0):
    dQP = dQParameterSetInstance(paramFilename, globalSess)
    return DeepQAgent(exploration_prob=exploration_prob, predictor=dQP)

# generateSA is a function that takes a gamestate, a list of legal moves,
# and returns an arrays that represent (s, a) pairs to be fed into
# something to evaluate them
class DeepQAgent:
    def __init__(self, verbosity=0, exploration_prob=.1,
                 tfsession=globalSess, predictor = globalDQParamSetInstance, generateSA=flattenSAGenerator):
        self.name = "dQAgent"
        self.session = tfsession
        self.predictor = predictor
        self.generateSA = generateSA
        self.exploration_prob = exploration_prob
        self.verbosity = verbosity

    def getMove(self, g):
        moves = dc.getMovesFromGameState(g)
        X = flattenSAGenerator(g, moves)
        scores = self.predictor.predict(X)
        if np.all(scores==0):
            if len(moves) == 1:
                return moves[0]
            else:
                return random.choice(moves[1:])
        if random.random() < self.exploration_prob:
            # sample the moves based on score (0-1)
            moveInd = np.random.choice(np.arange(len(moves)), p=scores.reshape(-1)/np.sum(scores))
        else:
            moveInd = np.argmax(scores)
        return moves[moveInd]