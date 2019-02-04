import numpy as np
import time
import random
import pickle

import utils.deckops as dc
from utils.misc import timer
from utils.data import *

import h5py
import os
from agents.staticAgents import RandomAgent, GreedyAgent, HumanAgent
from agents.dQAgent import *

randomAgent = RandomAgent()
greedyAgent = GreedyAgent()
humanAgent = HumanAgent()

# contains all info about a game at a point
# can be printed
class GameState:
    def __init__(self, A_Name, B_Name, A_Hand, B_hand, history):
        self.A_Name = A_Name
        self.B_Name = B_Name
        self.A_Hand = A_Hand
        self.B_Hand = B_hand
        self.history = history #list of moves

    def __str__(self):
        s = ""
        if len(self.history) == 0:
            s += "==================\n%s%s%s\n" % (self.A_Name[:4], " "*10, self.B_Name[:4])
            for i in range(15):
                s += "%s|%s          %s|%s\n" % (dc.handStringArr[i],
                                                 str(self.A_Hand[0,i]) if self.A_Hand[0,i]!= 0 else " ",
                                                 dc.handStringArr[i],
                                                 str(self.B_Hand[0,i]) if self.B_Hand[0,i]!= 0 else " ")
        else:
            s += "%s====\n%s%s%s\n" % (" "*14*(len(self.history)%2==0), self.A_Name[:4], " "*10, self.B_Name[:4])
            for i in range(15):
                st = dc.handStringArr[i]+"|"+str(self.history[-1][0, i]) if self.history[-1][0, i] != 0 else "    "
                s += "%s|%s   %s   %s|%s\n" % (dc.handStringArr[i],
                                               str(self.A_Hand[0,i]) if self.A_Hand[0,i]!= 0 else " ", st,
                                               dc.handStringArr[i],
                                               str(self.B_Hand[0,i]) if self.B_Hand[0,i]!= 0 else " ")
        return s


# ~ 13.4 ms
# A always goes first, but with a 50% chance will take an empty move
# Returns list of gameStates and whether or not A won
def game(agentA = randomAgent, agentB = randomAgent, verbose = 0):
    d = np.random.permutation(dc.deck)
    A = dc.cardsToHand(d[:18])
    B = dc.cardsToHand(d[18:36])
    gameStates = []
    history = []
    gameStates.append(GameState(agentA.name, agentB.name, A, B, history))
    if verbose:
        print(gameStates[-1])

    # First move is A's, but A will 50% of the time pass [that is, play emptyMove]
    #   to simulate randomly choosing a first player
    if random.random() > 0.5:
        history.append(agentA.getMove(gameStates[-1]))
    else:
        history.append(dc.emptyMove)
    A -= history[-1]
    gameStates.append(GameState(agentA.name, agentB.name, A, B, history))
    if verbose:
        print(gameStates[-1])

    i = 0
    while np.sum(A) > 0 and np.sum(B) > 0:
        if i%2 == 0:
            history.append(agentB.getMove(gameStates[-1]))
            B -= history[-1]
        else:
            history.append(agentA.getMove(gameStates[-1]))
            A -= history[-1]
        i += 1
        gameStates.append(GameState(agentA.name, agentB.name, A, B, history))
        if verbose:
            print(gameStates[-1])
    return gameStates, np.sum(A)<np.sum(B)

def stdTest(paramFileName, numGames = 10000):
    dQP = dQParameterSetInstance(paramFileName, globalSess)
    dQA = DeepQAgent(predictor = dQP, exploration_prob=0)
    vg = np.sum([game(dQA, greedyAgent)[1] for _ in range(numGames)])
    vr = np.sum([game(dQA, randomAgent)[1] for _ in range(numGames)])
    print("VS Greedy: %.3f%%\tVS Random: %.3f%%"%(100.*vg/numGames, 100.*vr/numGames))
    metric = (1-1.*vg/numGames) * (1-1.*vr/numGames) * 10000
    print("Standard metric: %.2f"%metric)

# def runXGamesRetFinals(paramFileName, numGames = 20000, exploration_prob=0.1):
#     dQP = dQParameterSetInstance(paramFileName, globalSess)
#     dQA = DeepQAgent(predictor=dQP, exploration_prob=exploration_prob)
#     print("Simulating...")
#     finalGameStates = [game(dQA, dQA)[0][-1] for _ in range(numGames)]
#     print("Converting...")
#     X_A, X_B, Y_A, Y_B = gameStatesToLabeledData_1(finalGameStates)
#     print("Stacking...")
#     X_A = np.stack(X_A)
#     X_B = np.stack(X_B)
#     Y_A = np.stack(Y_A)
#     Y_B = np.stack(Y_B)
#     return X_A, X_B, Y_A, Y_B

def runXGamesDeepQ(paramFileName, saveFileName, numGames = 20000, exploration_prob=0.1):
    dQP = dQParameterSetInstance(paramFileName, globalSess)
    dQA = DeepQAgent(predictor = dQP, exploration_prob=exploration_prob)
    print("Simulating...")
    finalGameStates = [game(dQA, dQA)[0][-1] for _ in range(numGames)]
    print("Converting...")
    X_A, X_B, Y_A, Y_B = gameStatesToLabeledData_1(finalGameStates)
    print("Saving...")
    X_A = np.stack(X_A)
    X_B = np.stack(X_B)
    Y_A = np.stack(Y_A)
    Y_B = np.stack(Y_B)
    f = h5py.File(saveFileName, "w")
    XAset = f.create_dataset("X_A", X_A.shape, compression="gzip")
    XAset[...] = X_A
    XBset = f.create_dataset("X_B", X_B.shape, compression="gzip")
    XBset[...] = X_B
    YAset = f.create_dataset("Y_A", Y_A.shape, compression="gzip")
    YAset[...] = Y_A
    YBset = f.create_dataset("Y_B", Y_B.shape, compression="gzip")
    YBset[...] = Y_B
    f.close()
    print("done.")

def runXGamesSaveLastGameStates(fname, x=10000, agentA = randomAgent, agentB = randomAgent):
    finalGameStates = []
    timesAWon = 0
    for i in range(x):
        gs = game(agentA, agentB, 0)
        finalGameStates.append(gs[0][-1])
        timesAWon += gs[1]
    print("A won %f"%(1.*timesAWon/x))
    saveGameStates(finalGameStates, fname)

def oneMillionGames():
    for i in range(10):
        fname = "T100k_%d.pkl"%(i)
        runXGamesSaveLastGameStates(fname, 100000)

def saveGameStates(gameStates, fname):
    f = open(fname, 'wb+')
    pickle.dump(gameStates, f)
    f.close()

def loadGameStates(fname):
    f = open(fname, 'rb')
    return pickle.load(f)





"""
Take a list of the final game states, turns into labeled data

Each game has a history of moves. Each game will produce a set of datapoints as long
    as the history. Each point in the history will become an (s,a) pair, where s
    is the sum of the history up to that point for the player in question, sum for the
    opponent, and the hand they have after the move they take, and a is the move they take.

Each of these 4 things can be represented by a hand. With the expanded hand representation
    each hand will be a (5,15) array with exactly 15 coordinates being 1 and the rest being
    0. These will be flattenned and concatenated into a (300,1) matrix.

Y will be a (2,1) matrix. The first coordinate represents if that player won in the end, the
    second represents how many moves away the end is. The first will be the actual label,
    the second will be used later if I want to add a gamma.
"""
def gameStatesToLabeledData_1(finalGameStates):
    X_A = []
    X_B = []
    Y_A = []
    Y_B = []
    for gameState in finalGameStates:
        A_Hand = gameState.A_Hand
        B_Hand = gameState.B_Hand
        if len(gameState.history)%2 == 1:
            hands = [A_Hand, B_Hand]
            X = [X_A, X_B]
            Y = [Y_A, Y_B]
        else:
            hands = [B_Hand, A_Hand]
            X = [X_B, X_A]
            Y = [Y_B, Y_A]

        played = [np.sum(gameState.history[-1::-2], axis=0),
                  np.sum(gameState.history[-2::-2], axis=0)]

        for i in range(0, len(gameState.history)):
            hand = hands[i%2]
            move = gameState.history[-i-1]
            played[i%2] -= move
            x = np.concatenate((dc.handToExpanded(played[i%2]).reshape(75,1),
                                dc.handToExpanded(played[(i+1)%2]).reshape(75, 1),
                                dc.handToExpanded(hand).reshape(75, 1),
                                dc.handToExpanded(move).reshape(75, 1)), axis=0)
            y = np.array([[1-2*(i%2)], [int(i/2)]])
            X[i%2].append(x)
            Y[i%2].append(y)
            hands[i%2] += move
    return X_A, X_B, Y_A, Y_B

def gameStatesFileToDataFile_1(fname):
    gameStates = loadGameStates(fname)
    X_A, X_B, Y_A, Y_B = gameStatesToLabeledData_1(gameStates)
    print("conversion done")

    X_A = np.stack(X_A)
    X_B = np.stack(X_B)
    Y_A = np.stack(Y_A)
    Y_B = np.stack(Y_B)

    pos = fname.find('.')
    if pos == -1:
        outname = fname+".h5"
    else:
        outname = fname[:pos]+".h5"
    f = h5py.File(outname, "w")

    XAset = f.create_dataset("X_A", X_A.shape, compression="gzip")
    XAset[...] = X_A
    XBset = f.create_dataset("X_B", X_B.shape, compression="gzip")
    XBset[...] = X_B
    YAset = f.create_dataset("Y_A", Y_A.shape, compression="gzip")
    YAset[...] = Y_A
    YBset = f.create_dataset("Y_B", Y_B.shape, compression="gzip")
    YBset[...] = Y_B
    f.close()
    print("save done")

def gameStatesFileToDataFile_1_Dir(dirname):
    fnames = [x for x in os.listdir(dirname) if x[-4:]=='.pkl']
    i=1
    for fname in fnames:
        gameStatesFileToDataFile_1(os.path.join(dirname,fname))
        print("%d of %d done"%(i, len(fnames)))

def dataFileToLabeledData_1(fname):
    f = h5py.File(fname, 'r')
    X_A = f["X_A"][...]
    X_B = f["X_B"][...]
    Y_A = f["Y_A"][...]
    Y_B = f["Y_B"][...]
    f.close()
    return X_A[:,:,0].T, X_B[:,:,0].T, Y_A[:,:,0].T, Y_B[:,:,0].T