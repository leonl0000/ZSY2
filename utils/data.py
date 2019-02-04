import numpy as np
import sys
import os
homeDir = os.path.dirname(os.path.dirname(__file__))
dataDir = os.path.join(homeDir, "data")

sys.path.append(homeDir)

import pickle
import h5py
import utils.deckops as dc




def saveObject(gameStates, fname):
    f = open(fname, 'wb+')
    pickle.dump(gameStates, f)
    f.close()

def loadObject(fname):
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
    gameStates = loadObject(fname)
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
    if len(X_A.shape) == 2:
        return X_A, X_B, Y_A, Y_B
    else:
        return X_A[:,:,0].T, X_B[:,:,0].T, Y_A[:,:,0].T, Y_B[:,:,0].T

def convertY(Y, discount):
    Y_ = (Y[0]*discount**Y[1]).reshape(1,-1)
    return (Y_[0] + 1)/2