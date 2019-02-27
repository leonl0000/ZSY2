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
from agents.dQAgent_old import *
from progbar import ProgBar

randomAgent = RandomAgent()
greedyAgent = GreedyAgent()
humanAgent = HumanAgent()

# contains all info about a game at a point
# can be printed
class GameState:
    def __init__(self, A_Name, B_Name, A_Hand, B_hand, history, end, ind, turn):
        self.A_Name = A_Name
        self.B_Name = B_Name
        self.A_Hand = A_Hand
        self.B_Hand = B_hand
        self.history = history #list of moves
        self.end = end
        self.ind = ind
        self.turn = turn

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
                st = dc.handStringArr[i]+"|"+str(self.history[-1][0][0, i]) if self.history[-1][0][0, i] != 0 else "    "
                s += "%s|%s   %s   %s|%s\n" % (dc.handStringArr[i],
                                               str(self.A_Hand[0,i]) if self.A_Hand[0,i]!= 0 else " ", st,
                                               dc.handStringArr[i],
                                               str(self.B_Hand[0,i]) if self.B_Hand[0,i]!= 0 else " ")
        return s

    def getExpandedHand(self):
        return dc.handToExpanded(self.A_Hand) if self.turn == 0 else dc.handToExpanded(self.B_Hand)



# ~ 6 ms for static agents
# Returns list of gameStates and whether or not A won
def game(agentA = randomAgent, agentB = randomAgent, verbose = 0):
    d = np.random.permutation(dc.deck)
    A = dc.cardsToHand(d[:18])
    B = dc.cardsToHand(d[18:36])
    gameStates = []
    history = []

    agents = [agentA, agentB]
    hands = [A, B]

    turn = 1 if random.random() > .5 else 0
    gameStates.append(GameState(agentA.name, agentB.name, A, B, history, False, 0, 1-turn))
    if verbose:
        print(gameStates[-1])
    i = 1
    while True:
        history.append((agents[turn].getMove(gameStates[-1]), turn))
        hands[turn] -= history[-1][0]
        done = not np.any(hands[turn])
        gameStates.append(GameState(agentA.name, agentB.name, A, B, history, done, i, turn))
        if done:
            break
        turn = 1 - turn
        if verbose:
            print(gameStates[-1])
        i += 1
    return gameStates, 1-turn

def gameStatesToData(endgame):
    gameStates, A_won = endgame
    gs = gameStates[-1]
    expanded_states = [None] * gs.ind
    expanded_actions = [None] * gs.ind
    actions = [None] * gs.ind
    step = list(range(1, gs.ind+1))
    remaining_steps = list(range(gs.ind-1, -1, -1))
    isWinner = ([1] if gs.ind%2 == 1 else []) + [0,1] * int(gs.ind/2)
    winner_hand = np.zeros((1, 15)).astype(np.int8)
    loser_hand = np.zeros((1, 15)).astype(np.int8) + (gameStates[-2].B_Hand if A_won else gameStates[-2].A_Hand)

    expanded_states[-1] = dc.handToExpanded(winner_hand)
    winner_action = gs.history[-1][0]
    expanded_actions[-1] = dc.handToExpanded(winner_action)
    actions[-1] = winner_action
    winner_hand += winner_action
    for i in range(2, gs.ind, 2):
        loser_action = gs.history[-i][0]
        expanded_states[-i] = dc.handToExpanded(loser_hand)
        expanded_actions[-i] = dc.handToExpanded(loser_action)
        actions[-i] = loser_action
        loser_hand += loser_action

        winner_action = gs.history[-i - 1][0]
        expanded_states[-i-1] = dc.handToExpanded(winner_hand)
        expanded_actions[-i-1] = dc.handToExpanded(winner_action)
        actions[-i-1] = winner_action
        winner_hand += winner_action
    if expanded_states[0] is None:
        loser_action = gs.history[0][0]
        expanded_states[0] = dc.handToExpanded(loser_hand)
        expanded_actions[0] = dc.handToExpanded(loser_action)
        actions[0] = loser_action
        loser_hand += loser_action
    assert(np.sum(winner_hand) == 18 and np.sum(loser_hand) == 18)
    return expanded_states, expanded_actions, actions, step, remaining_steps, isWinner

import itertools
# Only use multiprocessing pool if DQA is not imported
# Switch between the 2 ways to run games commented below
# from multiprocessing import Pool
def runXGamesAndSaveData(X, buffer=Buffer()):
    num_batches = int(X/buffer.index_every)
    bar = ProgBar("Games", 50)
    bar.start()
    p = Pool(10)
    for i in range(num_batches):
        bar.percent = int(i*100/num_batches)
        # endgames = p.map(game, [randomAgent] * buffer.index_every)
        endgames = [game() for j in range(buffer.index_every)]
        # data = p.map(gameStatesToData, endgames)
        data = [gameStatesToData(endgame) for endgame in endgames]
        buffer.addToBuffer(data, buffer.index_every)
    bar.stop()
    bar.join()
    return buffer


# TODO: Multi game not yet implemented!!!
def _flattenSAGenerator(gmoves):
    return flattenSAGenerator(gmoves[0], gmoves[1])

from multiprocessing import Pool
def multiGame(agentA, agentB, num_games=100):
    p = Pool(5)
    Decks = p.map(np.random.permutation, [dc.deck]*num_games)
    A_ind = int(num_games/2)
    A_hands = p.map(dc.cardsToHand, [Decks[i][:18] for i in range(num_games)])
    B_hands = p.map(dc.cardsToHand, [Decks[i][:18] for i in range(num_games)])
    historys = [[]] * num_games
    gameStates = [[GameState(agentA.name, agentB.name, A_hands[i], B_hands[i], historys[i], False, 0, 1 if i < A_ind else 0)] for i in range(num_games)]
    turns = [0] * A_ind + [1] * (A_ind + num_games % 2)
    movesA = p.map(dc.getOpeningMoves, A_hands[:A_ind])
    movesB = p.map(dc.getOpeningMoves, B_hands[A_ind:])
    moves = movesA + movesB
    flattenedSA_A = p.map(_flattenSAGenerator, [(gameStates[i][-1], movesA[i]) for i in range(len(movesA))])
    flattenedSA_B = p.map(_flattenSAGenerator, [(gameStates[i][-1], movesB[i]) for i in range(len(movesA))])
    flattenedSA_A = np.concatenate(flattenedSA_A, axis=1)
    p.close()
    p.join()
    # print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n' + str(flattenedSA_A.shape) + '\n\n\n\n\n\n\n\n\n\n')
    scores = agentA.predictor.predict(flattenedSA_A)
    return scores



def stdTest(paramFileName, numGames = 10000):
    dQP = dQParameterSetInstance(paramFileName, globalSess)
    dQA = DeepQAgent(predictor = dQP, exploration_prob=0)
    vg = np.sum([game(dQA, greedyAgent)[1] for _ in range(numGames)])
    vr = np.sum([game(dQA, randomAgent)[1] for _ in range(numGames)])
    print("VS Greedy: %.3f%%\tVS Random: %.3f%%"%(100.*vg/numGames, 100.*vr/numGames))
    metric = (1-1.*vg/numGames) * (1-1.*vr/numGames) * 10000
    print("Standard metric: %.2f"%metric)


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