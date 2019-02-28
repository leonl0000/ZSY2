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
# from agents.dQAgent_old import *
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
from multiprocessing import Pool
def runXGamesAndSaveData(X, buffer=Buffer()):
    num_batches = int(X/buffer.index_every)
    bar = ProgBar("Games", 50)
    bar.start()
    # p = Pool(3)
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

def stepGame(turn_start, names):
    d = np.random.permutation(dc.deck)
    A = dc.cardsToHand(d[:18])
    B = dc.cardsToHand(d[18:36])
    history = []
    gameStates = [GameState(names[0], names[1], A, B, history, False, 0, 1-turn_start)]
    moves = dc.getOpeningMoves(A)
    return [A, B], names, moves, history, gameStates, turn_start

def takeMove(stepGame, move, i):
    hands, names, moves, history, gameStates, turn = stepGame
    history.append((move, turn))
    hands[turn] -= move
    gameStates.append(GameState(names[0], names[1], hands[0], hands[1], history, not np.any(hands[turn]), i, turn))
    moves = dc.listLegalCounters(hands[1-turn], move)
    return hands, names, moves, history, gameStates, 1-turn

def multiGame(agentA, agentB, num_games=100, turn_start=0):
    names = [agentA.name, agentB.name]
    agents = [agentA, agentB]
    SA_functions = [agentA.getFunction_SAFromGameState(), agentB.getFunction_SAFromGameState()]
    stepGames = [stepGame(turn_start, names) for _ in range(num_games)]
    turn = turn_start
    endGames = []
    while len(stepGames) > 0:
        sa = [SA_functions[turn](sg[3], sg[sg[-1]], sg[2]) for sg in stepGames]
        saBInds = [0] * (len(sa)+1)
        for i in range(1, len(sa)+1):
            saBInds[i] = saBInds[i-1] + sa[i-1].shape[0]
        sa = np.concatenate(sa, axis=0)
        scores = agents[turn].sess.run(agents[turn].out, feed_dict={agents[turn].sa: sa})
        for i in range(len(saBInds)-1):
            if random.random() < agentB.exploration_prob:
                move = random.choice(stepGames[i][2])
            else:
                move = stepGames[i][2][np.argmax(scores[saBInds[i]:saBInds[i+1]])]
            stepGames[i] = takeMove(stepGames[i], move)

    return stepGames, moves