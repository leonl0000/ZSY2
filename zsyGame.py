import numpy as np
import random

import utils.deckops as dc
from utils.data import *

import h5py
import os
from agents.staticAgents import RandomAgent, GreedyAgent, HumanAgent
from tensorflow.python.keras.utils import Progbar

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
    num_batches = int(X/1000)
    bar = Progbar(num_batches, 50)
    for i in range(num_batches):
        endgames = [game() for j in range(buffer)]
        data = [gameStatesToData(endgame) for endgame in endgames]
        buffer.addToBuffer(data)
        bar.update(i)
    return buffer

def stepGame(turn_start, names):
    d = np.random.permutation(dc.deck)
    A = dc.cardsToHand(d[:18])
    B = dc.cardsToHand(d[18:36])
    history = []
    gameStates = [GameState(names[0], names[1], A, B, history, False, 0, 1-turn_start)]
    moves = dc.getOpeningMoves(A if turn_start == 0 else B)
    return [[A, B], moves, history, gameStates]

def takeMove(stepGame, move, turn, names):
    hands, moves, history, gameStates = stepGame
    history.append((move, turn))
    hands[turn] -= move
    done = not np.any(hands[turn])
    i = gameStates[-1].ind + 1
    gameStates.append(GameState(names[0], names[1], hands[0], hands[1], history, done, i, turn))
    if done:
        return gameStates, 1-turn
    else:
        moves = dc.getMoves(hands[1-turn], move)
        return [hands, moves, history, gameStates]

import time
def multiGame(agentA, agentB, num_games=100, print_stats=True):
    names = [agentA.name, agentB.name]
    agents = [agentA, agentB]
    A_ind = int(num_games/2)
    stepGames = [stepGame(0, names) for _ in range(A_ind)]
    stepGames2 = [stepGame(1, names) for _ in range(num_games-A_ind)]
    endGames = []
    gs = [g[3][-1] for g in stepGames2]
    manyMoves = agents[1].getManyMoves(gs)
    for i in range(len(stepGames2)-1, -1, -1):
        stepGames2[i] = takeMove(stepGames2[i], manyMoves[i], 1, names)
        if len(stepGames2[i]) == 2:
            endGames.append(stepGames2[i])
            del stepGames2[i]
    stepGames += stepGames2
    turn = 0
    while len(stepGames) > 0:
        gs = [g[3][-1] for g in stepGames]
        manyMoves = agents[turn].getManyMoves(gs)
        for i in range(len(stepGames)-1, -1, -1):
            stepGames[i] = takeMove(stepGames[i], manyMoves[i], turn, names)
            if len(stepGames[i]) == 2:
                endGames.append(stepGames[i])
                del stepGames[i]
        turn = 1-turn
    if print_stats:
        wins = np.sum([g[1] for g in endGames])
        rat = wins / num_games
        interval = 200 * np.sqrt(rat * (1 - rat) / num_games)
        rat *= 100
        print("Wins ratio: %.2f%% +- %.2f%%" % (rat, interval))

    return endGames