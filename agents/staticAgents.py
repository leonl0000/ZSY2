import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.deckops as dc
import random
import numpy as np

antiface = {'3':0, '4':1, '5':2, '6':3, '7':4, '8':5, '9':6, '10':7, '11':8, 'j':8, '12':9, 'q':9,
            '13':10, 'k':10, '14':11, 'a':11, '15':12, '2':12, '16':13, 'jk':13, 'jb':13, '17':14, 'jr': 14}

class RandomAgent:
    def __init__(self):
        self.name = "Random"

    def getMove(self, g):
        return random.choice(dc.getMovesFromGameState(g))



class GreedyAgent:
    def __init__(self):
        self.name = "Greedy"

    def getMove(self, g):
        moves = dc.getMovesFromGameState(g)
        return moves[0] if len(moves) == 1 else moves[1]

class HumanAgent:
    def __init__(self):
        self.name = "Human"

    def getMove(self, g):
        if len(g.history) != 0:
            lastMove = g.history[-1]
        else:
            lastMove = dc.emptyMove
        if len(g.history)%2 == 1:
            hand = g.B_Hand
            opCardCount = len(dc.handToCards(g.A_Hand))
        else:
            hand = g.A_Hand
            opCardCount = len(dc.handToCards(g.B_Hand))
        _ = os.system('cls')
        moves = dc.getMovesFromGameState(g)
        while True:
            print("Opponent Card Count: " + str(opCardCount))
            print("Current pattern:\n"+dc.stringHand(lastMove))
            print("Your Hand:\n"+dc.stringHand(hand))
            act = input("Enter Move: ").strip().lower().split(',')
            actCards = [antiface[a] for a in act] if(act[0] != '') else []
            move = dc.cardsToHand(actCards)
            eq = [np.all(move == m) for m in moves]
            if not np.any(eq):
                _ = os.system('cls')
                print("ILLEGAL MOVE\n\n")
            else:
                break
        return move
