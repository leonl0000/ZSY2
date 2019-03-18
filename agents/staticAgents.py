import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils.deckops as dc
import random
import numpy as np

antiface = {'3':0, '4':1, '5':2, '6':3, '7':4, '8':5, '9':6, '10':7, '11':8, 'j':8, '12':9, 'q':9,
            '13':10, 'k':10, '14':11, 'a':11, '15':12, '2':12, '16':13, 'jk':13, 'jb':13, '17':14, 'jr': 14}

class Agent:
    def name(self):
        raise NotImplementedError

    def getMove(self, g):
        moves = dc.getMovesFromGameState(g)
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self):
        self.name = "Random"

    def getMove(self, g):
        return random.choice(dc.getMovesFromGameState(g))

    def getManyMoves(self, gs):
        return [random.choice(dc.getMovesFromGameState(g)) for g in gs]



class GreedyAgent(Agent):
    def __init__(self):
        self.name = "Greedy"

    def getMove(self, g):
        moves = dc.getMovesFromGameState(g)
        return moves[0] if len(moves) == 1 else moves[1]

    def getManyMoves(self, gs):
        moves = [dc.getMovesFromGameState(g) for g in gs]
        return [m[0] if len(m) == 1 else m[1] for m in moves]

class HumanAgent(Agent):
    def __init__(self):
        self.name = "Human"

    def getMove(self, g):
        if len(g.history) != 0:
            lastMove = g.history[-1][0]
        else:
            lastMove = dc.emptyMove
        hand = g.A_Hand if g.turn else g.B_Hand
        opHand = g.B_Hand if g.turn else g.A_Hand
        opCardCount = np.sum(opHand)
        _ = os.system('cls')

        moves = dc.getMovesFromGameState(g)
        # print(moves)
        while True:
            print("Opponent Card Count: " + str(opCardCount))
            print("Current pattern:\n"+dc.stringHand(lastMove))
            print("Your Hand:\n"+dc.stringHand(hand))
            act = input("Enter Move: ").strip().lower().split(',')
            if act[0] == "peek":
                print(g.B_Name, g.B_Hand, '\n', g.A_Name, g.A_Hand)
            elif act[0] == 'lll':
                moves = dc.getMovesFromGameState(g)
                for m in moves:
                    print(m)
            else:
                validMove = True
                for a in act:
                    if a != '' and a not in antiface:
                        validMove = False
                # val = validMove
                if act[-1] == '' and len(act)>1:
                    act = act[:-1]
                if validMove:
                    actCards = [antiface[a] for a in act] if(act[0] != '') else []
                    move = dc.cardsToHand(actCards)
                    validMove = validMove and np.any([np.all(move == m) for m in moves])
                if not validMove:
                    _ = os.system('cls')
                    print("ILLEGAL MOVE\n\n")
                    # moves = dc.getMovesFromGameState(g)
                    # print('val', val)
                    # print('MOVE', move)
                    # for m in moves:
                    #     print(m)
                else:
                    break
        return move
