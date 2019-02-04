import numpy as np
import os
import random

'''
    GLOBALS
'''
deck = np.array([i/4+(i==53) for i in range(54)]).astype(np.int64)
handStringArr = [' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9', '10', ' J', ' Q', ' K', ' A', ' 2', 'JB', 'JR']
handString = ' 3| 4| 5| 6| 7| 8| 9|10| J| Q| K| A| 2|JB|JR'

OpeningMovesSaveFileName = os.path.join(os.path.dirname(__file__), "openingMoves.npz")
f = open(OpeningMovesSaveFileName, 'rb')
OpM = np.load(f)
f.close()

emptyMove = np.zeros((1,15)).astype(np.int64)


"""
    There are 3 ways to represent hands:
        - hand: (1,15) array representing how many of each card there is
        - cards: (n,) array with the card values
        - expanded: (5,15) array, each column is a one-hot array of whether
            there is 0, 1, 2, 3, or 4 of that card
    "cards" are dealt by shuffling and turned into "hand"
    "hand" is the standard representation, as it's easy to add and subtract them
    "expanded" is what will be used for deep learning because having, for example,
        2 Aces cannot be valued as double of having 1 Ace, as 2 Aces opens up
        different kinds of moves than 1 Ace.
"""
def stringHand(hand):
    h = [1*(i != 0)*' %d'%i + 1*(i == 0)*'  ' for i in hand[0]]
    h = ' '.join(h)
    return (handString + '\n' + h)

# ~ 11.4 microsec
d = np.random.permutation(deck)

def cardsToHand(cards):
    return np.bincount(cards, minlength=15).reshape(1,15)

def handToCards(hand):
    return np.array(sum([[i]*hand[0,i] for i in range(15)], []))

# ~ 3.3 microsecs
def expandedToHand(expanded):
    return np.dot([[4,3,2,1,0]], expanded)

def handToExpanded(hand):
    return np.unpackbits(2**hand.astype(np.uint8), axis=0)[3:].astype(np.int64)




# ~ 150 microsecs [only tested on 1 hand]
Bombs = 4*np.eye(15)[:13].astype(np.int64)
def listLegalCounters(hand, move):
    legalMoves = [emptyMove]
    highCard = np.nonzero(move)[1][-1]
    possibleMoves = [np.roll(move, i) for i in range(1, 15-highCard)]
    legalMoves += [move for move in possibleMoves if (hand >= move).all()]
    # If the move is NOT a bomb, then any bombs can beat it
    if not (Bombs-move).any(axis=1).all():
        legalMoves += list(Bombs[np.all(hand >= Bombs, axis=1)].reshape(-1, 1, 15))
    return legalMoves

# ~ 460 [only tested on 1 hand]
def getOpeningMoves(hand):
    return list(OpM[None, np.all(hand>=OpM, axis=1)].reshape(-1,1,15))

def getMoves(hand, move=np.zeros((1,15))):
    if np.all(move == 0):
        return getOpeningMoves(hand)
    else:
        return listLegalCounters(hand, move)

def getMovesFromGameState(g):
    if len(g.history) == 0:
        return getMoves(g.A_Hand)
    return getMoves(g.A_Hand if (len(g.history) % 2 == 0) else g.B_Hand, g.history[-1])
