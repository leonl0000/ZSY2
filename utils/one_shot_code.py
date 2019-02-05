"""
    This file contains code that was run once or a few times,
        mainly to get an intuitive sense of the scope of the
        problem or to create matrices that only need to be
        created once.
"""
import numpy as np
OpeningMovesSaveFileName = "openingMoves.npz"

"""
How many possible hands are there of length cardsLeft
  where the smallest card has value depth?
84,166,160 for cardsLeft=18
In other words, there are 84M possible hands to be dealt
  at the start. This is not the number of possible
  opening game states as that has the additional
  factor of the cards in the opponents hand
"""
countHandsCache = {}
def countHands(cardsLeft, depth=0):
    if cardsLeft == 0:
        return 1
    if depth == 15 and cardsLeft != 0:
        return 0
    if (cardsLeft, depth) in countHandsCache:
        return countHandsCache[(cardsLeft, depth)]
    if depth == 13 or depth == 14:
        val = countHands(cardsLeft-1,depth+1) + countHands(cardsLeft,depth+1)
    else:
        val = sum([countHands(cardsLeft-i, depth+1) for i in range(min(5, cardsLeft+1))])
    countHandsCache[(cardsLeft, depth)] = val
    return val



"""
Extension of countHands: How many possible opening states are there?
The number of 18 card hands is the space of possible opening observations
But, the state space includes the cards the opponent has as well
So, how many possible ways are there to distribute 18 cards to each of two players
    given that the suit doesn't matter?
RE: depth is the value of the smallest card allowed.
0 -> 3
...
7 -> 10
8 -> J
...
11 -> A
12 -> 2
13 -> JB (Joker black)
14 -> JR (Joker red)
Results: 151,632,049,354,500 possible opening states
"""
countStatesCache = {}
def countStates(cardsLeftA, cardsLeftB, depth=0):
    # If there are no cards left to distribute, there's only one way to do it!
    if cardsLeftA == 0:
        if cardsLeftB == 0:
            return 1
        else:
            return countHands(cardsLeftB, depth)
    if cardsLeftB == 0:
        return countHands(cardsLeftA, depth)
    # If the depth has maxed out (i.e. the min card allowed to distribute > Joker red)
    # there's no way to do it. It's invalid
    if depth == 15 and (cardsLeftA != 0 or cardsLeftB != 0):
        return 0
    if (cardsLeftA, cardsLeftB, depth) in countStatesCache:
        return countStatesCache[(cardsLeftA, cardsLeftB, depth)]
    if (cardsLeftB, cardsLeftA, depth) in countStatesCache:
        return countStatesCache[(cardsLeftB, cardsLeftA, depth)]
    num = 0
    if depth < 13:
        for i in range(min(5, cardsLeftA+1)):
            # give i cards of value "depth" to A
            for j in range(min(5-i, cardsLeftB+1)):
                # give j cards of value "depth" to B
                num += countStates(cardsLeftA-i, cardsLeftB-j, depth+1)
    else:
        num += countStates(cardsLeftA, cardsLeftB, depth+1) + \
                countStates(cardsLeftA-1, cardsLeftB, depth+1) + \
                countStates(cardsLeftA, cardsLeftB-1, depth+1)
    countStatesCache[(cardsLeftA, cardsLeftB, depth)] = num
    return num




"""
How many possible chains of length cardsLeft are there?
9021 for cardsLeft=18
Given this surprising result, I thought that the best way
    find all possible opening moves is to literally make
    an array that contains all of them and see which rows
    of that matrix a player's hand is greater than.
"""
def countChainsHelper(chain, cardsLeft):
    if cardsLeft==0:
        return (14-len(chain))*(len(chain)>1)
    if cardsLeft<0:
        return 0
    return sum([countChainsHelper(chain+[i], cardsLeft-i) for i in range(2,5)])

def countChains(cardsLeft):
    s = 0
    for c in range(2, cardsLeft+1):
        s += countChainsHelper([2], c-2)
        s += countChainsHelper([3], c-3)
        s += countChainsHelper([4], c-4)
    return s


# 2D array of all possible opening moves
# should be of size (9075, 15)
def createOpeningMatrix(maxChainLength=9):
    # all possible singles
    OpM = np.eye(15).astype(np.int64)

    # all possible doubles, triples, bombs
    # 13 and 14, corresponding to the red and black jokers
    # cannot be doubles or such because there is only 1 copy
    doubles = 2*OpM[:13]
    triples = 3*OpM[:13]
    bombs   = 4*OpM[:13]
    OpM = np.concatenate((OpM, doubles, triples, bombs), axis=0)

    # This is definitely NOT the most efficient way to find
    # all possible chains, but I only have to run this once
    # and save the result, so it should be fine
    cc = [doubles, triples, bombs]
    for chainLength in range(2,maxChainLength+1):
        for i in range(3**chainLength):
            cc_ind = np.unravel_index(i, [3]*chainLength)
            # if this chain needs more than 18 cards, it's not possible
            if(np.sum(cc_ind) + 2*len(cc_ind) > 18):
                # print(cc_ind)
                continue
            base_chain = np.zeros((1,15)).astype(np.int64)
            for j in range(chainLength):
                base_chain += cc[cc_ind[j]][j]
            OpM = np.concatenate([OpM] +
                                [np.roll(base_chain, j) for j in range(14-chainLength)], axis=0)
    return OpM

def saveOpeningMatrix(OpM, fname=OpeningMovesSaveFileName):
    f = open(fname, 'wb+')
    np.save(f, OpM)


