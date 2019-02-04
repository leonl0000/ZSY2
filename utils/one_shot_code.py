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
  factor of the number
"""
def countHands(cardsLeft, depth=0):
    if cardsLeft == 0:
        return 1
    if depth == 13 or depth == 14:
        return countHands(cardsLeft-1,depth+1) + countHands(cardsLeft,depth+1)
    if depth == 15 and cardsLeft != 0:
        return 0
    return sum([countHands(cardsLeft-i, depth+1) for i in range(min(5, cardsLeft+1))])


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


