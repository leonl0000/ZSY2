import time
import numpy as np
import zsyGame as zsy
import agents.oldDQA
oldAg = agents.oldDQA.getDQA()
staticAgents = [zsy.randomAgent, zsy.greedyAgent, oldAg]
from tensorflow.python.keras.utils import Progbar

def timeUp(tic = None, verbose = True):
    toc = time.time()
    if tic is None:
        return toc
    deltaT = toc - tic
    h = int(deltaT)/3600
    m = int((deltaT%3600)/60)
    s = deltaT%60
    if verbose:
        print("%d:%d:%.4f"%(h,m,s))
    return toc


def testStatic(agent, numGames=100, verbose=2):
    agent.setTest()
    tic = timeUp()
    toc = tic
    gs = [None] * len(staticAgents)
    for i, staticAgent in enumerate(staticAgents):
        gs[i] = zsy.multiGame(agent, staticAgent, numGames, verbose==1)
        toc = timeUp(toc, verbose==2)
    agent.setTrain()
    toc = timeUp(tic, verbose==2)
    return gs, toc-tic

def RoundRobin(Agents, matches = 10):
    numModels = len(Agents)
    wins = np.zeros(numModels)
    totalRounds = numModels*(numModels-1)/2
    progbar = Progbar(totalRounds, 50)
    k = 0
    endGames = []
    for i in range(numModels):
        for j in range(i+1, numModels):
            endGames.append(zsy.multiGame(Agents[i], Agents[j], matches, False))
            w = np.sum([g[1] for g in endGames[-1]])
            wins[i] += w
            wins[j] += matches-w
            k += 1
            progbar.update(k)
    winRates = wins / (numModels-1) / matches
    return winRates, endGames