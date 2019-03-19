import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
from utils.misc import staticAgents

def battleRoyale(agents, numGames):
    newWinRates = np.zeros((len(agents)))
    numMatches = len(agents) * (len(agents) - 1) / 2
    progbar = tf.keras.utils.Progbar(numMatches, 30, 1, 1)
    ind = 0
    progbar.update(ind)
    for a in range(len(agents)):
        for b in range(a+1, len(agents)):
            endGames = zsy.multiGame(agents[a], agents[b], numGames, False)
            aWins = np.sum([g[1] for g in endGames])
            newWinRates[a] += aWins
            newWinRates[b] += numGames - aWins
            ind += 1
            progbar.update(ind)
    newWinRates /= (numGames * (len(agents) - 1))
    return newWinRates



if __name__ == '__main__':
    configs = cfg.readConfigs('02_RoundRobin')
    agents = [cfg.initFromConfig(c) for c in configs]
    for agent in agents:
        agent.loadModel()

    agents = sorted(agents, key=lambda x: -x.vs)
    from agents.ComboAgent import ComboAgent
    comboAgents = [ComboAgent(agents[:3], "Mean"), ComboAgent(agents[:6], "Mean"), ComboAgent(agents[:9],"Mean"),
                    ComboAgent(agents[:3], "Min"), ComboAgent(agents[:6], "Min"), ComboAgent(agents[:9],"Min"),
                    ComboAgent(agents[:3], "Max"), ComboAgent(agents[:6], "Max"), ComboAgent(agents[:9],"Max")]

    allAgents = agents + comboAgents + staticAgents
    wr = battleRoyale(allAgents, 10)
    wr_ag = zip([agent.name for agent in allAgents], wr)
    wr_st = sorted(wr_ag, key=lambda x: -x[1])
    wrstr = [str(x) for x in wr_st]
    f = open(os.path.join("Experiments", "04_Aggregation", "out.txt"), "w+")
    f.writelines(wrstr)
    f.close()
