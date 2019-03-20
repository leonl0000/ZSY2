import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
from utils.misc import staticAgents
from agents.ComboAgent import ComboAgent

def battleRoyale(agents, numGames, maxgames=2000, maxByCombo=True):
    newWinRates = np.zeros((len(agents)))
    vsStatic = np.zeros((len(agents)-3,3))
    numMatches = len(agents) * (len(agents) - 1) / 2
    progbar = tf.keras.utils.Progbar(numMatches, 30, 1, 1)
    ind = 0
    progbar.update(ind)
    for a in range(len(agents)):
        for b in range(a+1, len(agents)):
            if maxByCombo:
                divisor = len(agents[a].agents) if type(agents[a]) == ComboAgent else 1
                divisor = max(divisor, len(agents[b].agents) if type(agents[b]) == ComboAgent else 1)
                _maxgames = int(maxgames / divisor)
            else:
                _maxgames = maxgames
            toPlay = numGames
            while toPlay > 0:
                p = min(_maxgames, toPlay)
                endGames = zsy.multiGame(agents[a], agents[b], p, False)
                aWins = np.sum([g[1] for g in endGames])
                newWinRates[a] += aWins
                newWinRates[b] += p - aWins
                toPlay -= p
            ind += 1
            progbar.update(ind)
    newWinRates /= (numGames * (len(agents) - 1))
    vsStatic /= numGames
    return newWinRates, vsStatic



if __name__ == '__main__':
    configs = cfg.readConfigs('05_BattleRoyale')
    agents = [cfg.initFromConfig(c) for c in configs]
    f = open(os.path.join("Experiments", "04_Aggregation", "out_05BR_1000.txt"), "w+")
    for agent in agents:
        agent.loadModel()

    agents = sorted(agents, key=lambda x: -x.vs)
    wrstr = "Top Models by VS\r\n"
    tm_info = [str(i+1) + '. ' + agents[i].name + ', %.5f'%agents[i].vs for i in range(9)]
    wrstr += '\r\n'.join(tm_info) + '\r\n'
    comboAgents = [ComboAgent(agents[:3], "Mean"), ComboAgent(agents[:6], "Mean"), ComboAgent(agents[:9],"Mean"),
                    ComboAgent(agents[:3], "Min"), ComboAgent(agents[:6], "Min"), ComboAgent(agents[:9],"Min"),
                    ComboAgent(agents[:3], "Max"), ComboAgent(agents[:6], "Max"), ComboAgent(agents[:9],"Max")]

    allAgents = agents + comboAgents + staticAgents
    wr, vs_static = battleRoyale(allAgents, 1000)
    wr_ag = zip([agent.name for agent in allAgents], wr)
    wr_st = sorted(wr_ag, key=lambda x: -x[1])
    wrstr += '\r\n'.join([a + ", %.6f"%b for (a, b) in wr_st]) + '\r\n'

    wrstr += '\r\nEach model vs Random, Greedy, and the old algorithm\r\n'
    table = zip([agent.name for agent in allAgents[:-3]], wr[:-3], vs_static[:,0], vs_static[:,1], vs_static[:,2])
    table = sorted(table, key=lambda x: -x[1])
    wrstr += '\r\n'.join([a + ", %.6f, %.6f, %.6f"%(b,c,d) for (a,_,b,c,d) in table])
    f.writelines(wrstr)
    f.close()
