import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
import utils.misc as ms

def testAgents(agents, test_quantity, test_winrate_exp):
    gss = []
    for agent in agents:
        gss.append(ms.testStatic(agent, test_quantity, 0)[0])
    winRates = [[np.sum([_g[1] for _g in g]) / test_quantity for g in gs] for gs in gss]
    for j, agent in enumerate(agents):
        agent.v_Random = agent.v_Random * test_winrate_exp + (1-test_winrate_exp) * winRates[j][0]
        agent.v_Greedy = agent.v_Greedy * test_winrate_exp + (1-test_winrate_exp) * winRates[j][1]
        agent.v_OldAg = agent.v_OldAg * test_winrate_exp + (1-test_winrate_exp) * winRates[j][2]
    return winRates

def battleRoyal(buffer, agents, sim_winrate_exp, numGames = 50000, multigame_size=100):
    winrates = [agent.vs for agent in agents]
    agent_probs = winrates / np.sum(winrates)
    newWinRatesList = [[] for _ in range(len(agents))]
    newWinRates = [-1] * len(agents)
    num_iters = int(numGames/multigame_size)
    progbar = tf.keras.utils.Progbar(num_iters, 50, 1, 1, [""])
    for i in range(num_iters):
        progbar.update(i)

        inds = np.random.choice(np.arange(len(agents)), 2, p=agent_probs)
        agent1 = agents[inds[0]]
        agent2 = agents[inds[1]]

        endGames = zsy.multiGame(agent1, agent2, multigame_size, False)
        data = [zsy.gameStatesToData(endGame) for endGame in endGames]
        buffer.addToBuffer(data)
        if inds[0] != inds[1]:
            agent1_winrate = np.sum([g[1] for g in endGames]) / multigame_size
            newWinRatesList[inds[0]].append(agent1_winrate)
            newWinRatesList[inds[1]].append(1-agent1_winrate)
            newWinRates[inds[0]] = np.mean(newWinRatesList[inds[0]])
            newWinRates[inds[1]] = np.mean(newWinRatesList[inds[1]])
    for i, agent in enumerate(agents):
        if newWinRates[i] != -1:
            agent.vs = agent.vs * sim_winrate_exp + newWinRates[i] * (1-sim_winrate_exp)
    buffer.reshuffle()
    buffer.saveToFile()





if __name__ == '__main__':
    buffer = zsy.Buffer(os.path.join('Experiments', '02_RoundRobin', 'minibuf.h5'))
    configs = cfg.readConfigs('02_RoundRobin')
    agents = [cfg.initFromConfig(c) for c in configs]
    epochs = 20
    save_every = 100
    test_every = 100
    test_quantity = 100
    test_winrate_exp = .9
    sim_winrate_exp = .5
    progbar_update_every = max(save_every, test_every)
    progbarTarget = 500 / progbar_update_every

    i = 0
    j = 0
    loaded = [agent.loadModel() for agent in agents]
    if loaded[0]:
        step = agents[0].sess.run(agents[0].globalStep)
        j = step
        i = int(j/500)
        print("Loaded Agents at epoch %d, total steps %d" % (i, j))

    while i < epochs:
        print('\n\nEpoch %d, buffer size %d' % (i, buffer.numPoints()))
        for agent in agents:
            agent.exploration_prob = 1/(i+2)

        progbar = tf.keras.utils.Progbar(progbarTarget, 30, 1, 1)
        while j < (i+1)*500:
            if j % progbar_update_every == 0:
                progbar.update((j % 500)/progbar_update_every)
            if j % save_every == 0:
                for agent in agents:
                    agent.saveModel(j)
            if j % test_every == 0:
                ti = j/test_every
                testAgents(agents, test_quantity, min(test_winrate_exp, ti/(ti+1)))
            sample = buffer.getSample(sample_size=4096)
            losses = [agent.trainOnSample(sample, j) for agent in agents]
            j += 1
        battleRoyal(buffer, agents, sim_winrate_exp)
        i += 1

    for agent in agents:
        agent.saveModel(epochs * 500)
    testAgents(agents, test_quantity, test_winrate_exp)
    sample = buffer.getSample(sample_size=4096)
    _ = [agent.trainOnSample(sample, epochs*500) for agent in agents]
