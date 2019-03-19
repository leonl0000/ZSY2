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
    progbar = tf.keras.utils.Progbar(len(agents), 30, 1, 1)
    for i, agent in enumerate(agents):
        progbar.update(i)
        rg, gg, og = ms.testStatic(agent, test_quantity, 0)[0]
        agent.v_Random = agent.v_Random*test_winrate_exp + (1-test_winrate_exp)*np.sum([g[1] for g in rg])/test_quantity
        agent.v_Greedy = agent.v_Greedy*test_winrate_exp + (1-test_winrate_exp)*np.sum([g[1] for g in gg])/test_quantity
        agent.v_oldAg = agent.v_oldAg*test_winrate_exp + (1-test_winrate_exp)*np.sum([g[1] for g in og])/test_quantity


def battleRoyale(buffer, agents, sim_winrate_exp, numGames = 50000, multigame_size=100):
    winrates = [agent.vs for agent in agents]
    newWinRates = np.zeros((len(agents)))
    numMatches = len(agents) * (len(agents) + 1) / 2
    gamesPerPair = int(numGames/numMatches)
    gamesPerPair += (gamesPerPair % 2 == 1)
    progbar = tf.keras.utils.Progbar(numMatches, 30, 1, 1)
    for a in range(len(agents)):
        for b in range(a, len(agents)):
            progbar.update(a*len(agents) + b)
            endGames = zsy.multiGame(agents[a], agents[b], gamesPerPair, False)
            data = [zsy.gameStatesToData(endGame) for endGame in endGames]
            buffer.addToBuffer(data)
            if a != b:
                aWinRate = np.sum([g[1] for g in endGames]) / gamesPerPair
                newWinRates[a] += aWinRate
                newWinRates[b] += 1-aWinRate
    progbar.update(numMatches)
    newWinRates /= (len(agents)-1)
    for b, agent in enumerate(agents):
        agent.vs = agent.vs*sim_winrate_exp + (1-sim_winrate_exp)*newWinRates[b]
    buffer.reshuffle()
    buffer.saveToFile()

if __name__ == '__main__':
    buffer = zsy.Buffer(os.path.join('Experiments', '05_BattleRoyale', 'minibuf.h5'))
    configs = cfg.readConfigs('05_BattleRoyale')
    agents = [cfg.initFromConfig(c) for c in configs]
    killed_agents = []
    init_num_agents = len(agents)
    epochs = 20
    save_every = 100
    test_quantity = 300
    test_winrate_exp = .5
    sim_winrate_exp = .5

    i = 0
    j = 0
    loaded = [agent.loadModel() for agent in agents]
    if loaded[0]:
        steps = agents[0].sess.run([agent.globalStep for agent in agents])
        step = np.max(steps)
        killed = [s < step for s in steps]
        killed_agents = [agent for i, agent in enumerate(agents) if killed[i]]
        agents = [agent for i, agent in enumerate(agents) if not killed[i]]
        j = step
        i = int(j/500)
        print("Loaded %d/%d Agents at epoch %d, total steps %d" % (len(agents), init_num_agents, i, j))


    while i < epochs:
        print('\n\nEpoch %d, buffer size %d' % (i, buffer.numPoints()))
        kill_thresh = .5 * (1-.75**(i * len(agents)/init_num_agents))
        explore_prob = 1/(i+2) * np.sqrt(init_num_agents/len(agents))
        print('%d/%d Agents remaining, kill_thresh %.2f, eps %.2f' %
              (len(agents), init_num_agents, kill_thresh, explore_prob))

        progbar = tf.keras.utils.Progbar(500, 30, 1, 1)
        while j < (i+1)*500:
            if j % save_every == 0:
                for agent in agents:
                    agent.saveModel(j)
            progbar.update(j)
            sample = buffer.getSample(sample_size=4096)
            losses = [agent.trainOnSample(sample, j) for agent in agents]
            j += 1

        testAgents(agents, test_quantity, test_winrate_exp)

        for agent in agents:
            agent.exploration_prob = explore_prob
        battleRoyale(buffer, agents, sim_winrate_exp)
        killed = [agent for agent in agents if agent.vs < kill_thresh]
        agents = [agent for agent in agents if agent.vs > kill_thresh]
        for agent in killed:
            print("\n%s with vs rate %.2f killed" % (agent.name, agent.vs), end="")
        killed_agents += killed
        i += 1

    for agent in agents:
        agent.saveModel(epochs * 500)
    testAgents(agents, test_quantity, test_winrate_exp)
    sample = buffer.getSample(sample_size=4096)
    _ = [agent.trainOnSample(sample, epochs*500) for agent in agents]
