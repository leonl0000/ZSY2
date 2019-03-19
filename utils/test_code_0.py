import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




from Exp3_Discordance import *
configs = cfg.readConfigs('02_RoundRobin')
agents = [cfg.initFromConfig(c) for c in configs]
for agent in agents:
    agent.loadModel()
    agent.exploration_prob = 0



agNames = [agent.name for agent in agents]
mx, min_ave_discordance, dif_choice, dif_choice_by_chance, discordant_state = discordanceGame([0,1], agents)
mx, min_ave_discordance, dif_choice, dif_choice_by_chance, discordant_state = discordanceGame([1,2], agents, prev_min_ave_discordance=min_ave_discordance, discordant_state=discordant_state)


from agents.ComboAgent import ComboAgent
c1 = ComboAgent(agents)
c2 = ComboAgent(agents[6:])

from utils.misc import timeUp, testStatic
# tic = timeUp()
# gs=zsy.multiGame(c2, agents[7], 1000)
# tic = timeUp(tic)



agents2 = agents + [c1, c2]

scores_sum = np.zeros((14,3))

for i in range(10):
    numGames = 500
    gsR = []
    gsG = []
    gsO = []
    for i, agent in enumerate(agents2):
        print(i, end=' ', flush=True)
        ret = testStatic(agent, numGames, 0)
        gsR.append(ret[0][0])
        gsG.append(ret[0][1])
        gsO.append(ret[0][2])
    print("\n", len(gsR), len(gsG), len(gsO))
    scores = []
    for i in range(len(gsR)):
        scores.append([
            np.sum([g[1] for g in gsR[i]])/numGames,
            np.sum([g[1] for g in gsG[i]])/numGames,
            np.sum([g[1] for g in gsO[i]])/numGames,
            ])
    scores_sum += np.array(scores)

for score in scores_sum:
    print(score)



ret = testStatic(agents[0])