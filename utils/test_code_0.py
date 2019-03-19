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


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
import utils.misc as ms

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from os.path import join



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





from agents.ComboAgent import ComboAgent
sortedAgs = sorted(agents, key=lambda x: -x.vs)
cc = sortedAgs[:9]
agent = ComboAgent(cc, "Min")
savedir = join("Unity Frozen", "Agg2")
if not os.path.isdir(savedir):
    os.mkdir(savedir)



output_node_name = agent.out.name[:-2]
saver = tf.train.Saver()
tf.train.write_graph(agent.sess.graph_def, savedir, agent.name + '_graph.pbtxt')
saver.save(agent.sess, join(savedir, agent.name + '.chkp'))
tf.reset_default_graph()
freeze_graph.freeze_graph(join(savedir, agent.name+'_graph.pbtxt'), None, False,
                          join(savedir, agent.name+'.chkp'), output_node_name,
                          "save/restore_all", "save/Const:0",
                          join(savedir, "frozen_"+agent.name+'.bytes'), True, "")
# GRAPH OPTIMIZING

input_node_names = [ag.sa.name[:-2] for ag in agent.agents]
input_graph_def = tf.GraphDef()
with tf.gfile.Open(join(savedir, 'frozen_'+agent.name+'.bytes'), "rb") as f:
    input_graph_def.ParseFromString(f.read())
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def, input_node_names, [output_node_name],
    tf.float32.as_datatype_enum)
with tf.gfile.FastGFile(join(savedir, 'opt_'+agent.name+'.bytes'), "wb") as f:
    f.write(output_graph_def.SerializeToString())
print("graph saved!")
