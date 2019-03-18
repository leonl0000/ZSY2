import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Next Battle Royal:
# Kill threshold: 1/2 * (1 - .9**(epoch * num_agents/initial_num_agents)
# Exploration Prob: 1/epoch * initial_num_agents/num_agents

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

from utils.deckops import OpM
import tensorflow as tf
hand = tf.placeholder(tf.int32, shape=[15], name="hand")
handr = tf.cast(tf.reshape(hand, [1, 15]), tf.int8)
OpMConst = tf.Variable(OpM, name="OpM")
inds = tf.reduce_all(handr-OpMConst>=0, 1)
ret = tf.cast(tf.boolean_mask(OpMConst, inds), tf.int32, name='ret')
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

tf.train.write_graph(sess.graph_def, 'OpM', 'OpM_graph.pbtxt')
tf.train.Saver().save(sess, 'OpM/OpM.chkp')
freeze_graph.freeze_graph('OpM/OpM_graph.pbtxt', None, False,
                              'OpM/OpM.chkp', 'ret', "save/restore_all", "save/Const:0",
                              'OpM/frozen_OpM.bytes', True, "")

input_graph_def = tf.GraphDef()
with tf.gfile.Open('OpM/frozen_OpM.bytes', "rb") as f:
    input_graph_def.ParseFromString(f.read())


output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, ['hand'], ['ret'],
        tf.int32.as_datatype_enum)


with tf.gfile.FastGFile('OpM/Opt_OpM.bytes', "wb") as f:
    f.write(output_graph_def.SerializeToString())

