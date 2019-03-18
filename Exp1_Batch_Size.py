import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
import utils.misc as ms

if __name__ == '__main__':
    buffer = zsy.Buffer('minibuf.h5')
    configs = cfg.readConfigs('01_BatchExperiment')
    agents = [cfg.initFromConfig(c) for c in configs]

    testInds = [0, 1, 3, 10, 30, 100, 300, 600, 900, 1200, 1500, 1800, 2100]

    numBatches = buffer.numBatchs(4096)
    summarySteps = (np.arange(100) * numBatches / 100).astype(np.int)
    summaryStepsIndex = 0
    progbar = tf.keras.utils.Progbar(numBatches, 30, 1, 1, ['loss'])
    progbar.update(0)
    for i in range(numBatches):
        if i in testInds:
            for agent in agents:
                agent.saveModel()
            gss = [ms.testStatic(agent, 1000, 0)[0] for agent in agents]
            winRates = [[np.sum([_g[1] for _g in g])/1000 for g in gs] for gs in gss]
            for j, agent in enumerate(agents):
                with tf.variable_scope(agent.config.name):
                    _ = agent.sess.run([agent.v_Random.assign(winRates[j][0]),
                                        agent.v_Greedy.assign(winRates[j][1]),
                                        agent.v_OldAg.assign(winRates[j][2])])
        sample = buffer.getSample(sample_size=4096)
        losses = [agent.trainOnSample(sample, i) for agent in agents]
        progbar.update(i+1, [('loss_%d' % a, loss) for a, loss in enumerate(losses)])

    for agent in agents:
        agent.saveModel()
