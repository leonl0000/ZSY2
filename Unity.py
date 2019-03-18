import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import agents.Configurator as cfg

def export_model(agent, saver, input_node_names, output_node_name):
    # creates the 'out' folder where our frozen graphs will be saved
    if not os.path.exists('out'):
        os.mkdir('out')
    # GRAPH SAVING - '.pbtxt'
    tf.train.write_graph(agent.sess.graph_def, 'out', agent.name + '_graph.pbtxt')
    # GRAPH SAVING - '.chkp'
    # KEY: This method saves the graph at it's last checkpoint (hence '.chkp')
    tf.reset_default_graph()
    saver.save(agent.sess, 'out/' + agent.name + '.chkp')
    # GRAPH SAVING - '.bytes'
    # freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                           # input_binary, checkpoint_path, output_node_names,
                           # restore_op_name, filename_tensor_name,
                           # output_frozen_graph_name, clear_devices, "")
    freeze_graph.freeze_graph('out/' + agent.name + '_graph.pbtxt', None, False,
                              'out/' + agent.name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + agent.name + '.bytes', True, "")
    # GRAPH OPTIMIZING
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + agent.name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)
    with tf.gfile.FastGFile('out/opt_' + agent.name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("graph saved!")



def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':
    configs = cfg.readConfigs('02_RoundRobin')
    agent = cfg.initFromConfig(configs[7])
    agent.loadModel()
    export_model(agent, agent.saver, [agent.name+"/Placeholder"], agent.name+"_2/output/Sigmoid")