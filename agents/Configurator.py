import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tensorflow as tf
from agents.staticAgents import Agent
from agents.LearningAgent import *
from agents.DenseNetAgent import *
from agents.ConvNetAgent import *

import csv
from ast import literal_eval as ltev

def readConfigs(experiment_name):
    with open(os.path.join('Experiments', experiment_name, 'configs.csv'), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        configs = [Config(configDict, experiment_name) for configDict in reader]
        return configs


def writeConfigs(experiment_name, configs=None):
    expdir = os.path.join('Experiments', experiment_name)
    paths = [expdir, os.path.join(expdir, 'Models'), os.path.join(expdir, 'Results')]
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)
    with open(os.path.join('Experiments', experiment_name, 'configs.csv'), 'w', newline='') as csvfile:
        fieldnames = ['kind', 'name', 'layers', 'activations', 'lr', 'lossfn']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        if configs is not None:
            for config in configs:
                writer.writerow(config.toStringDict())

def initFromConfig(config):
    if config.kind == 'DenseNetAgent':
        return DenseNetAgent(config)
    elif config.kind == 'ConvNetAgent':
        return ConvNetAgent(config)

class Config():
    def __init__(self, configDict, experiment_name):
        self.experiment_name = experiment_name
        self.kind = configDict['kind']
        self.name = configDict['name']
        layers = ltev(configDict['layers'])
        self.layers = []
        for layer in layers:
            multiplicity = 1
            if type(layer) == list:
                multiplicity = layer[1]
                layer = layer[0]
            self.layers += [layer] * multiplicity
        activations = ltev(configDict['activations'])
        self.activations = []
        for activation in activations:
            multiplicity = 1
            if type(activation) == list:
                multiplicity = activation[1]
                activation = activation[0]
            if activation == 'relu':
                self.activations += [tf.nn.relu] * multiplicity
            elif activation == 'leaky_relu':
                self.activations += [tf.nn.leaky_relu] * multiplicity
            elif activation == 'sigmoid':
                self.activations += [tf.nn.sigmoid] * multiplicity
        if configDict['lossfn'] == 'log_loss':
            self.lossfn = tf.losses.log_loss
        self.lr = ltev(configDict['lr'])
        self.model_path = os.path.join('Experiments', self.experiment_name, 'Models', self.name)
        self.results_path = os.path.join('Experiments', self.experiment_name, 'Results', self.name)

    def toStringDict(self):
        activations = []
        for activation in self.activations:
            if activation == tf.nn.relu:
                activations.append('relu')
            elif activation == tf.nn.leaky_relu:
                activations.append('leaky_relu')
            elif activation == tf.nn.sigmoid:
                activations.append('sigmoid')
        lossfn = ""
        if self.lossfn == tf.losses.log_loss:
            lossfn = 'log_loss'
        stringDict = {
            'kind' : self.kind,
            'name' : self.name,
            'layers' : str(self.layers),
            'activations' : str(activations),
            'lr' : self.lr,
            'lossfn' : lossfn
        }
        return stringDict

