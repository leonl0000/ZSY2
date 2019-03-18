import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.LearningAgent import *
import random


def getMoveMean(scores, actions):
    scores = sum(scores)
    return actions[np.argmax(scores)]

class ComboAgent(Agent):
    def __init__(self, agents, kind="mean", name="ComboAgent"):
        self.name = name
        self.agents = agents
        if kind == "mean":
            self.combo = getMoveMean


    def getMove(self, g):
        scores_actions = [agent.getScores(g) for agent in self.agents]
        scores = [s[0] for s in scores_actions]
        actions = scores_actions[0][1]
        return self.combo(scores, actions)


    def getManyMoves(self, gs):
        scores_actions = [agent.getManyScores(gs) for agent in self.agents]
        scores_sets = [s[0] for s in scores_actions]
        actions_sets = scores_actions[0][1]
        manyMoves = [None] * len(gs)
        for i in range(len(actions_sets)):
            scores = [ss[i] for ss in scores_sets]
            actions = actions_sets[i]
            manyMoves[i] = self.combo(scores, actions).reshape(1, 15)
        return manyMoves

    def setTest(self):
        pass
    def setTrain(self):
        pass

