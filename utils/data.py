import numpy as np
import sys
import os
homeDir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(homeDir)

import h5py
import utils.deckops as dc
import itertools
import tensorflow as tf

class Buffer:
    def __init__(self, fileName="buffer.h5", max_eps=250000):
        self.fileName = fileName
        self.sample_order = None
        self.sample_ind = 0

        if not os.path.isfile(self.fileName):
            self.expanded_states = np.zeros((0,5,15)).astype(np.uint8)
            self.expanded_actions = np.zeros((0,5,15)).astype(np.uint8)
            self.actions = np.zeros((0,1,15)).astype(np.int8)
            self.step = np.zeros((0)).astype(np.int8)
            self.remaining_steps = np.zeros((0)).astype(np.int8)
            self.isWinner = np.zeros((0)).astype(np.int8)
            self.buffer_idx = []
            self.max_eps = max_eps
        else:
            self.loadFromFile()

    # NOTE: Ensure all data is int8 or uint8, or else numpy concats will be VERY slow
    def addToBuffer(self, data):
        expanded_states = np.array(list(itertools.chain.from_iterable([d[0] for d in data])))
        expanded_actions = np.array(list(itertools.chain.from_iterable([d[1] for d in data])))
        actions = np.array(list(itertools.chain.from_iterable([d[2] for d in data])))
        step = np.array(list(itertools.chain.from_iterable([d[3] for d in data]))).astype(np.int8)
        remaining_steps = np.array(list(itertools.chain.from_iterable([d[4] for d in data]))).astype(np.int8)
        isWinner = np.array(list(itertools.chain.from_iterable([d[5] for d in data]))).astype(np.int8)
        idx = np.argwhere(step == 1).reshape(-1)
        if len(self.buffer_idx) + len(idx) > self.max_eps:
            cut_ind_ind = len(self.buffer_idx) + len(idx) - self.max_eps
            cut_ind = self.buffer_idx[cut_ind_ind]
            self.expanded_states = self.expanded_states[cut_ind:]
            self.expanded_actions = self.expanded_actions[cut_ind:]
            self.actions = self.actions[cut_ind:]
            self.step = self.step[cut_ind:]
            self.remaining_steps = self.remaining_steps[cut_ind:]
            self.isWinner = self.isWinner[cut_ind:]
            self.buffer_idx = [idx - cut_ind for idx in self.buffer_idx[cut_ind_ind:]]
        idx = list(idx+self.expanded_states.shape[0])
        self.buffer_idx += idx
        self.expanded_states = np.vstack([self.expanded_states, expanded_states])
        self.expanded_actions = np.vstack([self.expanded_actions, expanded_actions])
        self.actions = np.vstack([self.actions, actions])
        self.step = np.concatenate([self.step, step])
        self.remaining_steps = np.concatenate([self.remaining_steps, remaining_steps])
        self.isWinner = np.concatenate([self.isWinner, isWinner])

    def numGames(self):
        return len(self.buffer_idx)

    def numPoints(self):
        return self.expanded_states.shape[0]

    def numBatchs(self, batch_size=4096):
        return int(self.expanded_states.shape[0]/batch_size)

    # In an active setting, call this when adding/removing from buffer or else
    # the indices won't work (games are variable length)
    def reshuffle(self, shuffle=True):
        self.sample_order = np.random.permutation(self.expanded_states.shape[0]) \
            if shuffle else np.arange(self.expanded_states.shape[0])
        self.sample_ind = 0

    def getSample(self, sample_size=4096, increment_sample=True, shuffle=True, reOrder=False):
        if self.sample_order is None or reOrder or self.sample_ind + sample_size > len(self.sample_order):
            self.reshuffle(shuffle)
        idx = self.sample_order[self.sample_ind : self.sample_ind + sample_size]
        history_ag = [self.actions[(i-self.step[i]+1)+(self.step[i]+1)%2:i:2] for i in idx]
        history_op = [self.actions[(i-self.step[i]+1)+(self.step[i])%2:i:2] for i in idx]

        sample = (self.expanded_states[idx],
                  self.expanded_actions[idx],
                  self.actions[idx],
                  history_ag,
                  history_op,
                  self.remaining_steps[idx],
                  self.isWinner[idx])
        if increment_sample:
            self.sample_ind += sample_size
        return sample


    def saveToFile(self):
        f = h5py.File(self.fileName, 'w')
        ES_dset = f.create_dataset('expanded_states', self.expanded_states.shape, dtype=self.expanded_states.dtype)
        ES_dset[...] = self.expanded_states
        EA_dset = f.create_dataset('expanded_actions', self.expanded_actions.shape, dtype=self.expanded_actions.dtype)
        EA_dset[...] = self.expanded_actions
        A_dset = f.create_dataset('actions', self.actions.shape, dtype=self.actions.dtype)
        A_dset[...] = self.actions
        St_dset = f.create_dataset('step', self.step.shape, dtype=self.step.dtype)
        St_dset[...] = self.step
        RS_dset = f.create_dataset('remaining_steps', self.remaining_steps.shape, dtype=self.remaining_steps.dtype)
        RS_dset[...] = self.remaining_steps
        iW_dset = f.create_dataset('isWinner', self.isWinner.shape, dtype=self.isWinner.dtype)
        iW_dset[...] = self.isWinner
        idx_as_np = np.array(self.buffer_idx)
        idx_dest = f.create_dataset('buffer_idx', idx_as_np.shape, dtype=idx_as_np.dtype)
        idx_dest[...] = idx_as_np
        max_eps_dset = f.create_dataset('max_eps', (1,), dtype=np.int32)
        max_eps_dset[...] = self.max_eps
        f.close()

    def loadFromFile(self):
        f = h5py.File(self.fileName, 'r')
        self.expanded_states = f["expanded_states"][:]
        self.expanded_actions = f["expanded_actions"][:]
        self.actions = f["actions"][:]
        self.step = f["step"][:]
        self.remaining_steps = f["remaining_steps"][:]
        self.isWinner = f["isWinner"][:]
        self.buffer_idx = list(f['buffer_idx'][:])
        self.max_eps = f['max_eps'][:][0]

    def checkBuffer(self):
        self.reshuffle()
        progbar = tf.keras.utils.Progbar(self.expanded_states.shape[0], 50, 1, 1)
        for counter, i in enumerate(self.sample_order):
            hist_ag = np.sum(self.actions[(i - self.step[i] + 1) + (self.step[i] + 1) % 2:i:2])
            hist_op = np.sum(self.actions[(i-self.step[i]+1)+(self.step[i])%2:i:2])
            hand_ag = np.sum(dc.expandedToHand(self.expanded_states[i]))
            hand_op = np.sum(dc.expandedToHand(self.expanded_states[i-1])) if self.step[i] != 1 else 18
            act_ex_ag = np.sum(dc.expandedToHand(self.expanded_actions[i]))
            act_ag = np.sum(self.actions[i])
            if hist_ag + act_ag + hand_ag != 18:
                print(counter, i, 'Agent with action fail')
                return counter, i
            if hist_op + hand_op != 18:
                print(counter, i, 'Opp fail')
                return counter, i
            if hist_ag + act_ex_ag + hand_ag != 18:
                print(counter, i, 'Agent with expanded fail')
                return counter, i
            progbar.update(counter)

    def reIndex(self):
        self.buffer_idx = list(np.argwhere(self.step == 1).reshape(-1))


    def resizeBuffer(self, newMaxSize=250000):
        cur_size = len(self.buffer_idx)
        self.max_eps = newMaxSize
        if cur_size > self.max_eps:
            cut_ind = self.buffer_idx[-self.max_eps]
            self.expanded_states = self.expanded_states[cut_ind:]
            self.expanded_actions = self.expanded_actions[cut_ind:]
            self.actions = self.actions[cut_ind:]
            self.step = self.step[cut_ind:]
            self.remaining_steps = self.remaining_steps[cut_ind:]
            self.isWinner = self.isWinner[cut_ind:]
            self.buffer_idx = [idx - cut_ind for idx in self.buffer_idx[-self.max_eps:]]