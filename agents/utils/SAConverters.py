import numpy as np
import utils.deckops as dc

def DenseNetSAFromGameState(history, hand, actions):
    if len(history) >= 2:
        histories = np.concatenate([
            dc.handToExpanded(np.sum([hist[0] for hist in history[-2::-2]], axis=0)).reshape(75),
            dc.handToExpanded(np.sum([hist[0] for hist in history[-1::-2]], axis=0)).reshape(75)])
    elif len(history) == 1:
        histories = np.concatenate([
            np.zeros(75, ),
            dc.handToExpanded(history[0][0]).reshape(75)])
    else:
        histories = np.zeros((150,))
    SA = [np.concatenate([histories,
                          dc.handToExpanded(hand - action).reshape(75),
                          dc.handToExpanded(action).reshape(75)]) for action in actions]
    SA = np.stack(SA)
    return SA


def DenseNetSAFromSample(sample):
    buf_expanded_states, buf_expanded_actions, _, buf_history_ag, buf_history_op, _, buf_isWinner = sample
    states = [ES.reshape(75, ) for ES in buf_expanded_states]
    actions = [EA.reshape(75, ) for EA in buf_expanded_actions]
    # sum up all the past actions the agent took
    hist_self = [dc.handToExpanded(np.sum(hist, axis=0)).reshape(75, ) for hist in buf_history_ag]
    # sum up all the past actions the oppenent took
    hist_op = [dc.handToExpanded(np.sum(hist, axis=0)).reshape(75, ) for hist in buf_history_op]
    SA = np.array([hist_self, hist_op, states, actions]).transpose([1,0,2]).reshape(-1, 300)
    # SA = np.stack([np.concatenate([hist_self[i], hist_op[i], states[i], actions[i]])
    #                for i in range(len(buf_expanded_states))])
    return SA, buf_isWinner




def ConvNetSAFromGameState(history, hand, actions):
    if len(history) >= 2:
        history_ag = dc.handToExpanded(np.sum([hist[0] for hist in history[-2::-2]], axis=0))
        history_op = dc.handToExpanded(np.sum([hist[0] for hist in history[-1::-2]], axis=0))
    elif len(history) == 1:
        history_ag = np.zeros((5, 15))
        history_op = dc.handToExpanded(history[0][0])
    else:
        history_ag = np.zeros((5, 15))
        history_op = np.zeros((5, 15))
    history_ag = history_ag[None,:].repeat(len(actions), axis=0)
    history_op = history_op[None,:].repeat(len(actions), axis=0)
    actions = np.array(actions)
    states = hand - actions
    actions = dc.handToExpandedBatch(actions)
    states = dc.handToExpandedBatch(states)
    SA = np.concatenate([history_ag, history_op, states, actions], axis=2)
    return np.expand_dims(SA, axis=3)


def ConvNetSAFromSample(sample):
    buf_expanded_states, buf_expanded_actions, _, buf_history_ag, buf_history_op, _, buf_isWinner = sample
    # sum up all the past actions the agent took
    hist_ag = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_ag]))
    # sum up all the past actions the oppenent took
    hist_op = dc.handToExpandedBatch(np.array([np.sum(hist, axis=0) for hist in buf_history_op]))
    SA = np.concatenate([hist_ag, hist_op, buf_expanded_states, buf_expanded_actions], axis=2)
    return np.expand_dims(SA, axis=3), buf_isWinner
