import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zsyGame as zsy
import numpy as np
import tensorflow as tf
import agents.Configurator as cfg
import utils.misc as ms
import utils.deckops as dc
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
transp_zero = np.zeros((256, 4))
transp_zero[0] = [1,1,1,1]
masking_color = ListedColormap(transp_zero)

def cosineDiscordance(scores):
    # norm_scores = [score - score.mean() for score in scores
    norms = [np.linalg.norm(score) for score in scores]
    norm_scores = []
    for i, score in enumerate(scores):
        if norms[i] == 0:
            norm_scores.append(np.ones((len(score)))/np.sqrt(len(score)))
        else:
            norm_scores.append(score/norms[i])
    # print([(score.mean(), np.linalg.norm(score)) for score in norm_scores])
    discordance = np.zeros((len(scores), len(scores)))
    ave_discordance = 0
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            discordance[j, i] = norm_scores[i].dot(norm_scores[j])
            ave_discordance += discordance[j, i] if j != i else 0
    ave_discordance /= ((len(scores) * (len(scores)-1))/2)
    return discordance, ave_discordance

def discordanceGame(agentsBattle, agentsAll, prev_min_ave_discordance = 1, discordant_state = None):
    d = np.random.permutation(dc.deck)
    A = dc.cardsToHand(d[:18])
    B = dc.cardsToHand(d[18:36])
    gameStates = []
    history = []
    agentA = agentsAll[agentsBattle[0]]
    agentB = agentsAll[agentsBattle[1]]
    agentsBattle = [agentA, agentB]
    hands = [A, B]
    turn = 1 if random.random() > .5 else 0
    gameStates.append(zsy.GameState(agentA.name, agentB.name, A, B, history, False, 0, 1 - turn))
    game_ind = 1
    mxs = []
    dif_choice = np.zeros((len(agentsAll), len(agentsAll)))
    dif_choice_by_chance = 0
    min_ave_discordance = prev_min_ave_discordance
    count = 0
    while True:
        scores = [agent.getScores(gameStates[-1])[0] for agent in agentsAll]
        action_inds = [np.argmax(score) for score in scores]
        if len(scores[0]) != 1:
            count += 1
            for i in range(len(agentsAll)):
                for j in range(i, len(agentsAll)):
                    dif_choice[j,i] += action_inds[i] != action_inds[j]
            dif_choice_by_chance += (len(scores)-1)/len(scores)
            mx, ave = cosineDiscordance(scores)
            mxs.append(mx)
            if ave < min_ave_discordance:
                _hands = [A.copy(), B.copy()] if gameStates[-1].turn == 1 else [B.copy(), A.copy()]
                _history = [hist[0].copy() for hist in gameStates[-1].history]
                histAg = sum([hist for hist in _history[-2::-2]]) if len(history) > 1 else np.zeros((0, 15))
                histOp = sum([hist for hist in _history[-1::-2]]) if len(history) > 0 else np.zeros((0, 15))
                discordant_state = [*_hands, histAg, histOp, _history, mx.copy()]
                min_ave_discordance = ave
        history.append((agentsBattle[turn].getMove(gameStates[-1]), turn))
        hands[turn] -= history[-1][0]
        done = not np.any(hands[turn])
        gameStates.append(zsy.GameState(agentA.name, agentB.name, A, B, history, done, game_ind, turn))
        if done:
            break
        turn = 1 - turn
        game_ind += 1
    return mxs, count, dif_choice, dif_choice_by_chance, min_ave_discordance, discordant_state

def heatMap(arr, agNames, title="title", maskDiagonal=True, save=True):
    fig, ax = plt.subplots()
    plt.imshow(arr, cmap='Oranges', vmin=0, vmax=1)
    mask = np.tril(np.ones(arr.shape))
    if maskDiagonal:
        plt.imshow(mask, cmap=masking_color)
    ax.set_xticks(np.arange(len(agNames)))
    ax.set_yticks(np.arange(len(agNames)))
    ax.set_xticklabels(agNames)
    ax.set_yticklabels(agNames)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(agNames)):
        for j in range(len(agNames)):
            ax.text(j, i, "%.2f" % arr[i, j], size=8, ha="center", va="center", color="w")
    ax.set_title(title)
    fig.tight_layout()
    if save:
        plt.savefig(os.path.join("Experiments", "03_Discordance", title))
    else:
        plt.show()

if __name__ == '__main__':
    configs = cfg.readConfigs('02_RoundRobin')
    agents = [cfg.initFromConfig(c) for c in configs]
    for agent in agents:
        agent.exploration_prob = 0
        agent.loadModel()
    agNames = [agent.name for agent in agents]
    mxs = []
    min_ave_discordance = 1
    count = 0
    dif_choice = np.zeros((len(agents), len(agents)))
    dif_choice_by_chance = 0
    discordant_state = None
    for i in range(len(agents)):
        for j in range(i, len(agents)):
            _mxs, _count, _dif_choice, _dif_choice_by_chance, min_ave_discordance, discordant_state =\
                discordanceGame([i,j], agents, prev_min_ave_discordance=min_ave_discordance,discordant_state=discordant_state)
            mxs += _mxs
            count += _count
            dif_choice += _dif_choice
            dif_choice_by_chance += _dif_choice_by_chance

    mx = sum(mxs)/count
    heatMap(mx, agNames, "Cosine Discordance")
    dif_choice /= count
    heatMap(dif_choice, agNames, "Made Different Choices")
    print("count", count)
    print("min_ave_discordance", min_ave_discordance)
    print("dif_choice_by_chance", dif_choice_by_chance)
    print(discordant_state[1])
    print(discordant_state[2] + discordant_state[3])
    # print(discordant_state[3])
    print(discordant_state[0])
    print()
    print(discordant_state[4][-1][0])
    print(discordant_state[5])

