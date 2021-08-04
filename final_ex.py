import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------- constants----------------------------------------------------------
DATA_PATH_1 = "mouse_1.npy"  # mouse 1
DATA_PATH_2 = 'm2_007D2D1B0_scores.npy'  # mouse 2
DATA_PATH_3 = 'm3_0007D2E150_scores.npy'  # mouse 3
DATA_PATH_4 = 'm4_0007D31D1A_scores.npy'  # mouse 4

SIMULATION_NUM = 100
LICK = 1
NOT_LICK = 0
SCORE_DICT = {0: "Hit", 1: "FA", 2: "Miss", 3: "CR", "Hit": 0, "FA": 1,
              "Miss": 2, "CR": 3}
SCORES_LST = ["Hit", "FA", "Miss", "CR"]
STATES_NUM = 4
HIT_SCROE = 0
FA_SCORE = 1
MISS_SCORE = 2
CR_SCORE = 3
# dict {key = next state, value = reward}
REWARDS = {"Hit": 1, "FA": 0, "Miss": 0, "CR": 0}
GO = 1
NO_GO = 0


# --------- data handling -----------------------------------------------------
def get_mouse_action_reward(score):
    if score == HIT_SCROE or score == FA_SCORE:
        action = LICK
    else:
        action = NOT_LICK

    return action, REWARDS[SCORE_DICT[score]]


def get_mouse_stimuli(score):
    if score == HIT_SCROE or score == MISS_SCORE:
        stimuli = GO
    else:
        stimuli = NO_GO
    return stimuli


def score_to_stimuli(score_df):
    stimuli_lst = []
    for score in score_df:
        stimuli_lst.append(get_mouse_stimuli(score))
    return stimuli_lst


def get_data(data_path):
    score_df = np.load(data_path)
    actions, rewards = [], []
    for score in score_df:
        action, reward = get_mouse_action_reward(score)
        actions.append(action)
        rewards.append(reward)

    return score_df, actions, rewards


# ----------------------------------------------------------------------------

def get_agent_reward_and_next_state(stimuli, action):
    if stimuli == GO:
        if action == LICK:
            next_state = "Hit"
        else:
            next_state = "Miss"

    else:
        if action == LICK:
            next_state = "FA"
        else:
            next_state = "CR"

    return SCORE_DICT[next_state], REWARDS[next_state]


def td_model(stimuli_lst, eta, trails_num, beta=1):
    v = np.zeros((2, 2))
    actions = []
    rewards = []
    states = []
    for i in range(trails_num):
        stimuli = stimuli_lst[i]

        e1 = np.exp(beta * v[stimuli, LICK])
        e2 = np.exp(beta * v[stimuli, NOT_LICK])
        p = e1 / (e1 + e2)

        action = np.random.choice([LICK, NOT_LICK], p=[p, 1 - p])
        actions.append(action)
        next_state, reward = get_agent_reward_and_next_state(stimuli, action)
        rewards.append(reward)
        states.append(next_state)

        v[stimuli, action] += eta * (reward - v[stimuli, action])
    return actions, rewards, states


def reinforce_model(stimuli_lst, eta, trails_num):
    w1, w2 = 0, 0
    p1, p2 = 0.5, 0.5
    states, actions, rewards = [], [], []
    for t in range(trails_num):
        stimuli = stimuli_lst[t]
        if stimuli == GO:
            p = p1
        else:
            p = p2
        action = np.random.choice([LICK, NOT_LICK], p=[p, 1 - p])
        actions.append(action)
        next_state, reward = get_agent_reward_and_next_state(stimuli, action)
        states.append(next_state)
        rewards.append(reward)

        if stimuli == GO:
            w1 += eta * rewards[t] * (actions[t] - p1)
            p1 = 1 / (1 + np.exp(-1 * w1))
        else:
            w2 += eta * rewards[t] * (actions[t] - p2)
            p2 = 1 / (1 + np.exp(-1 * w2))
    return actions, rewards, states


def log_likelihood_td(stimuli_lst, actions, rewards, eta, beta):
    v = np.zeros((2, 2))
    log_likelihood = 0
    for t in range(SIMULATION_NUM):
        p = np.exp(beta * v[stimuli_lst[t], LICK]) / (
                np.exp(beta * v[stimuli_lst[t], LICK]) + np.exp(
            beta * v[stimuli_lst[t], NOT_LICK]))

        v[stimuli_lst[t], actions[t]] += eta * (
                rewards[t] - v[stimuli_lst[t], actions[t]])

        if actions[t] == LICK:
            log_likelihood += np.log(p)
        else:
            log_likelihood += np.log(1 - p)

    return log_likelihood


def log_likelihood_rein(stimuli_lst, actions, rewards, eta):
    w1, w2 = 0, 0
    p1, p2 = 0.5, 0.5
    log_likelihood = 0
    for t in range(SIMULATION_NUM):
        cur_action = actions[t]
        if stimuli_lst[t] == GO:
            w1 += eta * rewards[t] * (cur_action - p1)
            p1 = 1 / (1 + np.exp(-1 * w1))
            if cur_action == LICK:
                log_likelihood += np.log(p1)
            else:
                log_likelihood += np.log(1 - p1)
        else:
            # stimuli == NO GO
            w2 += eta * rewards[t] * (actions[t] - p2)
            p2 = 1 / (1 + np.exp(-1 * w2))

            if cur_action == LICK:
                log_likelihood += np.log(p2)
            else:
                log_likelihood += np.log(1 - p2)

    return log_likelihood


def find_best_eta_by_beta(stimuli_lst, actions, rewards, eta):
    best_beta_td = -1
    max_log_likelihood_td = -np.inf
    beta_lst = np.arange(0.05, 10, 0.05)
    for beta in beta_lst:
        cur_log_likelihood_td = log_likelihood_td(stimuli_lst, actions,
                                                  rewards, eta, beta)
        if cur_log_likelihood_td > max_log_likelihood_td:
            best_beta_td = beta
            max_log_likelihood_td = cur_log_likelihood_td

    return best_beta_td, max_log_likelihood_td


def find_best_eta_rein(stimuli, actions, rewards):
    best_eta_rein = -1
    max_log_likelihood_rein = -np.inf
    eta_lst = np.arange(0.005, 1, 0.005)
    for eta in eta_lst:
        cur_log_likelihood_rein = log_likelihood_rein(stimuli, actions,
                                                      rewards, eta)
        if cur_log_likelihood_rein > max_log_likelihood_rein:
            best_eta_rein = eta
            max_log_likelihood_rein = cur_log_likelihood_rein

    return best_eta_rein


def calc_accuracy(mouse_states, model_states):
    return np.sum(np.array(mouse_states) == np.array(model_states)) / len(
        mouse_states)


def find_best_eta_and_beta_td(stimuli_lts, actions, rewards):
    # find best eta and beta: for each eta find best beta. choose the tuple
    # (eta, beta) that maximize log likelihood
    eta_lst = np.arange(0.005, 1, 0.005)
    beta_lst = []
    log_likelihood_lst = []
    for eta in tqdm(eta_lst):
        max_beta, max_log_likelihood = find_best_eta_by_beta(stimuli_lts,
                                                             actions, rewards,
                                                             eta)
        beta_lst.append(max_beta)
        log_likelihood_lst.append(max_log_likelihood)

    max_ind = np.argmax(np.array(log_likelihood_lst))
    best_eta_td_exp = eta_lst[max_ind]
    best_beta_exp = beta_lst[max_ind]

    return best_eta_td_exp, best_beta_exp


def accuracy_by_param(stimuli_lst, trials_num, mouse_actions, eta_td, beta_td,
                      eta_rain):
    actions_td, rewards_td, states_td = td_model(stimuli_lst, eta_td,
                                                 trials_num, beta_td)
    actions_rein, rewards_rein, states_rein = reinforce_model(stimuli_lst,
                                                              eta_rain,
                                                              trials_num)

    td_accuracy = calc_accuracy(mouse_actions, actions_td)
    rein_accuracy = calc_accuracy(mouse_actions, actions_rein)
    print("TD accuracy: {}\nReinforcement accuracy: {}".format(td_accuracy,
                                                               rein_accuracy))


def find_best_model(best_eta_td, best_beta, best_eta_rein, mouse_actions,
                    stimuli_lst, trials_num):
    all_td_accuracy = []
    all_rein_accuracy = []

    for i in tqdm(range(SIMULATION_NUM)):
        actions_td, rewards_td, states_td = td_model(stimuli_lst, best_eta_td,
                                                     trials_num, best_beta)

        actions_rein, rewards_rein, states_rein = reinforce_model(stimuli_lst,
                                                                  best_eta_rein,
                                                                  trials_num)

        actions_td = np.array(actions_td)
        actions_rein = np.array(actions_rein)

        td_accuracy = calc_accuracy(mouse_actions, actions_td)
        rein_accuracy = calc_accuracy(mouse_actions, actions_rein)
        all_td_accuracy.append(td_accuracy)
        all_rein_accuracy.append(rein_accuracy)

    avg_td_accuracy = np.average(np.array(all_td_accuracy))
    avg_rein_accuracy = np.average(np.array(all_rein_accuracy))
    print("TD accuracy: {}\nReinforcement accuracy: {}".format(avg_td_accuracy,
                                                               avg_rein_accuracy))


def plot_accuracy_bins(td_accuracy_lst, rein_accuracy_lst, labels):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, td_accuracy_lst, width, label="TD-learning")
    rects2 = ax.bar(x + width / 2, rein_accuracy_lst, width, label='Reinforce')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    ax.set_xticks(x)
    ax.set_ylabel('accuracy')
    ax.set_xticklabels(labels)
    plt.legend(loc='lower right')
    plt.title("Models accuracy")
    fig.tight_layout()
    plt.show()


def main(data_path, mouse_num):
    # actual experiment data
    mouse_score_df, actions, rewards = get_data(data_path)
    stimuli_lst = score_to_stimuli(mouse_score_df)
    trials_num = len(mouse_score_df)

    best_eta_td_exp, best_beta_exp = find_best_eta_and_beta_td(stimuli_lst,
                                                               actions,
                                                               rewards)
    print("TD exp {}:\neta: {}\nbeta: {}".format(mouse_num, best_eta_td_exp,
                                                 best_beta_exp))
    best_eta_rein_exp = find_best_eta_rein(stimuli_lst, actions, rewards)
    print("REINFORCE exp {}:\neta: {}".format(mouse_num, best_eta_rein_exp))

    find_best_model(best_eta_td_exp, best_beta_exp, best_eta_rein_exp, actions,
                    stimuli_lst, trials_num)

