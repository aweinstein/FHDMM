import sys
import numpy as np
from pandas import DataFrame
import pandas as pd
from utils import softmax

pd.options.display.float_format = '{:.2f}'.format

class ContextualBandit(object):
    def __init__(self):
        # Contexts and their probabilities of winning
        self.contexts = {'punishment': 0.2,
                         'neutral': 0.5,
                         'reward': 0.8}
        self.actions = (23, 14, 8, 3)
        self.n = len(self.actions)
        self.get_context()

    def get_context_list(self):
        return list(self.contexts.keys())

    def get_context(self):
        self.context = np.random.choice(list(self.contexts.keys()))
        return self.context

    def reward(self, action):
        if action not in self.actions:
            print('Error: action not in', self.actions)
            sys.exit(-1)
        p = self.contexts[self.context]
        if np.random.rand() < p:
            r = action
        else:
            r = -action
        return r

class ContextualAgent(object):
    def __init__(self, bandit, beta=None, alpha=None):
        self.beta = beta
        self.bandit = bandit
        self.actions = self.bandit.actions
        self.contexts = self.bandit.get_context_list()
        self.n = bandit.n
        self.alpha = alpha
        self.Q = {}
        # init with small random numbers to avoid ties
        for context in self.contexts:
            self.Q[context] = np.random.uniform(0, 1e-4, self.n)

        self.log = {'context':[], 'reward':[], 'action':[],
                    'Q(c,23)':[], 'Q(c,14)':[], 'Q(c,8)':[], 'Q(c,3)': []}

    def run(self):
        context = self.bandit.get_context()
        action = self.choose_action(context)
        reward = self.bandit.reward(self.actions[action])

        # Update action-value
        self.update_action_value(context, action, reward)

        # Keep track of performance
        self.log['context'].append(context)
        self.log['reward'].append(reward)
        self.log['action'].append(self.actions[action])
        self.log['Q(c,23)'].append(self.Q[context][0])
        self.log['Q(c,14)'].append(self.Q[context][1])
        self.log['Q(c,8)'].append(self.Q[context][2])
        self.log['Q(c,3)'].append(self.Q[context][3])

    def choose_action(self, context):
        p = softmax(self.Q[context], self.beta)
        actions = range(self.n)
        action = np.random.choice(actions, p=p)
        return action

    def update_action_value(self, context, action, reward):
        error = reward - self.Q[context][action]
        self.Q[context][action] += self.alpha * error

def run_single_softmax_experiment(beta, alpha):
    """Run experiment with agent using softmax update rule."""
    print('Running a contextual bandit experiment')
    cb = ContextualBandit()
    ca = ContextualAgent(cb, beta=beta, alpha=alpha)
    trials = 360

    for _ in range(trials):
        ca.run()
    df = DataFrame(ca.log, columns=('context', 'action', 'reward', 'Q(c,23)',
                                    'Q(c,14)', 'Q(c,8)', 'Q(c,3)'))
    fn = 'softmax_experiment.csv'
    df.to_csv(fn, index=False)
    print('Sequence written in', fn)
    globals().update(locals())

if __name__ == '__main__':
    np.random.seed(42)
    run_single_softmax_experiment(0.5, 0.1)
    import vis
    import matplotlib.pyplot as plt
    plt.close('all')
    vis.plot_simulation_run()
