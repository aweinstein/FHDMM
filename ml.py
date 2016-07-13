import numpy as np
import pandas as pd
from utils import softmax
from scipy.optimize import minimize
from matplotlib import cm
import matplotlib.pyplot as plt

import agent

class ML(object):
    def __init__(self, df):
        """The DataFrame df must contain columns 'action' 'reward'.
        and 'cue'.
        """
        self.n_actions = 4
        self.cues = df['cue'].unique().tolist()
        self.n_cues = len(self.cues)

        self.df = df

    def neg_log_likelihood(self, alphabetas):
        df = self.df

        alphas = alphabetas[0::2]
        betas = alphabetas[1::2]
        df = self.df[self.df['cue'].isin(self.cues)]
        actions, rewards = df['action'].values, df['reward'].values
        cues = df['cue'].values
        prob_log = 0
        Q = dict([[cue, np.zeros(self.n_actions)] for cue in self.cues])
        for action, reward, cue in zip(actions, rewards, cues):
            alpha = alphas[self.cues.index(cue)]
            beta = betas[self.cues.index(cue)]
            Q[cue][action] += alpha * (reward - Q[cue][action])
            prob_log += np.log(softmax(Q[cue], beta)[action])

        return -prob_log

    def ml_estimation(self):
        bounds = ((0,1), (0,2)) * self.n_cues
        r = minimize(self.neg_log_likelihood, [0.1,0.1]*self.n_cues,
                     method='L-BFGS-B',
                     bounds=bounds)
        return r

    def fit_model(self):
        r = self.ml_estimation('Nelder-Mead')
        if r.status != 0:
            print('trying with Powell')
            r = self.ml_estimation('Powell')
        return r

    def plot_ml(self, ax, alpha, beta, alpha_hat, beta_hat):

        from itertools import product
        n = 50
        alpha_max = 0.2
        beta_max = 1.5
        if alpha is not None:
            alpha_max = alpha_max if alpha < alpha_max else 1.1 * alpha
            beta_max = beta_max if beta < beta_max else 1.1 * beta
        if alpha_hat is not None:
            alpha_max = alpha_max if alpha_hat < alpha_max else 1.1 * alpha_hat
            beta_max = beta_max if beta_hat < beta_max else 1.1 * beta_hat
        alphas = np.linspace(0, alpha_max, n)
        betas = np.linspace(0, beta_max, n)
        Alpha, Beta = np.meshgrid(alphas, betas)
        Z = np.zeros(len(Alpha) * len(Beta))
        for i, (a, b) in enumerate(product(alphas, betas)):
            Z[i] = self.neg_log_likelihood((a, b, 0, 0, 0, 0))
        Z.resize((len(alphas), len(betas)))
        ax.contourf(Alpha, Beta, Z.T, 50, cmap=cm.viridis)
        if alpha is not None:
            ax.plot(alpha, beta, 'rs', ms=7)
        if alpha_hat is not None:
            ax.plot(alpha_hat, beta_hat, 'r^', ms=7)
        ax.set_xlabel(r'$\alpha_c$', fontsize=20)
        ax.set_ylabel(r'$\beta_c$', fontsize=20)
        return

    def plot_single_subject(self, ax, r, subject, cue):
        alpha, beta = r.x
        converged = ('yes', 'no')[r.status]
        cue = ''.join([str(c) for c in self.cues])
        title = 'Subject: {}, cue: {}, converged: {}'.format(subject, cue,
                                                             converged)
        if r.status == 0:
            self.plot_ml(ax, alpha, beta, None, None)
        else:
            self.plot_ml(ax, None, None, None, None)
        ax.set_title(title)

def card_cue_bandit_experiment(alpha=0.1, beta=0.5):
    np.random.seed(42)
    print('Running experiment with alpha={} and beta={}'.format(alpha, beta))
    df = agent.run_single_softmax_experiment(beta, alpha)

    f = lambda x: {'reward':0, 'punishment':1, 'neutral':2}[x]
    df['cue'] = df['context'].map(f)

    f = lambda x: {23:0, 14:1, 8:2, 3:3}[x]
    df['action'] = df['action'].map(f)
    ml = ML(df)
    r = ml.ml_estimation()
    print(r)

    alpha_hat, beta_hat = r.x[:2]
    fig, ax = plt.subplots(1, 1)
    ml.plot_ml(ax, alpha, beta, alpha_hat, beta_hat)
    plt.tight_layout()
    plt.savefig('likelihood.pdf')
    plt.show()
    globals().update(locals())

def fit_behavioral_data():
    """Fit a model for all subjects. """
    df = pd.read_pickle('data.pkl')
    subjects = df.index.get_level_values('subject').unique()
    data = np.empty((subjects.size, 10))
    cues = (0, 1)
    for i, subject in enumerate(subjects):
        print('Fitting model for subject {}'.format(subject))
        df_s = df.loc[subject]
        for cue in cues:
            ml = ML(df_s[df_s['cue']==cue])
            r = ml.ml_estimation()
            data[i,2*cue:(2*cue+2)] = r.x
            data[i,2*cue+4:2*cue+6] = np.sqrt(np.diag(r.hess_inv.todense()))
            data[i,cue+8] = r.fun

    model = pd.DataFrame(data, pd.Index(subjects, name='subject'),
                         ['alpha_0', 'beta_0', 'alpha_1', 'beta_1',
                          'se_alpha_0', 'se_beta_0', 'se_alpha_1', 'se_beta_1',
                          'NLL_0', 'NLL_1'])
    return model


def fit_single_subject(subject=4):
    df = pd.read_pickle('data.pkl')
    print('Fitting model for subject {}'.format(subject))
    df_s = df.loc[subject]

    cues = (0, 1, 2)
    for cue in cues:
        ml = ML(df_s[df_s['cue']==cue])
        r = ml.ml_estimation()
        H_inv = r.hess_inv.todense()
        print('\t cue:{:d}'.format(cue))
        print('\t\tr:\n\t\t\t{}\n'.format(r.x))
        print('\tInverse of Hessian:\n{}\n'.format(H_inv))


    globals().update(locals())


if __name__ == '__main__x':
    #card_cue_bandit_experiment()
    #fit_behavioral_data()
    # np.set_printoptions(4)
    # fit_single_subject(14)
    model = fit_behavioral_data()
    model.to_pickle('model.pkl')

if __name__ == '__main__':
    card_cue_bandit_experiment(alpha=0.1, beta=0.5)
