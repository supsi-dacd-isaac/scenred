import numpy as np
from scenred import scenred
class RELM:
    '''
    RELM forecaster
    '''
    def __init__(self,X_te=None,n_elms=100,nodes=200,var_ratio=0.7,obs_ratio=0.3,lamb=1,activation='fast_sigmoid',initialization='uniform',q_vect=(0.25,0.5,0.74),scenarios_per_step=()):
        self.n_elms = n_elms
        self.nodes = nodes
        self.var_ratio = var_ratio
        self.obs_ratio = obs_ratio
        self.lamb = lamb
        self.activation = activation
        self.initialization = initialization
        self.q_vect = q_vect
        self.relms=[]
        self.n_target = 0
        self.scenarios_per_step = scenarios_per_step
        self.X_te = X_te

    def train(self,X,Y):
        n_obs,n_var = X.shape
        n_var_to_use = np.asanyarray(self.var_ratio * n_var,dtype=int)
        n_obs_to_use = np.asanyarray(self.obs_ratio * n_obs,dtype=int)
        self.n_target = Y.shape[1]
        print('Training RELM')
        relms= []
        for i in np.arange(self.n_elms):
            # bagging
            var_idx = np.random.permutation(n_var)
            var_idx = var_idx[0:n_var_to_use+1]
            obs_idx = np.random.permutation(n_obs)
            obs_idx = obs_idx[0:n_obs_to_use+1]
            X_i = X[obs_idx.reshape(-1,1), var_idx.reshape(1,-1)]
            y_i = Y[obs_idx,:]
            # Fit ith ELM
            elm = Elm()
            elm.train(X_i, y_i, self.nodes, self.lamb, self.initialization)
            relms.append(elm)
            relms[i].var_idx = var_idx
            if np.mod(round((i / self.n_elms) * 100), 10) == 0:
                print('*')

        self.relms = relms

    def predict(self,time):
        X_te = self.X_te[np.atleast_1d(time),:]
        y_i = np.zeros((X_te.shape[0],self.n_target,self.n_elms))
        for i in np.arange(self.n_elms):
            X_i = X_te[:,self.relms[i].var_idx]
            y_i[:,:,i] = self.relms[i].predict(X_i)

        y_hat = np.mean(y_i,2).reshape(-1,1)
        quantiles = np.quantile(y_i,self.q_vect,2) # n_sa*n_obs*n_quantiles
        quantiles = np.moveaxis(quantiles, 0, 2)
        return y_hat,quantiles,y_i

    def predict_scenarios(self,time):
        '''
        :param X_te: a vector containing regressors for a single timestep
        :return: a scenario tree of future values
        '''
        X_te = self.X_te[np.atleast_1d(time), :]
        y_i = np.zeros((self.n_target,self.n_elms))
        for i in np.arange(self.n_elms):
            X_i = X_te[:,self.relms[i].var_idx]
            y_i[:,i] = self.relms[i].predict(X_i)

        [S_s, P_s, J_s, Me_s, g] = scenred(np.copy(y_i).reshape(y_i.shape[0],y_i.shape[1],1), metric='cityblock',
                                           nodes=self.scenarios_per_step)

        return g,S_s

class Elm():
    def __init__(self):
        self.W = []
        self.Wr = []
        self.br = []
        self.rse = []
        self.means = []
        self.var_idx=[]
        der_0  =np.exp(0)/(np.exp(0)+1)**2
        self.activation = lambda x: np.maximum(np.minimum(0.5 + der_0 * x, 1), 0)

    def train(self, X, y, n_hidden, lamb , initialization):

        n_obs,n_regr = X.shape
        [X, rse, means] = self.normalize(X)

        if initialization == 'uniform':
            Wr = np.random.rand(n_hidden, n_regr) - 0.5
            br = np.random.rand(n_hidden, 1) - 0.5
        elif initialization =='normal':
            Wr = np.random.randn(n_hidden, n_regr)
            br = np.random.randn(n_hidden, 1)
        else:
            assert 1>2, 'error: initialization type not understood'

        # compute activations matrix
        A = Wr.dot(X.T) +np.tile(br,(1,n_obs))
        # compute first layer output
        H = self.activation(A.T)
        # Train with regularization
        W = np.linalg.pinv(H.T.dot(H)+lamb*np.eye(H.shape[1])).dot( H.T.dot(y))

        self.W = W
        self.Wr = Wr
        self.br = br
        self.rse = rse
        self.means = means

    def predict(self, X):
        y = self.activation((self.Wr.dot(self.renormalize( X, self.rse, self.means).T)+np.tile(self.br,(1,X.shape[0]))).T).dot(self.W)
        return y

    def normalize(self, X):
        [n, p] = X.shape
        means_0 = np.mean(X,0)
        means = np.tile(means_0, (n, 1))
        sX = X - means
        rse = 1.0 / np.std(sX,0)
        sX = sX * rse
        return sX, rse, means_0

    def renormalize(self, X, rse, means):
        sX = X - means
        sX = sX * rse
        return sX


class HoltWinters:
    """
    Holt_Winters class

    Attributes:
    """

    def __init__(self, alpha=0.1, beta=1e-5, gamma=np.array([0.3]), period=np.array([96]),
                 horizon=np.array(range(1, 97)), method='add'):
        """
        Constructor
        :param alpha:
        :type alpha: float
        :param beta:
        :type bety: float
        :param gamma:
        :type gamma: numpy ndarray
        :param period:
        :type period: numpy ndarray
        :param horizon
        :type horizon: numpy ndarray
        :param method:
        :type method: str
        """

        assert 0 < period.size <= 2, "wrong period size (the class accepts 1 or 2 seasons)"
        assert period.size == gamma.size, "gamma and period must have the same length"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.period = period
        self.horizon = horizon

        self.am1 = 0
        self.bm1 = 0
        self.s1mp = np.zeros(self.period[0])
        if self.period.size == 2:
            self.s2mp = np.zeros(self.period[1])

    def train(self, y_history):
        """
        Fit
        :param y_history: y history
        :type y_history: numpy ndarray
        """
        if self.method == 'add':
            self._fit_add(y_history)

    def _fit_add(self, y_history):
        """
        Fit additive
        :param y_history: y history
        :type y_history: numpy ndarray
        """
        if self.period.size == 1:
            a = np.zeros(y_history.size)
            b = np.zeros(y_history.size)
            s = np.zeros(y_history.size)
            for i in range(max(self.period), y_history.size):
                a[i] = self.alpha * (y_history[i] - s[i - self.period[0]]) + (1 - self.alpha) * (
                            a[i - 1] + b[i - 1])
                b[i] = self.beta * (a[i] - a[i - 1]) + (1 - self.beta) * b[i - 1]
                # s[i] = self.gamma[0] * (y_history[i] - a[i-1] - b[i-1]) + (1 - self.gamma[0]) * s[i - self.period[0]]
                s[i] = self.gamma[0] * (y_history[i] - a[i]) + (1 - self.gamma[0]) * s[i - self.period[0]]
                # y_hat = np.zeros(self.horizon.size)
                # for j, h in enumerate(self.horizon):
                #    y_hat[j] = a[i] + h * b[i] + s[i - self.period[0] + (h - 1) % self.period[0]+1]
            self.am1 = a[-1]
            self.bm1 = b[-1]
            self.s1mp = s[-self.period[0]:]
        else:
            a = np.zeros(y_history.size)
            b = np.zeros(y_history.size)
            s1 = np.zeros(y_history.size)
            s2 = np.zeros(y_history.size)
            for i in range(max(self.period), y_history.size):
                a[i] = self.alpha * (y_history[i] - s1[i - self.period[0]] - s2[i - self.period[1]]) + (
                            1 - self.alpha) * (a[i - 1] + b[i - 1])
                b[i] = self.beta * (a[i] - a[i - 1]) + (1 - self.beta) * b[i - 1]
                s1[i] = self.gamma[0] * (y_history[i] - a[i] - s2[i - self.period[1]]) + (1 - self.gamma[0]) * \
                        s1[i - self.period[0]]
                s2[i] = self.gamma[1] * (y_history[i] - a[i] - s1[i - self.period[0]]) + (1 - self.gamma[1]) * \
                        s2[i - self.period[1]]
            self.am1 = a[-1]
            self.bm1 = b[-1]
            self.s1mp = s1[-self.period[0]:]
            self.s2mp = s2[-self.period[1]:]

    def predict(self, y):
        """
        Predict
        :param y: y
        :type y: numpy ndarray
        """
        if self.method == 'add':
            return self._predict_add(y)

    def _predict_add(self, y):
        """
        Predict additive
        :param y: y
        :type y: numpy ndarray
        """
        y_hat = np.zeros(self.horizon.size)
        if self.period.size == 1:
            a = self.alpha * (y - self.s1mp[0]) + (1 - self.alpha) * (self.am1 + self.bm1)
            b = self.beta * (a - self.am1) + (1 - self.beta) * self.bm1
            # s = self.gamma[0] * (y - self.am1 -self.bm1) + (1 - self.gamma[0]) * self.s1mp[0]
            s = self.gamma[0] * (y - a) + (1 - self.gamma[0]) * self.s1mp[0]
            self.s1mp = np.roll(self.s1mp, -1)
            self.s1mp[-1] = s
            self.am1 = a
            self.bm1 = b
            for i, h in enumerate(self.horizon):
                y_hat[i] = a + h * b + self.s1mp[(h - 1) % self.period[0]]
        else:
            a = self.alpha * (y - self.s1mp[0] - self.s2mp[0]) + (1 - self.alpha) * (self.am1 + self.bm1)
            b = self.beta * (a - self.am1) + (1 - self.beta) * self.bm1
            s1 = self.gamma[0] * (y - a - self.s2mp[0]) + (1 - self.gamma[0]) * self.s1mp[0]
            s2 = self.gamma[1] * (y - a - self.s1mp[0]) + (1 - self.gamma[1]) * self.s2mp[0]
            self.s1mp = np.roll(self.s1mp, -1)
            self.s1mp[-1] = s1
            self.s2mp = np.roll(self.s2mp, -1)
            self.s2mp[-1] = s2
            self.am1 = a
            self.bm1 = b
            for i, h in enumerate(self.horizon):
                y_hat[i] = a + h * b + self.s1mp[(h - 1) % self.period[0]] + self.s2mp[(h - 1) % self.period[1]]
        return y_hat

class pre_trained_forecaster:
    def __init__(self,dataset,scenarios_per_step):
        '''
        :param dataset: dict containing future scenarios, in a n_obs*n_sa*n_scens ndarray
        '''

        assert 'scenarios' in dataset.keys(), 'Error, dataset must contain a scenario tensor'
        assert len(dataset['scenarios'].shape)==3, 'Error, scenarios in dataset must be a 3-tensor'
        assert 'y_hat' in dataset.keys(), 'Error, dataset must contain a y_hat matrix'
        assert len(dataset['y_hat'].shape)==2, 'Error, y_hat in dataset must be a 2-matrix'
        self.scenarios = dataset['scenarios']
        self.y_hat = dataset['y_hat']
        self.scenarios_per_step = scenarios_per_step

    def train(self,X,y):
        return 0

    def predict(self,time):
        y_hat = self.y_hat[time,:]
        return y_hat

    def predict_scenarios(self,time):
        scen_t = np.squeeze(self.scenarios[time,:,:])
        [S_s, P_s, J_s, Me_s, g] = scenred(np.copy(scen_t).reshape(scen_t.shape[0],scen_t.shape[1],1), metric='cityblock',
                                           nodes=self.scenarios_per_step)
        return g, S_s

