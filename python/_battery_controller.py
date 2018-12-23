# --------------------------------------------------------------------------- #
# Import section
# --------------------------------------------------------------------------- #
import numpy as np
import scipy.sparse as sparse
import cvxpy as cvx
import scipy as sp
from functools import reduce
from scenred import scenred
import forecasters
import networkx as nx
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SECS_IN_15M = 60*15
SECS_IN_1Y = 365*24*3600
YEARS_LIFETIME = 20

# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
def controller_pars_builder(e_n: float=1,
                            c: float=1,
                            ts: int=SECS_IN_15M,
                            tau_sd: int=SECS_IN_1Y,
                            lifetime: int=YEARS_LIFETIME,
                            dod: float=0.8,
                            nc: float=3000,
                            eta_in: float=0.87,
                            eta_out: float=0.87,
                            h: float=48,
                            pb: float=0.2,
                            ps: float=0.07,
                            type: str='peak_shaving',
                            alpha: float=1,
                            rho: float=0.5,
                            n_final_scens: int=10,
                            n_init_scens: int = 5):
    """
    :param e_n: nominal capacity of the battery [kWh]
    :type e_n: double
    :param c: C factor [kW/kWh]
    :type c: double
    :param ts: sampling time for the problem solution [s]
    :type ts: int        
    :param tau_sd: time constant for self-discharge [s]
    :type tau_sd: int
    :param lifetime: lifetime of the battery [years]
    :type lifetime: int
    :param dod: depth of discharge [-]
    :type dod: int
    :param nc: maximum cycles before removal [-]
    :type nc: double
    :param eta_in: charge efficiency [-]
    :type eta_in: double
    :param eta_out: discharge efficiency [-]
    :type eta_out: double
    :param h:
    :type h: float
    :param pb:
    :type pd: float
    :param ps:
    :type ps: float
    :param type:
    :type type: str
    :param alpha:
    :type alpha: float
    :param rho:
    :type rho: float
    :param g:
    :type g: networkx DiGraph
    """
    return {'E_n': e_n,
            'C': c,
            'ts': ts,
            'tau_sd': tau_sd,
            'lifetime': lifetime,
            'DOD': dod,
            'Nc': nc,
            'eta_in': eta_in,
            'eta_out': eta_out,
            'h': h,
            'pb': pb,
            'ps': ps,
            'type': type,
            'alpha': alpha,
            'rho': rho,
            'n_final_scens':n_final_scens,
            'n_init_scens':n_init_scens}

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
class BatteryController:
    def __init__(self, dataset, pars=None, logger=None):
        """
        Constructor
        :param pars: controller parameters
        :type pars: dict
        :param logger: logger object
        :type logger: logger
        """
        if pars is None:
            controller_pars = controller_pars_builder()
        else:
            controller_pars = controller_pars_builder(e_n=pars['e_n'], c=pars['c'], ts=pars['ts'],
                                                      tau_sd=SECS_IN_1Y, lifetime=10000000000,
                                                      dod=pars['dod'], nc=100000000, eta_in=pars['eta_in'],
                                                      eta_out=pars['eta_out'], h=pars['h'],pb=pars['pb'], ps=pars['ps'],
                                                      type=pars['type'], alpha=pars['alpha'],rho=pars['rho'],
                                                      n_final_scens = pars['n_final_scens'],
                                                      n_init_scens = pars['n_init_scens'])

        if 'X_tr' in dataset.keys():
            self.forecaster_type = 'online'
        elif 'scenarios' in dataset.keys():
            self.forecaster_type = 'pre-trained'
        else:
            assert 1==0, 'dataset must contain X_tr or a set of scenarios'

        # number of scenarios per step in case of stochastic control
        N_FINAL_SCENARIOS = pars['n_final_scens']
        N_INIT_SCEN = pars['n_init_scens']

        if self.forecaster_type == 'online':  # train a RELM forecaster
            # create predictor
            scens_per_step = np.linspace(N_INIT_SCEN, N_FINAL_SCENARIOS, dataset['y_tr'].shape[1], dtype=int)
            relm = forecasters.RELM(X_te=dataset['X_te'],scenarios_per_step=scens_per_step, lamb=1e-5)
            relm.train(dataset['X_tr'], dataset['y_tr'])
            self.forecaster = relm
            self.y_max = np.max(dataset['y_tr'].ravel())
            self.y_min = np.min(dataset['y_tr'].ravel())

        else:  # assume that forecasts are available in the dataset
            scens_per_step = np.linspace(N_INIT_SCEN, N_FINAL_SCENARIOS, dataset['scenarios'].shape[1], dtype=int)
            self.forecaster = forecasters.pre_trained_forecaster(dataset,scenarios_per_step = scens_per_step)
            self.y_max = np.max(dataset['scenarios'].ravel())
            self.y_min = np.min(dataset['scenarios'].ravel())


        self.pars = controller_pars
        self.cvx_solver = None
        self.pm = None
        self.ref = None
        self.u = None
        self.dsch_punish = None

        self.cvx_solver_st = None
        self.pm_st = None
        self.ref_st = None
        self.u_st = None
        self.x_st = None
        self.p_st = None
        self.dsch_punish_st = None
        self.scen_idxs = None

        self.x_mat = None
        self.b_mat = None
        self.l_mat = None
        self.cvx_mat_solver = None

        # control horizon
        self.h = self.pars['h']

        # Battery current states, initialized at half maximum capacity
        self.E_0 = self.pars['E_n']
        self.x_start = self.E_0/2
        self.x = None

        # State matrix
        self.Ac = -1/self.pars['tau_sd']

        # Input matrix
        self.Bc = np.array([self.pars['eta_in']/3600, - 1/(self.pars['eta_out'] * 3600)])

        # Exact discretization of Ac
        self.Ad = np.asanyarray(np.exp(self.pars['ts']*self.Ac),float)

        # Exact discretization of Bc
        Bd = []
        if np.size(self.Ad) == 1:
            Bd.append(np.divide((self.Ad - 1) * self.Bc, self.Ac).reshape(1, 2))
        else:
            for i in np.arange(np.size(self.Ad)):
                Bd.append(np.divide((self.Ad[i] - 1)*self.Bc, self.Ac).reshape(1,2))
        self.Bd = np.vstack(Bd)
        # Calendar aging coefficient [1/s]
        self.k_t = -0.2/self.pars['lifetime']*SECS_IN_1Y

        # Coefficient of degradation
        self.k_q = -0.2/(2*self.pars['Nc']*self.pars['E_n']*self.pars['DOD'])

        # Logger
        self.logger = logger

        # build difference matrix for total power from / to the battery
        D = np.zeros((self.h, 2 * self.h))
        for i in np.arange(self.h):
            D[i, i*2: 2 *(i+1)] = [1, -1]

        self.D = D

        # build fixed constraints
        self.x_u = self.E_0
        self.x_l = self.E_0 * (1-self.pars['DOD'])
        self.u_l = np.zeros((1,2)).reshape(-1,1)
        self.u_u = np.ones((1, 2)).reshape(-1,1) * self.E_0 * self.pars['C']

        #-------- Norm 2 matrix ----------
        self.Tcvx = self.D
        self.alpha = self.pars['alpha']
        self.rho = self.pars['rho']

        k = self.alpha / (2 * self.rho)
        self.k = k

        # ------ Build a CVX solver as reference -------
        if self.pars['type'] in ['stochastic','dist_stoc']:
            if self.forecaster_type == 'online':
                g,S_s = self.forecaster.predict_scenarios(0)
            elif self.forecaster_type == 'pre-trained':
                g,S_s = self.forecaster.predict_scenarios(time=0)
            self.g = g
            self.build_stochastic_cvx_solver(k)
        else:
            self.build_cvx_solver(k)

        self.x_start.value = [self.E_0*0.5]

        # ------ Initialize a Holt Winter forecaster -------
        if np.size(self.pars['ts'])>1:
            n_step_per_day = int(86400/self.pars['ts'][0])
        else:
            n_step_per_day = int(86400 / self.pars['ts'])

        #self.forecaster = HoltWinters(alpha=0.05, beta=0, gamma=np.array([0.9,0.9]), period=np.array([1440,1440*7]), horizon=np.array(range(1,n_step_per_day+1)), method='add')
        self.P_hat = None

        self.batch = Batch(self)
        self.x_batch = self.E_0/2

    def build_cvx_solver(self, k):
        """
        Define the CVX solver
        :param k: sum_squares factor
        :type k: float
        """
        p_buy = self.pars['pb']
        p_sell = self.pars['ps']
        H = self.h
        x = cvx.Variable((H + 1,1))
        y = cvx.Variable((H, 1))
        u = cvx.Variable((H, 2))
        pm = cvx.Parameter((H, 1))
        x_start = cvx.Parameter(1)
        one_v = np.ones((1, H))
        constraints = [x[1:] <= self.x_u]
        if np.size(self.Ad)>1:
            assert np.size(self.Ad) == H , 'Error:length of ts must be equal to the horizon'
            for i in np.arange(len(self.Ad)):
                #constraints.append( x[i+1] == self.Ad[i] * x[i] +  u[[i],:] * self.Bd[[i],:].T)
                constraints.append(x[1:] == np.diag(self.Ad) * x[0:-1] + cvx.reshape(cvx.diag(u * self.Bd.T),(len(self.Ad),1)))
        else:
            constraints.append( x[1:] == self.Ad * x[0:-1] +  u * self.Bd.T)
        constraints.append(x[1:] <= self.x_u)
        constraints.append(x[1:] >= self.x_l)
        constraints.append(u[:, 0] >= 0)
        constraints.append(u[:, 1] >= 0)
        constraints.append(u[:, 0] <= self.E_0 * self.pars['C'])
        constraints.append(u[:, 1] <= self.E_0 * self.pars['C'])
        constraints.append(y >= p_buy * (pm + cvx.reshape((u[:, 0] - u[:, 1]),(H,1))))
        constraints.append(y >= p_sell * (pm + cvx.reshape((u[:, 0] - u[:, 1]),(H,1))))
        constraints.append(x[0] == x_start)
        # eco_cost = ct * (pm + (u[0] - u[1]))
        if self.pars['type'] == 'economic':
            cost = one_v * y
            ref = None
            dsch_punish = None
        elif self.pars['type'] in ['stochastic','distributed','peak_shaving','dist_stoc']:
            ref = cvx.Parameter((self.Tcvx.shape[0], 1))
            dsch_punish = cvx.Parameter((1,self.Tcvx.shape[0]))
            u_punish = 1e-6 * cvx.sum_squares(u)
            cost = one_v * y + dsch_punish*u[:,1] + k*cvx.sum_squares(self.Tcvx*cvx.reshape(u.T,(H*2,1))+pm-ref)+u_punish
        else:
            raise TypeError('pars["type"] not recognized')

        # peak_cost = cvx.square(pm + (u[0]-u[1]))
        obj = cost
        prob = cvx.Problem(cvx.Minimize(obj), constraints)

        self.cvx_solver = prob
        self.pm = pm
        self.ref =ref
        self.x_start = x_start
        self.u = u
        self.x = x
        self.dsch_punish = dsch_punish

    def build_stochastic_cvx_solver(self, k):
        """
        Define the CVX solver
        :param k: sum_squares factor
        :type k: float
        """
        g = self.g
        p_buy = self.pars['pb']
        p_sell = self.pars['ps']
        n_n = len(g.nodes)
        node_set = np.linspace(0,n_n-1,n_n,dtype=int)
        x = cvx.Variable((n_n ,1))
        y = cvx.Variable((n_n, 1))
        u = cvx.Variable((n_n, 2))
        pm = cvx.Parameter((n_n, 1))
        p = cvx.Parameter((1,n_n),nonneg=True)
        x_start = cvx.Parameter(1)
        # probabilities vector
        all_t = np.array(list(nx.get_node_attributes(g, 't').values()))
        t = np.unique(all_t)
        leafs = np.array([ n  for n in  node_set[all_t==np.max(t)]])
        x_leafs = cvx.Variable((len(leafs), 1))

        constraints = [x[1:] <= self.x_u]
        constraints.append(x[1:] >= self.x_l)
        constraints.append(x_leafs<= self.x_u)
        constraints.append(x_leafs>= self.x_l)
        scen_idxs_hist = np.zeros((max(t)+1,len(leafs)),dtype=int)
        for s in np.arange(len(leafs)):
            scen_idxs = np.sort(np.array(list(nx.ancestors(g, leafs[s]))))
            scen_idxs = np.asanyarray(np.insert(scen_idxs, len(scen_idxs),leafs[s],0),int)
            scen_idxs_hist[:,s] = scen_idxs
            if np.size(self.Ad) > 1:
                constraints.append(cvx.vstack((x[scen_idxs[1:]],x_leafs[[s]])) == np.diag(self.Ad) * x[scen_idxs] + cvx.reshape(cvx.diag(u[scen_idxs,:] * self.Bd.T), (len(self.Ad), 1)))
                #constraints.append(x[scen_idxs[1:]] == np.diag(self.Ad[0:-1]) * x[scen_idxs[0:-1]] + cvx.reshape(cvx.diag(u[scen_idxs[0:-1],:] * self.Bd.T[:,:-1]), (len(self.Ad)-1, 1)))
            else:
                constraints.append(cvx.vstack((x[scen_idxs[1:]],x_leafs[[s]])) == self.Ad * x[scen_idxs] + u[scen_idxs,:] * self.Bd.T)

        constraints.append(u[:, 0] >= 0)
        constraints.append(u[:, 1] >= 0)
        constraints.append(x[1:] <= self.x_u)
        constraints.append(x[1:] >= self.x_l)

        constraints.append(u[:, 0] <= self.E_0 * self.pars['C'])
        constraints.append(u[:, 1] <= self.E_0 * self.pars['C'])
        constraints.append(y >= p_buy * (pm + u[:, [0]] - u[:, [1]]))
        constraints.append(y >= p_sell * (pm + u[:, [0]] - u[:, [1]]))
        constraints.append(x[0] == x_start)
        # eco_cost = ct * (pm + (u[0] - u[1]))
        ref = cvx.Parameter((n_n, 1))
        dsch_punish = cvx.Parameter((1,n_n))

        batt_punish = dsch_punish * cvx.diag(p) * u[:, [1]]
        u_punish = 1e-6 * (u[:, [1]].T* cvx.diag(p.T) * u[:, [1]] + u[:, [0]].T* cvx.diag(p.T) * u[:, [0]])
        ref_punish = k * p * (u[:, [0]] - u[:, [1]] + pm - ref)**2
        cost = p * y + batt_punish + ref_punish

       # for i in np.arange(n_n):
        #     #u_punish = 1e-6 * p[0,i] * ((u[i, [0]])**2 + (u[i, [1]])**2 )
        #    ref_punish =  k * p[0,i] * (u[i, [0]] - u[i, [1]] + pm[i] - ref[i])**2
        #    cost += ref_punish #+ u_punish

        #u_punish = 1e-6*(cvx.sum_squares(cvx.diag(p.T)*u[:,[0]])+cvx.sum_squares(cvx.diag(p.T)*u[:,[1]]))
        #cost = p*y + dsch_punish*cvx.diag(p)*u[:,1] + k*cvx.sum_squares(cvx.diag(p.T)*(u[:, [0]] - u[:, [1]] +pm -ref)) + u_punish

        #cost = cvx.sum(y) + k * cvx.sum_squares((u[:, [0]] - u[:, [1]] + pm - ref))
        #D = np.zeros((self.h+1, 2 * n_n))
        #for i in np.arange(self.h+1):
        #    D[i, i * 2: 2 * (i + 1)] = [1, -1]
        #cost = k*cvx.sum_squares(D*cvx.reshape(u.T,(n_n*2,1))-ref)

        # peak_cost = cvx.square(pm + (u[0]-u[1]))
        obj = cost
        prob = cvx.Problem(cvx.Minimize(obj), constraints)

        self.cvx_solver_st = prob
        self.pm_st = pm
        self.p_st = p
        self.ref_st =ref
        self.x_start = x_start
        self.u_st = u
        self.x_st = x
        self.dsch_punish_st = dsch_punish
        self.scen_idxs = scen_idxs_hist

    def solve_step(self,time,ref=None,coord=False):

        #P_hat = self.P_hat
        if self.pars['type'] in ['stochastic','dist_stoc']:
            if self.P_hat is None:
                self.P_hat, Ss = self.forecaster.predict_scenarios(time)
            # P must be a networkx graph
            # default reference does peak shaving
            if ref is None:
                #ref = -np.array(list(nx.get_node_attributes(self.P_hat, 'v').values()))
                ref = np.zeros((len(self.P_hat),1))
            try:
                self.p_st.value = np.array(list(nx.get_node_attributes(self.P_hat, 'p').values())).reshape(1,-1)
            except:
                print('vacca maiala')
            self.pm_st.value = np.array(list(nx.get_node_attributes(self.P_hat, 'v').values()))
            self.ref_st.value = ref
            self.dsch_punish_st.value = 1*(np.array(list(nx.get_node_attributes(self.P_hat, 'v').values()))<ref).T
            solution = self.cvx_solver_st.solve()
            if np.size(self.Ad)==1:
                self.x_start.value = self.Ad * self.x_start.value + self.Bd.dot(
                    self.u_st.value[0, :].reshape(-1, 1)).flatten()
            else:
                self.x_start.value = self.Ad[0] * self.x_start.value + self.Bd[[0],:].dot(self.u_st.value[0,:].reshape(-1, 1)).flatten()

            U = self.u_st.value
            p_battery_0 = self.u_st.value[0, 0] - self.u_st.value[0, 1]
            P_hat = np.array(list(nx.get_node_attributes(self.P_hat, 'v').values()))
            SOC = self.x_st.value

        else:
            if self.P_hat is None:
                self.P_hat,quantiles, y_i = self.forecaster.predict(time)
            # default reference does peak shaving
            if ref == None:
                #ref = -self.P_hat
                ref = np.zeros((self.P_hat.shape[0],1))
            self.pm.value = self.P_hat
            if self.pars['type'] == 'peak_shaving':
                self.ref.value = ref
                self.dsch_punish.value = 1*(self.pm.value<ref).T
            solution = self.cvx_solver.solve()
            if np.size(self.Ad)==1:
                self.x_start.value = self.Ad * self.x_start.value + self.Bd.dot(
                    self.u.value[0, :].reshape(-1, 1)).flatten()
            else:
                self.x_start.value = self.Ad[0] * self.x_start.value + self.Bd[[0],:].dot(self.u.value[0,:].reshape(-1, 1)).flatten()

            U = self.u.value
            p_battery_0 = self.u.value[0, 0] - self.u.value[0, 1]
            P_hat = self.P_hat
            SOC = self.x.value

        P_controlled = P_hat+U[:, [0]]-U[:, [1]]

        # reset the forecast if there's not the need to keep it for dstributed coordination scheme
        if not coord:
            self.P_hat = None

        return P_controlled, P_hat, U, SOC

    def solve_batch(self,P_hat):
        """
        Solve a single step
        """
        #P_hat = self.P_hat
        self.pm.value = P_hat
        p_b = np.ones((self.h,1)) * self.pars['pb']
        p_s = np.ones((self.h, 1)) * self.pars['ps']

        if self.pars['type'] == 'peak_shaving':
            r = -P_hat

        method = 'cvx'
        self.u_batch = self.batch.solve_batch(self.x_batch,P_hat,r,method)
        self.e_batch = self.batch.H0*self.x_batch +self.batch.Hu.dot(self.u_batch.reshape(-1,1))
        if np.size(self.Ad) == 1:
            self.x_batch = self.Ad * self.x_batch + self.u_batch[0, :].dot(self.Bd.T)
        else:
            self.x_batch = self.Ad[0] * self.x_batch + self.u_batch[0,:].dot(self.Bd[[0],:].T)
        p_set = self.u_batch[0,0] - self.u_batch[0,1]
        return p_set



class Batch:
    def __init__(self, controller):
        """
        Define optimiyation problem in matrix form
        :param battery: battery
        :type battery: Battery object
        :param x_0: initial state
        :type x_0: double
        :param ct: cost profile
        :type ct: NumPy array
        :param h: number of steps ahead
        :type h: int
        """
        self.controller = controller
        self.p_buy = controller.pars['pb']
        self.p_sell = controller.pars['ps']
        self.D = controller.D
        self.h = controller.h
        self.Ad = controller.Ad
        self.Bd = controller.Bd
        self.U_u = np.kron(np.ones((controller.h, 1)), controller.u_u.reshape(-1, 1))
        self.U_l = np.kron(np.ones((controller.h, 1)), controller.u_l.reshape(-1, 1))
        self.X_u = np.kron(np.ones((controller.h, 1)), controller.x_u)
        self.X_l = np.kron(np.ones((controller.h, 1)), controller.x_l)

        self.H0 = self.build_H0()
        self.Hu = self.build_Hu()
        self.A = self.build_A()
        self.k = controller.k
        self.T = controller.Tcvx
        Z = 1e-16*np.ones((self.h,self.h))
        T = np.hstack([Z,self.T])
        self.Q = self.k*T.T.dot(T)

        self.cvx_mat_solver = None
        self.build_cvx_matrix_solver()


    def build_H0(self):
        '''
        Define matrix X for the optimization
        :param A: discretized state matrix for the controlled battery
        :type A: numpy ndarray
        :param h: horizon of the MPC in steps
        :type: double
        :return: X
        :type: numpy ndarray
        '''
        H0 = np.zeros((self.h, 1),dtype=float)
        r = 1
        if np.size(self.Ad) == 1:
            for ix in np.arange(self.h+1):
                H0[r*ix:(ix+1)*r ,:] = np.power(self.Ad,ix)
        else:
            for ix in np.arange(self.h):
                H0[r * ix :(ix +1)* r, :] = np.power(self.Ad[ix], ix)
        return H0


    def build_Hu(self):
        '''
        Define matrix X for the optimization
        :param A: discretized state matrix for the controlled battery
        :type A: numpy ndarray
        :param B: discretized input matrix for the controlled battery
        :type B: numpy ndarray
        :param h: horizon of the MPC in steps
        :type: double
        :return: M
        :type: numpy ndarray
        '''

        nb = 1
        mb = 2

        Hu = np.zeros((self.h * nb, self.h * mb))

        if np.size(self.Ad) == 1:
            for i in np.arange(self.h):
                matarray = []
                for j in np.arange(self.h-i):
                    matarray.append(np.power(self.Ad, i)*self.Bd)
                ABdiag = sp.linalg.block_diag(*matarray)
                Hu[nb*i:,0: self.h*mb - i*mb] = Hu[nb*i:, 0:self.h*mb -i*mb] + ABdiag
        else:
            for step in np.arange(self.h):
                column = []
                for t in np.arange(self.h-step):
                    if t>0:
                        Aprod = reduce(np.dot,self.Ad[step:step+t])
                    else:
                        Aprod =1

                    column.append(np.dot(Aprod,self.Bd[step]))
                Hu[step*nb:,step*mb:(step+1)*mb] = np.vstack(column)

        return Hu

    def build_A(self):
        E = np.eye(self.h)
        Eu = np.eye(self.h*2)
        Z = np.zeros((self.h,self.h))
        Zu = np.zeros((self.h*2, self.h))

        A1 = np.hstack([E,-self.D*self.p_buy])
        A1 = np.vstack([A1,np.hstack([E,-self.D*self.p_sell])])
        A2 = np.hstack([Zu,Eu])
        A2 = np.vstack([A2,np.hstack([Zu,- Eu])])
        A3 = np.hstack([Z, self.Hu])
        A3 = np.vstack([A3, np.hstack([Z, -self.Hu])])
        A = np.vstack([A1,A2,A3])
        return A

    #@jit
    def build_b(self,x_0,Pm):
        b1 = np.vstack([self.p_buy * Pm, self.p_sell * Pm])
        b2 = np.vstack([self.U_l,-self.U_u])
        b3 = np.vstack([self.X_l-self.H0*x_0,-(self.X_u-self.H0*x_0)])
        b = np.vstack([b1,b2,b3])
        return b

    def solve_batch(self,x_0,Pm,r,method):
        '''
        Solve the batch cost problem min c_t*(Am*x-r_t) s.t. Ain*x<=bin
        :param Ain: inequality matrix for the problem
        :type Ain: numpy ndarray
        :param bin: inequality vector for the problem
        :type bin: numpy array
        :param Am: transformation matrix for the objective
        :type Am: numpy ndarray
        :param X: X matrix
        :type X: numpy ndarray
        :param M: M matrix
        :type M: numpy ndarray
        :param r_t: reference vector for the batch (uncontrolled loads vector)
        :type r_t: numpy array
        :param c_t: cost vector for the batch
        :type c_t: numpy array
        :param method: solver method. WARNING: gurobi is not implemented yet!
        :type method: string in {'cvx','gurobi'}
        :return:u_opt, r_opt, e_opt
        :type: numpy ndarray
        '''

        b = self.build_b(x_0,Pm)
        lin_cost = np.ones((1,self.h))
        dsch_punish = 100*np.tile(np.array([0,1]),(1,self.h))*np.tile(np.asanyarray(r>0,int),(1,2)).reshape(1,-1)
        ref_cost = -2*self.k*r.T.dot(self.T)
        l = np.hstack([lin_cost,dsch_punish+ref_cost])
        z_opt = self.solve_lin_q_prog(self.A,b,l,self.Q,method)
        u_opt = z_opt[self.h:].reshape(-1,2)

        return u_opt

    def solve_lin_q_prog(self,A,b,l,Q, method):
        '''
        Solve cost problem min c_t*(A*x-b) s.t. Ain*x<=bin  using matrix formulation
        :param A: A matrix
        :type A: numpy ndarray
        :param b: b reference vector
        :type b: numpy array
        :param Ain: inequality constraint matrix
        :type Ain: numpy ndarray
        :param bin: inequality constraint vector
        :type bin: numpy array
        :param method: solver method. WARNING: gurobi is not implemented yet!
        :type method: string in {'cvx','gurobi'}
        :return:
        '''

        if method == 'cvx':
            '''x = cvx.Variable((np.shape(A)[1],1))
            obj = l*x + cvx.quad_form(x,Q)
            constraints = [A*x >= b]
            prob = cvx.Problem(cvx.Minimize(obj), constraints)
            solution = prob.solve(verbose=False)
            x_opt = x.value
            '''

            self.controller.b_mat.value = b
            self.controller.l_mat.value = l
            self.cvx_mat_solver.solve(verbose=False)
            x_opt = self.controller.x_mat.value

        elif method == 'quadprog':
            x_opt = self.quadprog_solve_qp(Q, l, G=-A, h=-b.flatten())
            x_opt = x_opt[0:n]
        elif method == 'osqp':
            #prob = osqp.OSQP()
            #P = 1e-16 * sparse.eye(np.shape(A)[1])
            #        prob.setup(P=P,q=f.T,A=A,l=lb,u=ub)
    #        prob.codegen('code',python_ext_name='emosqp')
            emosqp.update_bounds(lb,ub)
            emosqp.update_lin_cost(f.T)
            a = emosqp.solve()
            x_opt = a[0][0:n]

        return x_opt

    def quadprog_solve_qp(self,P, q, G=None, h=None, A=None, b=None):
        qp_G = .5 * (P + P.T)  # make sure P is symmetric
        qp_a = -q
        if A is not None:
            qp_C = -np.vstack([A, G]).T
            qp_b = -np.hstack([b, h])
            meq = A.shape[0]
        else:  # no equality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0
        return quadprog.solve_qp(qp_G, qp_a.T.flatten(), qp_C, qp_b.flatten(), meq)[0]

    def build_cvx_matrix_solver(self):

        A = self.A
        Q = self.Q
        x = cvx.Variable((self.h*3, 1))
        b = cvx.Parameter((self.h * 8, 1))
        l = cvx.Parameter((1,self.h * 3))

        obj = l * x + cvx.quad_form(x, Q)
        constraints = [A * x >= b]
        prob = cvx.Problem(cvx.Minimize(obj), constraints)
        self.cvx_mat_solver = prob
        self.controller.x_mat = x
        self.controller.l_mat = l
        self.controller.b_mat = b
