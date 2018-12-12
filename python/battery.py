import numpy as np
import cvxpy as cvx
from _battery_controller import BatteryController
from forecasters import RELM
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SECS_IN_15M = 60*15
SECS_IN_1Y = 365*24*3600
YEARS_LIFETIME = 20


class Battery:
    def __init__(self, pars, dataset, cap_nom=810, c_nom = 1.44):
        '''
        Battery model identified in "Achieving the Dispatchability of Distribution
        Feeders through Prosumers Data Driven Forecasting and Model Predictive Control of Electrochemical
        Storage, IEEE TRANS. ON SUSTAINABLE ENERGY, TO BE PUBLISHED (2016)."  based on the SOC. The model is rescaled
        based on the nominal battery capacity.
        :param pars:
        :param dataset: a dictionary containing
        :param cap_nom:
        :param c_nom:
        '''

        self.cap_ref = 810  # reference capacity of the identified battery in Ah
        self.c_ref = 720/500  # reference c factor of the identified battery
        self.e_ref = 500 # reference energy in kWh

        self.e_nom = self.e_ref*cap_nom/self.cap_ref # rescale capacity to get nominal energy

        self.battery_controller_pars = {'e_n': self.e_nom,
                                        'h':pars['h'],
                                        'c': c_nom,
                                        'ts': pars['ts'],
                                        'tau_sd': SECS_IN_1Y,
                                        'dod': 0.8,
                                        'eta_in': 0.98,
                                        'eta_out': 0.98,
                                        'pb': pars['pb'],
                                        'ps': pars['ps'],
                                        'type': pars['type'],
                                        'alpha': pars['alpha'],
                                        'rho': pars['rho'],
                                        'n_final_scens': pars['n_final_scens'],
                                        'n_init_scens': pars['n_init_scens']
                                }

        self.cap_nom = cap_nom # nominal capacity in Ah
        self.c_nom = c_nom # nominal c factor
        self.id_pars = self._get_id_pars()
        self.SOC_init = 0.5  # initial battery state of charge
        self.SOC_vect = np.array([0.1,0.3,0.45,0.7,0.9]) # array of identified SOC points (from the article)
        self.dt = pars['ts']
        self.states_init = self.SOC_init*np.ones((3,1)) # battery states
        self.v_init = self.id_pars['E'][2] # battery potential @ 0.5 SOC

        # simulation variables
        self.i = None
        self.states = None
        self.v = None
        self.v_old = None
        self.SOC_0 = None
        self.SOC = None
        self.P = None
        self.P_asked = None
        self.simulator = None

        # battery constraints
        self.i_max = self.cap_nom*self.c_nom
        self.i_min = 0
        self.v_max = 810
        self.v_min = 510
        self.SOC_max = 1
        self.SOC_min = 1-self.battery_controller_pars['dod']

        # retrieve the system matrices based on the SOC
        self.A = []
        self.B = []
        self.K = []
        self.D = []
        self.G = []
        self.A_soc = None
        self.B_soc = None
        self.K_soc = None
        self.D_soc = None
        self.G_soc = None
        for i in np.arange(len(self.id_pars['R_s'])):
            # continuous matrices
            Ac = np.eye(3) * np.array(
                [-1 / (self.id_pars["R_1"][i] * self.id_pars["C_1"][i]), -1 / (self.id_pars["R_2"][i] * self.id_pars["C_2"][i]),
                 -1 / (self.id_pars["R_3"][i] * self.id_pars["C_3"][i])])
            Bc = np.hstack(
                [np.array([1 / self.id_pars["C_1"][i], 1 / self.id_pars["C_2"][i], 1 / self.id_pars["C_3"][i]]).reshape(-1, 1),
                 np.zeros((3, 1))])

            # exactly discretized matrices
            self.A.append(np.asanyarray(np.exp(self.dt*Ac),float))
            self.B.append((np.linalg.inv(Ac).dot(self.A[i] - 1)).dot(Bc).reshape(3, -1))
            self.K.append(np.diag([self.id_pars["k_1"][i],self.id_pars["k_2"][i],self.id_pars["k_3"][i]]))
            self.D.append(np.array([self.id_pars["R_s"][i],self.id_pars["E"][i]]).reshape(1,-1))
            self.G.append(self.id_pars["sigma2"][i]**0.5)

        # build simulator and initialize variables

        # build battery controller for each set of matrices
        self.battery_simulators= []
        for i in np.arange(len(self.id_pars['R_s'])):
            self.A_soc = self.A[i]
            self.B_soc = self.B[i]
            self.K_soc = self.K[i]
            self.D_soc = self.D[i]
            self.G_soc = self.G[i]
            simulator_i = self._battery_simulator()
            self.battery_simulators.append(simulator_i)

        # initialize variabels
        self.states_old.value = self.states_init
        self.SOC_0.value = np.array([self.SOC_init])
        self.v_old.value = np.array([self.v_init])

        self.battery_controller = BatteryController(dataset, self.battery_controller_pars)

    def call_controller(self,Pm):
        '''
        Call the battery controller and get requested power. Retrieve available power and superimpose it to current
        power profile.
        :return:
        '''



    def simulate(self,P_asked):
        '''
        Simulate a battery model identified in "Achieving the Dispatchability of Distribution
        Feeders through Prosumers Data Driven Forecasting and Model Predictive Control of Electrochemical
        Storage, IEEE TRANS. ON SUSTAINABLE ENERGY, TO BE PUBLISHED (2016)."  based on the SOC. The model is rescaled
        based on the nominal battery capacity.
        '''

        # find which parameters must be used, based on SOC
        ss_idx = int(np.argmin(np.abs(self.SOC-self.SOC_vect)))

        # find current setpoint, given P
        self.P.value = P_asked
        self.simulators[ss_idx].solve()

        # update old voltage
        self.v_old.value = self.v.value
        # update SOC
        self.SOC_0.value = self.SOC.value
        # update states
        self.states_old.valeu = self.states.value

        # return actual P and SOC
        P =  self.P.value
        SOC = self.SOC.value
        states = self.states.value

        return P,SOC,states


    def _get_id_pars(self):
        '''
        The parameters are rescaled for the nominal capacity of the battery
        :return:
        '''
        pars = {"E": [592.2, 625.0, 652.9, 680.2, 733.2],
                "R_s": [0.029,0.021,0.015,0.014,0.013],
                "R_1": [0.095,0.075,0.090,0.079,0.199],
                "C_1": [8930,9809,13996,9499,11234],
                "R_2": [0.04,0.009,0.009,0.009,0.010],
                "C_2": [909,2139,2482,2190,2505],
                "R_3": [2.5e-3,4.9e-5,2.4e-4,6.8e-4,6.0e-4],
                "C_3": [544.2,789.0,2959.7,100.2,6177.3],
                "k_1": [0.639,0.677,0.617,0.547,0.795],
                "k_2": [ - 5.31, - 0.22, - 0.36, - 0.28,0.077],
                "k_3": [5.41,40,0.40,2.83,- 0.24],
                "sigma2":  [- 1.31,- 0.42,0.3426,3.5784,2.7694]
                }
        return pars


    def _battery_simulator(self):

        i = cvx.Variable(1)
        states = cvx.Variable((3,1))
        v = cvx.Variable(1)
        P = cvx.Variable(1)
        SOC = cvx.Variable(1)

        v_old = cvx.Parameter(1)
        P_asked = cvx.Parameter(1)
        SOC_old = cvx.Parameter(1)
        states_old = cvx.Parameter((3,1))

        # current constraints
        constraints = [i <= self.i_max]
        constraints.append(i>= self.i_min)

        # voltage constraints
        constraints.append(v <= self.v_max)
        constraints.append(v >= self.v_min)
        # SOC constraints
        constraints.append(SOC <= self.SOC_max)
        constraints.append(SOC >= self.SOC_min)
        # dynamics - v
        constraints.append(states == self.A_soc*(states_old) + self.B_soc[:,[0]]*i)
        constraints.append(v == cvx.sum(states) + self.D_soc[0,0]*i + self.D_soc[0,1] + self.G_soc*np.random.randn(1))

        # dynamics - SOC
        constraints.append(SOC == SOC_old + i*self.dt/self.c_nom/3600)

        # constraint on requested energy
        constraints.append(P == (v+v_old)*i/2)
        constraints.append(cvx.abs(P) <= cvx.abs(P_asked))
        simulator = cvx.Problem(cvx.Maximize(cvx.abs(i)), constraints)

        self.i = i
        self.states = states
        self.states_old = states_old
        self.v = v
        self.v_old = v_old
        self.SOC_0 = SOC_old
        self.SOC = SOC
        self.P = P
        self.P_asked = P_asked

        return simulator

