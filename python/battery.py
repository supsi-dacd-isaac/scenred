import numpy as np
import cvxpy as cvx
from _battery_controller import BatteryController
import scipy as sp
from forecasters import RELM
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        self.cap_nom = cap_nom # nominal capacity in Ah
        self.scaling_factor = self.cap_ref / cap_nom  # scaling factor for the ss parameters
        self.c_ref = 720/500  # reference c factor of the identified battery
        self.c_nom = c_nom # nominal c factor
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


        self.id_pars = self._get_id_pars()
        self.SOC_init = np.atleast_1d(0.5)  # initial battery state of charge
        self.SOC_vect = np.array([0.1,0.3,0.45,0.7,0.9]) # array of identified SOC points (from the article)
        self.dt = pars['ts']
        self.states_init = self.SOC_init*np.ones((3,1)) # battery states
        self.v_init = np.atleast_1d(self.id_pars['E'][2]) # battery potential @ 0.5 SOC

        # simulation variables
        self.i = []
        self.states = []
        self.states_old = []
        self.v = []
        self.v_old = []
        self.SOC_old = []
        self.SOC = []
        self.P = []
        self.P_asked = []
        self.randn = []
        self.simulator = []

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
            Ac = np.diag(np.array(
                [-1 / (self.id_pars["R_1"][i] * self.id_pars["C_1"][i]), -1 / (self.id_pars["R_2"][i] * self.id_pars["C_2"][i]),
                 -1 / (self.id_pars["R_3"][i] * self.id_pars["C_3"][i])]))
            Bc = np.hstack(
                [np.array([1 / self.id_pars["C_1"][i], 1 / self.id_pars["C_2"][i], 1 / self.id_pars["C_3"][i]]).reshape(-1, 1),
                 np.zeros((3, 1))])
            # rescale for nominal capacity
            Bc = Bc * self.scaling_factor

            # exactly discretized matrices - the first self.dt is the smallest timestep
            self.A.append(np.asanyarray(sp.linalg.expm(self.dt[0]*Ac/10),float))
            self.B.append((np.linalg.inv(Ac).dot(self.A[i] - 1)).dot(Bc).reshape(3, -1))
            self.K.append(np.diag([self.id_pars["k_1"][i],self.id_pars["k_2"][i],self.id_pars["k_3"][i]]))
            # in order to keep the ratio of Joule loss constant, we rescale for the scaling factor
            self.D.append(np.array([self.id_pars["R_s"][i]* self.scaling_factor,self.id_pars["E"][i]]).reshape(1,-1))
            self.G.append(np.sign(self.id_pars["sigma2"][i])*np.abs(self.id_pars["sigma2"][i])**0.5)

        # build simulator and initialize variables

        # build battery controller for each set of matrices
        self.battery_simulators= []
        for i in np.arange(len(self.id_pars['R_s'])):
            self.A_soc = self.A[i]
            self.B_soc = self.B[i]
            self.K_soc = self.K[i]
            self.D_soc = self.D[i]
            self.G_soc = self.G[i]
            simulator_i, current,states,states_old,v,v_old,SOC_old,SOC,P,P_asked,randn= self._battery_simulator()
            self.i.append(current)
            self.states.append(states)
            states_old.value = self.states_init
            self.states_old.append(states_old)
            self.v.append(v)
            v_old.value = self.v_init
            self.v_old.append(v_old)
            SOC_old.value = self.SOC_init
            self.SOC_old.append(SOC_old)
            self.SOC.append(SOC)
            self.P.append(P)
            self.P_asked.append(P_asked)
            self.randn.append(randn)

            self.battery_simulators.append(simulator_i)
            self.SOC[i].value = np.atleast_1d(self.SOC_init)

        # initialize variabels
        #self.SOC_old.value = np.array([self.SOC_init])
        #self.v_old.value = np.array([self.v_init])
        #self.states_old.value = self.states_init

        self.battery_controller = BatteryController(dataset, self.battery_controller_pars)
        self.history = {}
        self.history['P_cont'] = []
        self.history['P_cont_real'] = []
        self.history['P_uncont'] = []
        self.history['SOC_real'] = []

    def do_mpc(self):
        0

    def solve_step(self,time,coord=False):
        '''
        Call the battery controller and get requested power. Retrieve available power and superimpose it to current
        power profile.
        :return:
        '''

        P_controlled, P_uncontrolled, U, E = self.battery_controller.solve_step(time,coord = coord)
        P_battery,SOC_real,states = self.simulate(np.atleast_1d(U[0,0]-U[0,1]))
        P_final = P_uncontrolled[0] + P_battery
        self.history['P_cont'].append(P_controlled[0][0])
        self.history['P_cont_real'].append(P_final[0])
        self.history['P_uncont'].append(P_uncontrolled[0])
        self.history['SOC_real'].append(SOC_real[0])

        return P_final, P_controlled, P_uncontrolled, U, E


    def simulate(self,P_asked):
        '''
        Simulate a battery model identified in "Achieving the Dispatchability of Distribution
        Feeders through Prosumers Data Driven Forecasting and Model Predictive Control of Electrochemical
        Storage, IEEE TRANS. ON SUSTAINABLE ENERGY, TO BE PUBLISHED (2016)."  based on the SOC. The model is rescaled
        based on the nominal battery capacity.
        '''

        # find which parameters must be used, based on SOC
        ss_idx = int(np.argmin(np.abs(self.SOC[0].value-self.SOC_vect)))

        # find current setpoint, given P
        self.P_asked[ss_idx].value = P_asked
        self.randn[ss_idx].value = np.random.randn(1)
        self.battery_simulators[ss_idx].solve()

        for i in np.arange(len(self.v_old)):
            # update old voltage
            self.v_old[i].value = self.v[ss_idx].value
            self.v[i].value = self.v[ss_idx].value

            # update SOC
            self.SOC_old[i].value = self.SOC[ss_idx].value
            self.SOC[i].value = self.SOC[ss_idx].value

            # update states
            self.states_old[i].value = self.states[ss_idx].value
            self.states[i].value = self.states[ss_idx].value

        # return actual P and SOC
        P =  self.P[ss_idx].value
        SOC = self.SOC[ss_idx].value
        states = self.states[ss_idx].value

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
                "k_2": [ -5.31, -0.22, -0.36, -0.28,0.077],
                "k_3": [5.41,40,0.40,2.83,-0.24],
                "sigma2":  [-1.31, -0.42,0.3426,3.5784,2.7694]
                }
        return pars


    def _battery_simulator(self):

        i = cvx.Variable(1)
        states = cvx.Variable((3,1))
        v = cvx.Variable(1)
        P = cvx.Variable(1)
        SOC = cvx.Variable(1)
        one_vec = np.ones((1,3))

        v_old = cvx.Parameter(1)
        P_asked = cvx.Parameter(1)
        SOC_old = cvx.Parameter(1)
        states_old = cvx.Parameter((3,1))
        randn = cvx.Parameter(1)

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
        constraints.append(states == self.A_soc*states_old + self.B_soc[:,[0]]*i)
        constraints.append(v == one_vec*states + self.D_soc[0,0]*i + self.D_soc[0,1] + self.G_soc*randn)

        # dynamics - SOC
        constraints.append(SOC == SOC_old + i*self.dt/self.cap_nom/3600)

        # constraint on requested energy
        constraints.append(P == v_old*i )
        #constraints.append(2*P >= (one_vec*states  + self.D_soc[0,1] +v_old)*i + self.D_soc[0,0]*cvx.sum_squares(i))

        #constraints.append(
        #    P >= ((one_vec * states + self.D_soc[0, 0] * i + self.D_soc[0, 1] + self.G_soc * randn) + v_old) * i / 2)

        #constraints.append(P ==  v_old* i )
        constraints.append(P <= P_asked)
        simulator = cvx.Problem(cvx.Maximize(i), constraints)



        return simulator,i,states,states_old,v,v_old,SOC_old,SOC,P,P_asked,randn


    def online_stochastic_plot(self,N):

        fig, ax = plt.subplots(2)
        dims = self.battery_controller.scen_idxs.shape
        l_unc = ax[0].plot(np.zeros(dims), alpha=0.2, linewidth=0.5)
        l_cont = ax[0].plot(np.zeros(dims), alpha=0.2, linewidth=0.5)
        l_con_past = ax[0].plot(np.ones(dims[0]).reshape(-1, 1), np.nan * np.ones(dims))
        l_real_past = ax[0].plot(np.ones(dims[0]).reshape(-1, 1), np.nan * np.ones(dims), '+')
        l_unc_past = ax[0].plot(np.ones(dims[0]).reshape(-1, 1), np.nan * np.ones(dims))

        p_in = ax[1].plot(np.zeros(dims), alpha=0.2, linewidth=0.5)
        p_out = ax[1].plot(np.zeros(dims), alpha=0.2, linewidth=0.5)
        e_batt = ax[1].plot(np.zeros(dims), alpha=0.2, linewidth=0.5)
        soc_real_past = ax[1].plot(np.ones(dims[0]).reshape(-1, 1), np.nan * np.ones(dims))

        xdata = np.arange(dims[0]).reshape(-1, 1)
        cmap = plt.get_cmap('Set1')
        line_colors = cmap(np.linspace(0, 1, 5))
        x_past = np.arange(1 - dims[0], 1)
        past_data_1 = np.nan * np.ones(dims[0])
        past_data_2 = np.nan * np.ones(dims[0])
        past_data_3 = np.nan * np.ones(dims[0])
        past_data_4 = np.nan * np.ones(dims[0])

        min_y = np.sign(self.battery_controller.y_min) * np.abs(self.battery_controller.y_min) * 1.1
        max_y = np.sign(self.battery_controller.y_max) * np.abs(self.battery_controller.y_max) * 1.1

        def animate(i):
            P_final, P_controlled, P_uncontrolled, U, E = self.solve_step(time=i)
            PS = P_uncontrolled

            for s in np.arange(self.battery_controller.scen_idxs.shape[1]):
                Udata = U[self.battery_controller.scen_idxs[:, s].ravel(), :]
                E_n = E[self.battery_controller.scen_idxs[:, s].ravel(), :]
                Pundata = PS[self.battery_controller.scen_idxs[:, s].ravel()]
                Pcontdata = Pundata + Udata[:, [0]] - Udata[:, [1]]

                l_unc[s].set_data(xdata, Pundata)
                l_unc[s].set_color(line_colors[0, :])
                l_cont[s].set_data(xdata, Pcontdata)
                l_cont[s].set_color(line_colors[1, :])
                p_in[s].set_data(xdata, Udata[:, [0]])
                p_in[s].set_color(line_colors[2, :])
                p_out[s].set_data(xdata, Udata[:, [1]])
                p_out[s].set_color(line_colors[3, :])
                e_batt[s].set_data(xdata, E_n)
                e_batt[s].set_color(line_colors[4, :])

            past_data_1[dims[0] - np.minimum(len(np.hstack(self.history['P_cont'])), dims[0]):] = np.hstack(
                self.history['P_cont'])[-dims[0]:]
            past_data_2[dims[0] - np.minimum(len(np.hstack(self.history['P_cont_real'])), dims[0]):] = np.hstack(
                self.history['P_cont_real'])[-dims[0]:]
            past_data_3[dims[0] - np.minimum(len(np.hstack(self.history['P_uncont'])), dims[0]):] = np.hstack(
                self.history['P_uncont'])[-dims[0]:]
            past_data_4[dims[0] - np.minimum(len(np.hstack(self.history['SOC_real'])), dims[0]):] = np.hstack(
                self.history['SOC_real'])[-dims[0]:]

            l_con_past[0].set_data(x_past, past_data_1)
            l_con_past[0].set_color(line_colors[1, :])
            l_real_past[0].set_data(x_past, past_data_2)
            l_real_past[0].set_color(line_colors[1, :])
            l_unc_past[0].set_data(x_past, past_data_3)
            l_unc_past[0].set_color(line_colors[0, :])
            ax[0].set_xlim(-dims[0], dims[0])
            ax[0].set_ylim(min_y, max_y)
            ax[0].set_ylabel('power [kW]')
            ax[0].set_xlabel('timestep [-]')

            soc_real_past[0].set_data(x_past, past_data_4)
            soc_real_past[0].set_color(line_colors[4, :])

            ax[1].set_xlim(-dims[0], dims[0])
            ax[1].set_ylim(0, self.e_nom*1.1)
            ax[1].set_ylabel('power [kW]')
            ax[1].set_xlabel('timestep [-]')

            ax[1].figure.canvas.draw()
            ax[0].figure.canvas.draw()
            print(i)

        ani = matplotlib.animation.FuncAnimation(fig, animate, frames=N, repeat=False)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('animation.mp4', writer=writer, dpi=300)
        plt.show()