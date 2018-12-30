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

        # simulation variables for negative power
        self.i_m = []
        self.states_m = []
        self.states_old_m = []
        self.v_m = []
        self.v_old_m = []
        self.SOC_old_m = []
        self.SOC_m = []
        self.P_m = []
        self.P_asked_m = []
        self.randn_m = []
        self.simulator_m = []

        # battery constraints
        self.i_max = self.cap_nom*self.c_nom
        self.i_min = -self.cap_nom*self.c_nom
        self.v_max = 810*1.5
        self.v_min = 510*0.75
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
            #Ac = np.diag(np.array(
            #   [-1 / (self.id_pars["R_1"][i] * self.id_pars["C_1"][i]), -1 / (self.id_pars["R_2"][i] * self.id_pars["C_2"][i])]))
            #Bc = np.hstack(
            #    [np.array([1 / self.id_pars["C_1"][i], 1 / self.id_pars["C_2"][i]]).reshape(-1, 1),
            #    np.zeros((2, 1))])
            # rescale for nominal capacity
            Bc = Bc #* self.scaling_factor

            # exactly discretized matrices - the first self.dt is the smallest timestep
            self.A.append(np.asanyarray(sp.linalg.expm(self.dt[0]*Ac/10),float))
            self.B.append((np.linalg.inv(Ac).dot(self.A[i] - 1)).dot(Bc).reshape(3, -1))
            self.K.append(np.diag([self.id_pars["k_1"][i],self.id_pars["k_2"][i],self.id_pars["k_3"][i]]))
            # in order to keep the ratio of Joule loss constant, we rescale for the scaling factor
            self.D.append(np.array([self.id_pars["R_s"][i],self.id_pars["E"][i]]).reshape(1,-1))
            self.G.append(np.sign(self.id_pars["sigma2"][i])*np.abs(self.id_pars["sigma2"][i])**0.5)

        # build simulator and initialize variables

        # build battery controller for each set of matrices
        self.battery_simulators = []
        self.battery_simulators_m = []

        for i in np.arange(len(self.id_pars['R_s'])):
            A = np.copy(self.A[i])
            B = np.copy(self.B[i])
            K = np.copy(self.K[i])
            D = np.copy(self.D[i])
            G = np.copy(self.G[i])
            simulator_i, current, states, states_old, v, v_old, SOC_old, SOC, P, P_asked, randn = self._battery_simulator_plus(A,B,G,D)
            simulator_i_m, current_m, states_m, states_old_m, v_m, v_old_m, SOC_old_m, SOC_m, P_m, P_asked_m, randn_m = self._battery_simulator_minus(A,B,G,D)
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

            self.i_m.append(current_m)
            self.states_m.append(states_m)
            states_old_m.value = self.states_init
            self.states_old_m.append(states_old_m)
            self.v_m.append(v_m)
            v_old_m.value = self.v_init
            self.v_old_m.append(v_old_m)
            SOC_old_m.value = self.SOC_init
            self.SOC_old_m.append(SOC_old_m)
            self.SOC_m.append(SOC_m)
            self.P_m.append(P_m)
            self.P_asked_m.append(P_asked_m)
            self.randn_m.append(randn_m)

            self.battery_simulators_m.append(simulator_i_m)
            self.SOC_m[i].value = np.atleast_1d(self.SOC_init)
        # initialize variabels
        #self.SOC_old.value = np.array([self.SOC_init])
        #self.v_old.value = np.array([self.v_init])
        #self.states_old.value = self.states_init

        self.battery_controller = BatteryController(dataset, self.battery_controller_pars)
        self.history = {}
        self.history['P_cont'] = []
        self.history['P_cont_real'] = []
        self.history['P_hat'] = []
        self.history['SOC_real'] = []
        self.history['SOC'] = []
        self.history['P_battery'] = []
        self.history['P_battery_real'] = []
        self.history['P_obs'] = []
        self.history['P_obs_cont'] = []

    def do_mpc(self,n_steps,P_obs,coord=False,do_plots=False):

        if do_plots:
            fig,ax = plt.subplots(2)
        for t in np.arange(n_steps):

            # solve horizon
            P_final, P_controlled, P_hat, U, E = self.solve_step(t,coord=coord)

            # update controller state if SOC is equal to min or max SOC
            SOC_dist_min = np.abs(self.history['SOC_real'][-1] - self.SOC_min)
            SOC_dist_max = np.abs(self.history['SOC_real'][-1] - self.SOC_max)
            if SOC_dist_min<1e-3:
                self.battery_controller.x_start.value = np.atleast_1d(self.SOC_min * self.e_nom)
            elif SOC_dist_max<1e-3:
                self.battery_controller.x_start.value = np.atleast_1d(self.SOC_max * self.e_nom)
            self.history['P_obs'].append(P_obs[t])
            self.history['P_obs_cont'].append(P_obs[t] + self.history['P_battery_real'][t])

            # do plots if required
            if do_plots:
                ax[0].clear()
                ax[1].clear()
                cmap = plt.get_cmap('Set1')
                line_colors = cmap(np.linspace(0, 1, 6))
                if self.battery_controller_pars['type'] in ['stochastic','dist_stoc']:
                    for s in np.arange(self.battery_controller.scen_idxs.shape[1]):
                        U_data = U[self.battery_controller.scen_idxs[:, s].ravel(), :]
                        E_n = E[self.battery_controller.scen_idxs[:, s].ravel(), :]
                        P_hat_plt = P_hat[self.battery_controller.scen_idxs[:, s].ravel()]
                        Pcontdata = P_hat_plt + U_data[:, [0]] - U_data[:, [1]]
                        ax[0].plot(Pcontdata,label='cont',color=line_colors[0,:],linewidth=0.5,alpha = 0.2)
                        ax[0].plot(P_hat_plt,label='hat',color=line_colors[1,:],linewidth=0.5,alpha = 0.2)
                        ax[1].plot(U_data[:,0],label='Pin',color=line_colors[2,:],linewidth=0.5,alpha = 0.2)
                        ax[1].plot(U_data[:,1],label='Pout',color=line_colors[3,:],linewidth=0.5,alpha = 0.2)
                        ax[1].plot(E_n,label = 'E',color=line_colors[4,:],linewidth=0.5,alpha = 0.2)
                else:
                    U_data = U
                    E_n = E
                    P_hat_plt = P_hat
                    Pcontdata = P_hat_plt + U_data[:, [0]] - U_data[:, [1]]
                    ax[0].plot(Pcontdata, label='cont')
                    ax[0].plot(P_hat_plt, label='hat')
                    ax[1].plot(U_data[:,0], label='Pin' )
                    ax[1].plot(U_data[:,1], label='Pout')
                    ax[1].plot(E_n, label='E')
                    plt.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()
        cost, peak_sh, cost_real, peak_sh_real = self.cost_KPI(P_obs)

        return self.history, cost, peak_sh, cost_real, peak_sh_real

    def solve_step(self,time,coord=False):
        '''
        Call the battery controller and get requested power. Retrieve available power and superimpose it to current
        power profile.
        :return:
        '''

        P_controlled, P_hat, U, E = self.battery_controller.solve_step(time,coord = coord)
        P_asked = np.atleast_1d(U[0, 0] - U[0, 1])*1e3      # convert in W
        #P_battery_real,SOC_real,states = self.simulate(P_asked)
        P_battery_real = np.atleast_1d(P_asked)
        SOC_real = np.atleast_1d(self.battery_controller.x_start.value[0]/self.e_nom)
        P_final = P_hat[0] + P_battery_real/1e3         # convert back in kW
        self.history['P_cont'].append(P_controlled[0][0])
        self.history['P_cont_real'].append(P_final[0])
        self.history['P_hat'].append(P_hat[0])
        self.history['P_battery'].append(P_asked[0]/1e3)
        self.history['P_battery_real'].append(P_battery_real[0]/1e3)
        self.history['SOC_real'].append(SOC_real[0])
        self.history['SOC'].append(self.battery_controller.x_start.value[0]/self.e_nom)
        #print(P_asked[0]/1e3,P_battery_real[0]/1e3)
        return P_final, P_controlled, P_hat, U, E


    def simulate(self,P_asked):
        '''
        Simulate a battery model identified in "Achieving the Dispatchability of Distribution
        Feeders through Prosumers Data Driven Forecasting and Model Predictive Control of Electrochemical
        Storage, IEEE TRANS. ON SUSTAINABLE ENERGY, TO BE PUBLISHED (2016)."  based on the SOC. The model is rescaled
        based on the nominal battery capacity.
        '''

        # find which parameters must be used, based on SOC
        #print(P_asked,self.SOC[0].value,self.SOC_min)
        ss_idx = int(np.argmin(np.abs(self.SOC[0].value-self.SOC_vect)))
        # find current setpoint, given P
        if P_asked>=0:
            self.P_asked[ss_idx].value = P_asked
            self.randn[ss_idx].value = np.random.randn(1)
            self.battery_simulators[ss_idx].solve(solver='SCS',verbose=True)
            if self.SOC[ss_idx].value <0:
                print('error')
            # reconcile all the variables
            for i in np.arange(len(self.v_old)):
                # update old voltage
                self.v_old[i].value = self.v[ss_idx].value
                self.v[i].value = self.v[ss_idx].value
                self.v_old_m[i].value = self.v[ss_idx].value
                self.v_m[i].value = self.v[ss_idx].value

                # update SOC
                self.SOC_old[i].value = self.SOC[ss_idx].value
                self.SOC[i].value = self.SOC[ss_idx].value
                self.SOC_old_m[i].value = self.SOC[ss_idx].value
                self.SOC_m[i].value = self.SOC[ss_idx].value

                # update states
                self.states_old[i].value = self.states[ss_idx].value
                self.states[i].value = self.states[ss_idx].value
                self.states_old_m[i].value = self.states[ss_idx].value
                self.states_m[i].value = self.states[ss_idx].value
            # return actual P and SOC
            P = self.P[ss_idx].value
            SOC = self.SOC[ss_idx].value
            states = self.states[ss_idx].value
        else:
            self.P_asked_m[ss_idx].value = P_asked
            self.randn_m[ss_idx].value = np.random.randn(1)
            self.battery_simulators_m[ss_idx].solve(solver='SCS',verbose=True)
            if self.SOC_m[ss_idx].value <0:
                print('error')
            # reconcile all the variables
            for i in np.arange(len(self.v_old)):
                # update old voltage
                self.v_old[i].value = self.v_m[ss_idx].value
                self.v[i].value = self.v_m[ss_idx].value
                self.v_old_m[i].value = self.v_m[ss_idx].value
                self.v_m[i].value = self.v_m[ss_idx].value

                # update SOC
                self.SOC_old[i].value = self.SOC_m[ss_idx].value
                self.SOC[i].value = self.SOC_m[ss_idx].value
                self.SOC_old_m[i].value = self.SOC_m[ss_idx].value
                self.SOC_m[i].value = self.SOC_m[ss_idx].value

                # update states
                self.states_old[i].value = self.states_m[ss_idx].value
                self.states[i].value = self.states_m[ss_idx].value
                self.states_old_m[i].value = self.states_m[ss_idx].value
                self.states_m[i].value = self.states_m[ss_idx].value

            # return actual P and SOC
            P = self.P_m[ss_idx].value
            SOC = self.SOC_m[ss_idx].value
            states = self.states_m[ss_idx].value


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
                #"R_3": [2.5e-3,4.9e-5,2.4e-4,6.8e-4,6.0e-4],
                #"C_3": [544.2,789.0,2959.7,100.2,6177.3],
                "R_3" : [2.4e-4,2.4e-4,2.4e-4,2.4e-4,2.4e-4],
                "C_3": [2959.7, 2959.7, 2959.7, 2959.7, 2959.7],
                "k_1": [0.639,0.677,0.617,0.547,0.795],
                "k_2": [ -5.31, -0.22, -0.36, -0.28,0.077],
                "k_3": [5.41,40,0.40,2.83,-0.24],
                "sigma2":  [-1.31, -0.42,0.3426,3.5784,2.7694]
                }
        return pars


    def _battery_simulator_plus(self,A,B,G,D):

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
        constraints.append(i>= 0)

        # voltage constraints
        constraints.append(v <= self.v_max)
        constraints.append(v >= self.v_min)
        # SOC constraints
        constraints.append(SOC <= self.SOC_max)
        constraints.append(SOC >= self.SOC_min)
        # dynamics - v
        constraints.append(states == A*states_old + B[:,[0]]*i)
        constraints.append(v == cvx.sum(states)/self.scaling_factor + D[0,0]*i + D[0,1] + G*randn)

        # dynamics - SOC
        constraints.append(SOC == SOC_old + i*self.dt/self.cap_nom/3600)

        # constraint on requested energy
        constraints.append(P == v_old*i )
        constraints.append(P >= P_asked)
        simulator = cvx.Problem(cvx.Maximize(i), constraints)
        #simulator = cvx.Problem(cvx.Minimize(cvx.norm1(P-P_asked)), constraints)

        return simulator,i,states,states_old,v,v_old,SOC_old,SOC,P,P_asked,randn

    def _battery_simulator_minus(self,A,B,G,D):

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
        constraints = [i <= 0]
        constraints.append(i>= self.i_min)

        # voltage constraints
        constraints.append(v <= self.v_max)
        constraints.append(v >= self.v_min)
        # SOC constraints
        constraints.append(SOC <= self.SOC_max)
        constraints.append(SOC >= self.SOC_min)
        # dynamics - v
        constraints.append(states == A*states_old + B[:,[0]]*i)
        constraints.append(v == cvx.sum(states)/self.scaling_factor + D[0,0]*i + D[0,1] + G*randn)

        # dynamics - SOC
        constraints.append(SOC == SOC_old + i*self.dt/self.cap_nom/3600)

        # constraint on requested energy
        constraints.append(P == v_old*i )
        constraints.append(P <= P_asked)
        simulator = cvx.Problem(cvx.Minimize(i), constraints)
        #simulator = cvx.Problem(cvx.Minimize(cvx.norm1(P-P_asked)), constraints)

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
        soc_model_past = ax[1].plot(np.ones(dims[0]).reshape(-1, 1), np.nan * np.ones(dims),'+')

        xdata = np.arange(dims[0]).reshape(-1, 1)
        cmap = plt.get_cmap('Set1')
        line_colors = cmap(np.linspace(0, 1, 6))
        x_past = np.arange(1 - dims[0], 1)
        past_data_1 = np.nan * np.ones(dims[0])
        past_data_2 = np.nan * np.ones(dims[0])
        past_data_3 = np.nan * np.ones(dims[0])
        past_data_4 = np.nan * np.ones(dims[0])
        past_data_5 = np.nan * np.ones(dims[0])

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
            past_data_3[dims[0] - np.minimum(len(np.hstack(self.history['P_hat'])), dims[0]):] = np.hstack(
                self.history['P_hat'])[-dims[0]:]
            past_data_4[dims[0] - np.minimum(len(np.hstack(self.history['SOC_real'])), dims[0]):] = np.hstack(
                self.history['SOC_real'])[-dims[0]:]*self.e_nom
            past_data_5[dims[0] - np.minimum(len(np.hstack(self.history['SOC'])), dims[0]):] = np.hstack(
                self.history['SOC'])[-dims[0]:]*self.e_nom

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

            soc_model_past[0].set_data(x_past, past_data_5)
            soc_model_past[0].set_color(line_colors[5, :])

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

    def cost_KPI(self,P_obs):
        p_s = self.battery_controller_pars['ps']
        p_b = self.battery_controller_pars['pb']
        Pm = P_obs + np.array(self.history['P_battery']).reshape(-1,1)
        Pm_real = P_obs + np.array(self.history['P_battery_real']).reshape(-1,1)
        cost = self.dt[0]*Pm*((Pm>=0)*p_b + (Pm<0)*p_s)
        cost_real = self.dt[0]*Pm_real*((Pm_real>=0)*p_b + (Pm_real<0)*p_s)
        peak_sh = 0.5*Pm**2
        peak_sh_real = 0.5*Pm_real**2

        return cost,peak_sh,cost_real,peak_sh_real
