import numpy as np
import matplotlib.pyplot as plt
from forecasters import RELM

class Aggregator:
    def __init__(self, agents,rho,GHI_f,T_f,mode):
        self.agents = agents    # agents list,
        self.rho = rho      # ADMM constant
        self.GHI_f = GHI_f # forecast matrix for GHI, (n_obs,n_sa,n_scens)
        self.T_f = T_f     # forecast matrix for T,   (n_obs,n_sa,n_scens)
        self.mode = mode  # choose between 'deterministic' and 'stochastic' modes

    def train(self):
        '''
        Train :
            1) Voltage forecaster at the PCC, given (GHI,T)
            2) Voltage map coefficients
            3) Power forecasters for each building
            4) Meteo forecasters
        :return:
        '''


    def solve_step(self,t):
        # retrieve voltage forecasts
        0