import numpy as np
import pandas as pd
import scipy.stats as si
from scipy import optimize
import QuantLib as ql

class Simulator:
    def __init__(self, process, periods_in_day=1, bid_ask_spread=0.03, rf_df=None):
        self.process = process
        self.D = periods_in_day
        self.bid_ask_spread = bid_ask_spread
        self.rf_df = rf_df  # DataFrame containing risk-free rates

    def set_properties_gbm(self, q, mu):
        self.q = q
        self.mu = mu

    def set_properties_heston(self, v0, kappa, theta, sigma, rho, q, r):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.q = q
        self.r = r

    def simulate(self, S0, T=252, dt=1/252):
        if self.process == 'GBM':
            self._sim_gbm(S0, self.mu, T, dt)
        else:
            self._sim_heston(S0, self.v0, self.kappa, self.theta, self.sigma, self.rho, self.q, self.r, T, dt)

    def _sim_gbm(self, S0, mu, T, dt):
        self.St = np.zeros(int(T/dt))
        self.St[0] = S0

        for t in range(1, len(self.St)):
            self.St[t] = self.St[t-1] * np.exp(mu * dt + np.sqrt(dt) * np.random.normal())
