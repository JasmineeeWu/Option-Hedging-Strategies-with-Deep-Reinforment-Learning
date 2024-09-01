import numpy as np
import pandas as pd
import scipy.stats as si
from scipy import optimize
import QuantLib as ql
from scipy.stats import norm

def log(x):
    return np.log(x)

def exp(x):
    return np.exp(x)

def calc_impl_volatility(S, K, r, q, T, P):
    P_adj = P
    def price_comp(sigma):
        return P_adj - call_price(S, K, r, q, sigma, T)

    v = None
    t = 0
    s = -1
    while v is None and t < 20:
        P_adj = P + t * s * 0.0001
        try:
            v = optimize.brentq(price_comp, 0.001, 100, maxiter=1000)
        except:
            v = None
            if s > 0:
                t += 1
            s *= -1

    return v

def _d(S, K, r, q, v, T):
    d1 = (log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    return d1, d2

def _N(d1, d2):
    return si.norm.cdf(d1), si.norm.cdf(d2)

def put_price(S, K, r, q, v, T):
    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)

    return -S * exp(-q * T) * (1 - N1) + K * exp(-r * T) * (1 - N2)

def call_price(S, K, r, q, v, T):
    if T <= 0.0:
        return max(S-K, 0)

    d1, d2 = _d(S, K, r, q, v, T)
    N1, N2 = _N(d1, d2)

    price = S * exp(-q * T) * N1 - K * exp(-r * T) * N2

    return price

def calculate_delta(S, K, r, q, v, T, option_type='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * v**2) * T) / (v * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
