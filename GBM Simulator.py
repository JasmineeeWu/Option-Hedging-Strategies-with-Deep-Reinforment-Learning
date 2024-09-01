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
    def return_set(self, strike_min, strike_max, quote_datetime, min_exp, max_exp, datearray):
        num_options = 10
        strike_prices = np.random.uniform(strike_min, strike_max, num_options)
        expirations = np.random.randint(min_exp, max_exp, num_options)

        datearray_list = datearray.tolist()
        quote_index = datearray_list.index(quote_datetime)

        expiration_dates = [
            datearray_list[min(quote_index + exp, len(datearray_list) - 1)]
            for exp in expirations
        ]

        option_types = np.random.choice(['call', 'put'], num_options)

        quote_datetimes = [
            datearray_list[min(quote_index + int(expirations[i]), len(datearray_list) - 1)]
            for i in range(num_options)
            for _ in self.St
        ]

        St = self.St / self.St[0]
        Ts = np.array([
            exp - np.arange(0, len(self.St) / (1 / self.D), 1 / self.D)
            for exp in expirations
        ])

        df = pd.DataFrame({
            'underlying_price': np.tile(St, num_options),
            'expiration': np.repeat(expiration_dates, len(St)),
            'strike': np.repeat(strike_prices, len(St)),
            'quote_datetime': quote_datetimes[:len(self.St) * num_options],
            'ticker': 'simulated',
            'option_type': np.repeat(option_types, len(St))
        })

        prices = []
        bids = []
        asks = []

        min_price = 0.01

        for i in range(num_options):
            for t in range(len(self.St)):
                expiration_date = pd.to_datetime(df['expiration'].iloc[t])
                rf_rate_row = self.rf_df.loc[self.rf_df['Date'] == expiration_date]
                if not rf_rate_row.empty:
                    rf_rate = rf_rate_row['RF'].values[0] / 100
                else:
                    rf_rate = self.rf_df['RF'].iloc[-1] / 100

                price = call_price(St[t], strike_prices[i], rf_rate, self.q, 0.2, Ts[i][t]/252) if option_types[i] == 'call' else put_price(St[t], strike_prices[i], rf_rate, self.q, 0.2, Ts[i][t]/252)

                price = max(price, min_price)
                bid = price * (1 - self.bid_ask_spread / 2)
                ask = price * (1 + self.bid_ask_spread / 2)

                bid = max(bid, min_price)
                ask = max(ask, min_price)

                prices.append(price)
                bids.append(bid)
                asks.append(ask)

        df['bid'] = pd.Series(bids).reindex(df.index).fillna(min_price).values
        df['ask'] = pd.Series(asks).reindex(df.index).fillna(min_price).values

        return df
