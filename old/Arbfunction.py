import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import seaborn as sns


from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import norm
from pykalman import KalmanFilter
from prettytable import PrettyTable
from itertools import combinations

class Datafetcher:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = self.load_tickers()

        @staticmethod
        def load_tickers():
            with open('tickers.json', 'r') as f:
                tickers_json = json.load(f)
            return tickers_json["tickers"]
        
        def get_data(self, column='Close'):
            data = yf.download(tickers=self.tickers, start=self.start_date, end=self.end_date, progress=False)
            return data[column].dropna(how='any')
            return data
        
        @staticmethod
        def split_data(price_data, volume_data, split_ratio = 0.70):
            # Split the data into training and testing sets
            total_rows = len(price_data)
            train_rows = int(split_ratio * total_rows)
            train_data = price_data.iloc[:train_rows]
            test_data = price_data.iloc[train_rows:]
            volume_train_data = volume_data.iloc[:train_rows]
            volume_test_data = volume_data.iloc[train_rows:]
            return train_data, test_data, volume_train_data, volume_test_data
        

class Cointegration:
    @staticmethod
    def cointegration_test(data):
        #Perform a cointegration test on all pairs of columns in the given DataFrame
        n = data.shape[1]
        pvalues_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i, j in combinations(range(n), 2):
            result = coint(data[keys[i]], data[keys[j]])
            pvalues_matrix[i, j] = result[1]
            if result[1] < 0.05:
                pairs.append((keys[i], keys[j]))
            return pvalues_matrix, pairs
        

class PairProcessor:
    @staticmethod
    def __init__(self, lookback_period = 200, step_size = 20, periods_per_year = 252):
        self.lookback_period = lookback_period
        self.step_size = step_size
        self.periods_per_year = periods_per_year
        self.data = None
        self.volumne_data = None
        self.pvalues_df = None  
        self.pairs = None
        self.tbill = None
        self.portfolio_data = {}
        self.sharpe_values_dict = {}

    def set_data(self, data, volume_data, pvalues_df, pairs, tbill):
        self.data = data
        self.volume_data = volume_data
        self.pvalues_df = pd.DataFrame(pvalues_df, index = data.column, columns=data.columns)
        self.pairs = pairs
        self.tbill = tbill

    def process_all_pairs(self):
        #Process all pairs based on the provided pairs and p values
        stacked_df = self.pvalues_df.stack()
        stacked_df.index.names = ['asset1', 'asset2']
        stacked_df = stacked_df.reset_index()
        stacked_df.columns = ['asset1', 'asset2', 'P-value']
        stacked_df = stacked_df.dropna(subset=['P-value'])
        stacked_df = stacked_df.reset_index(drop=True)
        sequential_results = [
            result for asset1, asset2 in self.pairs
            for result in self.process_pair(asset1, asset2,self.data)
        ]
        results_df = pd.DataFrame(sequential_results, columns=['asset1', 'asset2', 'start', 'end', 'beta (kalman)', 'half-life'])
        simulator = PairsTradingSimulator()
        results, portfolio_data = simulator.simulate_pairs_trading(
            data = self.data, liquidity_data = self.volume_data, Cointegration_results = results_df
        )
        self.portfolio_data = portfolio_data
        unique_results = results.drop_duplicates(subset=['asset1', 'asset2'])[['asset1', 'asset2', 'terminal wealth']]
        sorted_unique_results = unique_results.sort_values(by='Terminal Wealth', ascending=False)
        return sorted_unique_results

    def process_pair(asset1, asset2, data, lookback_period=200, step_size=20):
        #Analyze and process a pair of assets for cointegration and calculates the Kalman-filtered beta values and half-life of mean reversion
        num_windows = (len(data) - lookback_period) // step_size + 1
        beta_kalman_values = np.empty(num_windows)
        half_life_values = np.empty(num_windows)
        x = np.array([data[asset1].iloc[start:start + lookback_period].values
                      for start in range(0, len(data) - lookback_period + 1, step_size)])
        y = np.array([data[asset2].iloc[start:start + lookback_period].values
                        for start in range(0, len(data) - lookback_period + 1, step_size)])
        delta = 1e-3
        trans_cov = delta / (1 - delta) * np.eye(2)
        for idx, (x, y) in enumerate(zip(x, y)):
            obs_mat = np.expand_dims(np.vstack([x], [np.ones(len(x))]).T, axis=1)
            kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, 
                                initial_state_mean=np.zeros(2),
                                initial_state_covariance=np.ones((2, 2)),
                                transition_matrices=np.eye(2),
                                observation_matrices=obs_mat,
                                observation_covariance=1.0,
                                transition_covariance = trans_cov)
            state_means, _ = kf.filter(y[:, np.newaxis]) 
            beta_kalman_values[idx] = state_means[-1, 0]
            spread = y - beta_kalman_values[idx] * x
            lagged_spread = spread[:-1]
            delta_spread = spread[1:] - spread[:-1]
            spread_model = sm.OLS(delta_spread, lagged_spread).fit()
            half_life_values[idx] = -np.log(2) / spread_model.params[0]
                                                                             


                                