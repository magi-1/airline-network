import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from itertools import combinations
import os
import numpy as np
import statsmodels.api as sm
from numpy import linalg
import pandas as pd
import datetime as dt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import random
import plotly.graph_objects as go
from IPython.display import HTML
import plotly.graph_objects as go
import networkx as nx
import plotly.express as px

global CODES, NAMES, CLRS
CODES = ['WN','DL','AA','OO','UA','YX',
         'B6','MQ','OH','AS','9E','YV',
         'NK','EV','F9','G4','HA']
NAMES = ['Southwest', 'Delta', 'American', 
         'SkyWest', 'United', 'Midwest', 
         'Jet Blue', 'Am. Eagle', 'PSA', 
         'Alaska', 'Endeavor','Mesa', 'Spirit', 
         'ExpressJet', 'Frontier', 'Allegiant', 
         'Hawaiian']
CLRS = sns.color_palette("tab10", len(CODES))

global CODES, NAMES, CLRS, AIRLINE_DATA
CODES = ['WN','DL','AA','OO','UA','YX',
         'B6','MQ','OH','AS','9E','YV',
         'NK','EV','F9','G4','HA']
NAMES = ['Southwest', 'Delta', 'American', 
         'SkyWest', 'United', 'Midwest', 
         'Jet Blue', 'Am. Eagle', 'PSA', 
         'Alaska', 'Endeavor','Mesa', 'Spirit', 
         'ExpressJet', 'Frontier', 'Allegiant', 
         'Hawaiian']

class VAR_model:
    def __init__(self, train, test, segments, carrier, lag = 1):
        self.trans_train = train.trans_df
        self.train_info = train.matr
        self.train_delays = train.delay_matr
        self.airports = train.delay_matr.columns
        
        self.trans_test = test.trans_df
        self.test_info = test.matr
        self.test_delays = test.delay_matr
        self.segments = segments
        
        self.carrier = carrier
        self.lag = lag
        
        
        self.train_diff = train.diff_matr
        self.test_diff = test.diff_matr
        
        self.predicted = self.generate_predictions()
        
    def estimate_VAR(self, data):
        return VAR(data).fit(self.lag)
    
    def transform(self, data, DIFF, Type, airport):
        
        if Type == 'train':
            
            df = self.trans_train
            params = df[df['airport'] == airport]
            new_data = np.exp(data+params['mu_log'].iloc[0])+params['shift'].iloc[0]

        elif Type == 'test':
            data = data[1:]
            DIFF = self.test_diff[airport]
            df = self.trans_test
            params = df[df['airport'] == airport]
            new_data = np.exp(DIFF+data+params['mu_log'].iloc[0])+params['shift'].iloc[0]

            #new_data = np.exp(DIFF+data+params['mu_log'].iloc[0])+params['shift'].iloc[0]
        return new_data
    
    def generate_predictions(self):
        
        X = self.train_delays
        X.index = self.train_info.datetime.unique()
        Y = self.test_delays
        Y.index = self.test_info.datetime.unique()
        time_delta = dt.timedelta(days = 365)
        
        diff_X = self.train_diff
        diff_X.index = self.train_info.datetime.unique()[1:]
        diff_Y = self.test_diff
        diff_Y.index = self.test_info.datetime.unique()[1:]
        
        rolling_preds = [] ; rolling_dates = []
        rolling_granger = []
        for i in range(len(self.segments)-1):
            dates = []
            t0 = self.segments[i]
            t1 = self.segments[i+1]            
            sub_train = X[(X.index >= t0) & (X.index < t1)]
            sub_test = Y[(Y.index >= t0+time_delta) & (Y.index < t1+time_delta)]

            if len(sub_test) >= 2:
                predictions = pd.DataFrame()
                model_t = VAR(sub_train)
                results = model_t.fit(self.lag)
                for i in range(len(sub_test)-self.lag-1):
                    obs_i = sub_test.iloc[i:i+self.lag].to_numpy()
                    predictions[sub_test.index[i+self.lag]] = results.forecast(obs_i, 1)[0]
                    dates.extend(sub_test.index[i:i+self.lag])
                    
                row = []
                for ap in self.airports:
                    other_aps = [i for i in self.airports if i != ap]
                    if aps[0] != aps[1]:
                        GC_test = results.test_causality(caused = ap, causing = other_aps).summary()
                        pvalue = GC_test.data[1::2][0][-2]
                        row.append({'caused': ap, 'pval': pvalue})
                GC_df = pd.DataFrame(row).sort_values(by = ['pval'])                
                predictions.index = sub_test.columns
                predictions = predictions.T
                rolling_dates.extend(dates)
                rolling_preds.append(predictions)
                rolling_granger.append(GC_df)
        self.granger = rolling_granger
        predictions = pd.concat(rolling_preds)
        t0_new = rolling_dates[0]
        t1_new = rolling_dates[-1]
        self.observed = Y[(Y.index >= t0_new) & (Y.index < t1_new)]
        
        diff_X = diff_X[(diff_X.index > t0_new-time_delta)  & (diff_X.index <= t1_new-time_delta)]
        diff_Y = diff_Y[(diff_Y.index > t0_new)  & (diff_Y.index <= t1_new)]
        #print(self.observed.index[:3])
        #print(diff_X.index[:3])
        #print(predictions.index[:3])
        #print(diff_Y.index[:3])
        for ap in self.airports:
            self.observed[ap] = self.transform(self.observed[ap], None,  'train', ap)
            predictions[ap] = self.transform(predictions[ap], diff_X[ap], 'test', ap)
        predictions.index += dt.timedelta(days = 1)
        
        return predictions.iloc[1:]
    
    def plot_results(self):
        dates0 = self.observed.index
        dates1 = self.predicted.index
        for ap in self.airports:
            print(ap)
            o = self.observed[ap]
            p = self.predicted[ap]
            plt.figure(figsize = (15,7))
            plt.plot(dates0, o, c = 'k')
            plt.plot(dates1, p, c = 'r')
            plt.savefig(f'pred+{ap}.png')
            plt.show()