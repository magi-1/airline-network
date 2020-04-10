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

class visual:
    '''
    Note that this class takes multiple different kinds of
    data. It will not execute if you call a plotting method
    that doesnt math the data type (dataframe with correct columns)
    '''
    def __init__(self, data, carrier):
        self.data = data
        self.carrier = carrier
        self.index = CODES.index(carrier)
        
    def flight_frequency(self):
        dates = self.data['FL_DATE'].unique()
        counts = self.data['ORIGIN'].value_counts()/len(dates)
        aps = counts.index

        fig = go.Figure(data=[go.Bar(
                x=aps, y=counts,
                marker_color=['rgb{}'.format(CLRS[self.index]) for i in range(len(aps))])
                         ])
        fig.update_layout(
            title="{} - Flights Per Day ({}, {})".format(NAMES[self.index], dates[0], dates[-1]),
            plot_bgcolor="white", 
            width = 950, height = 500)
        
        color = 'rgb(209, 209, 209)'
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color)
        fig.show()
        
    def time_series(self):
        fig = go.Figure()
        for c in self.data['airport'].unique():
            sub = self.data[self.data['airport'] == c]
            fig.add_trace(go.Scatter(
                            x = sub['datetime'],
                            y = sub['avg_dep'],
                            name = c,
                            opacity=.9))

        fig.update_layout(title = '{} - Mean Delay'.format(NAMES[self.index]),
                          xaxis_range=[
                              self.data['datetime'].min(),
                              self.data['datetime'].max()
                          ],
                         xaxis_rangeslider_visible=True,
                         width = 950, height = 600, 
                         plot_bgcolor = "white")
        color = 'rgb(209, 209, 209)'
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color)
        fig.show()
        
    def cov_matr(self):
        # airpots, covariance & precision matrix
        aps = self.data.columns
        cov_matr = np.corrcoef(self.data.T)
        prec_matr = linalg.inv(cov_matr)
        
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (18,7))
        sns.heatmap(cov_matr, ax = ax0, center=0, 
                    xticklabels = aps, yticklabels = aps)
        sns.heatmap(prec_matr, ax = ax1, center=0, 
                    xticklabels = aps, yticklabels = aps)
        plt.show()
        
    def dist_corr(self):
        aps = list(self.data.columns)
        cov_matr = np.corrcoef(self.data.T)
        prec_matr = linalg.inv(cov_matr)
        pairs = []
        for i in aps:
            for j in aps:
                if i!=j and (j, i) not in pairs:
                    pairs.append((i,j))

        values = []
        for p in pairs:
            subset = AIRLINE_DATA.routes[
                (AIRLINE_DATA.routes['ORIGIN'] == p[0]) & 
                (AIRLINE_DATA.routes['DEST'] == p[1])]
            d = subset['DISTANCE'].values
            
            if len(d) > 0:
                values.append({'distance': d[0],
                             'corr': cov_matr[aps.index(p[0])][aps.index(p[1])],
                             'origin': p[0],
                             'dest': p[1]})

        dist_corr = pd.DataFrame(values)
        plt.figure(figsize = (10,6))
        plt.grid(b=True, which='major', color='grey', linewidth=0.3)
        plt.grid(b=True, which='minor', color='grey', linewidth=0.3)
        plt.scatter(dist_corr['distance'], dist_corr['corr'], s = 40, color = 'k')
        plt.xlabel('Distance', fontsize = 17)
        plt.ylabel('Correlation', fontsize = 17)
        plt.show()