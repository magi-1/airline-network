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

class meinhausen_bulman:
    def __init__(self, delay_matr, carrier, lambd):
        self.delay_matr = delay_matr
        self.carrier = carrier
        self.lambd = lambd
        self.nodes = list(delay_matr.columns)
        self.params = self.estimate_parmeters()
        self.edges = self.stitch_edges()
        
    def estimate_parmeters(self):
        params = {}
        for node_i in self.nodes:
            data_i = self.delay_matr[node_i]
            data_rest = self.delay_matr[[i for i in self.nodes if i != node_i]]
            lassoreg_i = linear_model.Lasso(self.lambd)
            lassoreg_i.fit(data_rest, data_i)
            params[node_i] = lassoreg_i.set_params()
        return params
            
    def stitch_edges(self):
        delta = 0.01
        edges = pd.DataFrame()
        for node_i, model_i in self.params.items():
            ind = list(self.params.keys()).index(node_i)
            coeffs = [i if abs(i) >= delta else 0 for i in model_i.coef_]
            coeffs.insert(ind, 1)
            edges[node_i] = coeffs

        for node_i in self.nodes:
            for node_j in range(len(self.nodes)):
                edge_pair = [edges[node_i].iloc[node_j], 
                             edges[self.nodes[node_j]].iloc[self.nodes.index(node_i)]]
                max_edge = max(edge_pair)
                edges[node_i].iloc[node_j] = max_edge
                edges[self.nodes[node_j]].iloc[self.nodes.index(node_i)] = max_edge
        return edges
    
    def view_graph(self, g_type):
        N = self.nodes
        
        G=nx.Graph()
        G.add_nodes_from(N)
        routes = MY_DATA[['airport', 'lat', 'lon']].drop_duplicates()
        plotly_rows = []
        for n_i in N:
            for n_j in N:
                if n_i != n_j:
                    weight = self.edges[n_i].iloc[N.index(n_j)]
                    if weight != 0:
                        edge = (n_i, n_j)
                        G.add_edge(n_i, n_j, weight = weight)

                    org = routes[routes['airport'] == n_i]
                    dest = routes[routes['airport'] == n_j]
                    plotly_rows.append({
                        'weight': weight,
                        'start_lon': org['lon'].values[0],
                        'end_lon': dest['lon'].values[0],
                        'start_lat': org['lat'].values[0],
                        'end_lat': dest['lat'].values[0],
                        'start_airport': n_i,
                        'end_airport': n_j
                    })
        plotly_df = pd.DataFrame(plotly_rows).drop_duplicates()
        
        if g_type == 'circular':
            plt.figure(figsize = (2.5,2.5))
            
            nx.draw_circular(G, node_size = 125, 
                             node_color = [CLRS[CODES.index(self.carrier)]], 
                             with_labels=False)
            plt.show(G)
            
        elif g_type == 'geographic':
            sub = plotly_df[['start_airport','start_lat','start_lon']].drop_duplicates()
            inds = plotly_df[[i for i in plotly_df.columns if 'end' not in i]].drop_duplicates().index
            plotly_df = plotly_df.loc[inds]

            fig = go.Figure()
            for ap in sub['start_airport'].unique():
                ap_i = sub[sub['start_airport'] == ap]
                fig.add_trace(
                    go.Scattergeo(
                        locationmode = 'USA-states',
                        lon = ap_i['start_lon'], 
                        lat = ap_i['start_lat'],
                        hoverinfo = 'text',
                        name = ap,
                        mode = 'markers',
                        marker = dict(
                            size = 2,
                            color = 'rgb(0,0,0)',
                            line = dict(width = 3,color = 'rgb(0,0,0)'))
                    )
                )
            line_colors = ['red','green']
            def sign_color(weight):
                if weight >= 0:
                    return 'green'
                else:
                    return 'red'

            flight_paths = []
            for i in range(len(plotly_df)):
                fig.add_trace(
                    go.Scattergeo(
                        locationmode = 'USA-states',
                        lon = [plotly_df['start_lon'].iloc[i], plotly_df['end_lon'].iloc[i]],
                        lat = [plotly_df['start_lat'].iloc[i], plotly_df['end_lat'].iloc[i]],
                        mode = 'lines',
                        name = plotly_df['end_airport'].iloc[i],
                        opacity = 1,
                        line = dict(
                            width = abs(plotly_df['weight'].iloc[i])*5, 
                            color = sign_color(plotly_df['weight'].iloc[i])) 
                    )
                )
            fig.update_layout(
                showlegend = False,
                geo = dict(
                    scope = 'north america',
                    projection_type = 'azimuthal equal area',
                    showland = True,
                    landcolor = 'rgb(230,230,230)',
                    countrycolor = 'rgb(204, 204, 204)'),
                width = 960, height = 700)
            fig.show()

class fuse_lasso:

    DELTA = 0.01
    def __init__(self, flight_data, M, lambd_grid, alpha_grid):
        self.flight_data = flight_data
        self.M = M
        self.nodes = list(M.columns)
        self.T = M.shape[0]
        self.d = M.shape[1]
        self.diff_matr = self.make_diff_matr()
        self.lambd_grid = lambd_grid
        self.alpha_grid = alpha_grid
        self.edge_list = self.edge_estimation()
        self.graphs = self.precision_matricies()
        
    def make_diff_matr(self):
        D = (np.identity(self.T) + np.diag([-1]*(self.T-1),k=1))[:-1]
        return D
    
    def MSE(self, X, y, beta):
        return (.5)*sum([(y[t] - sum([beta[t][i]*X[t][i] for i in range(self.d-1)]))**2 for t in range(self.T)])

    def l1_norm(self, beta):
        return sum([cp.norm1(beta[t]) for t in range(self.T)])

    def fusion(self, beta):
        return sum([cp.norm1(vec) for vec in [self.diff_matr@beta.T[i] for i in range(self.d-1)]])

    def obj_func(self, X, y, beta_matr, _lamda_, _alpha_):
        return self.MSE(X, y, beta_matr) + _lamda_*self.l1_norm(beta_matr) + _alpha_*self.fusion(beta_matr)
    
    def edge_estimation(self):
        estimates = []
        for node_i in self.nodes:
            y = self.M[node_i].to_numpy()
            X = self.M[[j for j in self.nodes if j != node_i]].to_numpy()
            
            # node_i optimization 
            _lambd_ = cp.Parameter(nonneg=True)
            _alpha_ = cp.Parameter(nonneg=True)
            beta_matr = cp.Variable(shape = (self.T, self.d-1))
            optim_problem = cp.Problem(
                cp.Minimize(self.obj_func(X, y, beta_matr, _lambd_, _alpha_))
            )
            
            for l in self.lambd_grid:
                _lambd_.value = l
                for a in self.alpha_grid:
                    _alpha_.value = a
                    optim_problem.solve(solver='ECOS')
                    estimates.append({'node': node_i,
                                     'l': l, 'a': a,  
                                     'error': optim_problem.value, 
                                     'beta_matrix': beta_matr.value})
        return pd.DataFrame(estimates)
    
    def precision_matricies(self):
        omega_list = []
        for t in range(self.T):

            # creaing precision_matrix(t)  
            edges_t = pd.DataFrame()
            for i in range(len(self.edge_list['node'])):
                node_i = self.nodes[i]
                beta_t_i = list(
                    self.edge_list[self.edge_list['node'] == node_i]['beta_matrix'].iloc[0][t]
                )
                beta_t_i.insert(i,1)
                edges_t[node_i] = beta_t_i

            # stitching the edges for each precision_matrix(t)  
            for node_i in self.nodes:
                for node_j in self.nodes:
                    index_i = self.nodes.index(node_i)
                    index_j = self.nodes.index(node_j)
                    max_edge = max([edges_t[node_i].iloc[index_j], 
                                    edges_t[node_j].iloc[index_i]])
                    if max_edge < 0.01:
                        max_edge = 0
                    edges_t[node_i].iloc[index_j] = max_edge
                    edges_t[node_j].iloc[index_i] = max_edge
            omega_list.append(edges_t)
        return omega_list
    
    def plot_paramaters(self, demo = False):
        dates = self.flight_data['datetime'].unique()
        
        num_nodes = len(self.nodes)
        if demo == True:
            num_nodes = 3
        for i in range(num_nodes):
            params = []
            for omega_t in self.graphs:
                indicies = [x for x in range(len(self.nodes)) if x != i]
                params.append(omega_t[self.nodes[i]].iloc[indicies])
            plt.figure(figsize = (13,4))
            plt.plot(dates, params)
            plt.title(self.nodes[i])
            plt.show()
            
    def generate_segments(self, threshold, plot = True):
        omega = self.graphs
        dates = self.flight_data['datetime'].unique()
        omega_delta = [omega[i+1].to_numpy()-omega[i].to_numpy() for i in range(len(omega)-1)]
        
        rows = []
        for i in range(len(omega_delta)):
            Sum = 0
            omega_dt = omega_delta[i]
            for j in range(len(omega_dt)):
                Sum += omega_dt[j].max()
            rows.append({'dates': dates[i], 'value': Sum})
        df = pd.DataFrame(rows)

        if plot == True:
            plt.figure(figsize = (14, 6))
            plt.plot(df['dates'], df['value'], color = 'k')
            plt.title('lambda = {}'.format(self.alpha_grid[0]))
            #plt.hlines(y = threshold, 
            #           xmin = dates[0], xmax= dates[-1], 
             #          color = 'red',
              #         linestyles  = 'dashed',
              #         linewidth = 4)
            plt.show()


        good = df[(df['value'] >= threshold)]
        dates = good['dates']
        bad_dates = []
        for i in range(len(dates)-1):
            if (dates.iloc[i+1]-dates.iloc[i] < dt.timedelta(days = 30)):
                bad_dates.append(dates.iloc[i+1])    

        return good[~(good['dates']).isin(bad_dates)]