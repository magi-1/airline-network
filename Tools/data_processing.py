
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

from AirLib import data_accessories

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

class clean_raw_data:
    
    # reading from data_accessories
    ap_info = data_accessories.ap_info 
    
    def __init__(self, airports, routes, fares, delays):
        
        # reading/cleaning data
        self.airports = airports
        self.routes = routes
        self.fares = fares
        self.delays = self.process_delays(delays)
        self.make_airport_coords()
        
    def make_airport_coords(self):
        
        ''' 
        Merging additional information with given airport 
        info. Adding airport name as well as lat/lon data.
        '''
        
        airport_name = [] ; lat = [] ; lon = []
        x = [] ; y = [] ; z = []
        for AP in self.airports['Airport'].unique():
            
            ap_i = self.ap_info[self.ap_info['AIRPORT'] == AP].iloc[-1] 
            airport_name.append(ap_i['DISPLAY_AIRPORT_NAME'])
            
            # gettinbg airport latitude and longitude
            lon_i = ap_i['LONGITUDE']
            lat_i = ap_i['LATITUDE']
            
            
            # converts to geocentric coordinates meters
            geo_centr = EarthLocation(lon_i, lat_i).value 
            
            lat.append(lat_i)
            lon.append(lon_i) 
            x.append(geo_centr[0])
            y.append(geo_centr[1])
            z.append(geo_centr[2])
           
        # lat / long data
        self.airports['airport_name'] = airport_name
        self.airports['lon'] = lon
        self.airports['lat'] = lat
        
        # geocentric cartesian coordinates (meters)
        self.airports['x'] = x
        self.airports['y'] = y
        self.airports['z'] = z
        
    def process_delays(self, data):
        # getting rid of nan
        self.delays = data.dropna(subset=['ARR_DELAY','DEP_DELAY'])
        data['datetime'] = pd.to_datetime(data['FL_DATE'])
        data = data[data['CANCELED'] != 1]
        return data.fillna(0)

class prep_delay_data:

    def __init__(self, data, carrier, N_airports):
        self.carrier = carrier
        self.N_airports = N_airports
        self.top_airports = self.find_top_aps(data)
        self.delay_matr = self.clean_and_format(data)  
        
    def summarize(self):
        pass
    
    def find_top_aps(self, data):
        top_airports = {}
        for i in range(len(CODES)):
            carr = CODES[i]
            sub = AIRLINE_DATA.delays[AIRLINE_DATA.delays['CARRIER'] == carr]
            ap_counts = sub['ORIGIN'].value_counts()
            top_airports[carr] = ap_counts.index[:self.N_airports]
        return list(top_airports[self.carrier])
    
    def clean_and_format(self, data):
        data['datetime'] = pd.to_datetime(data['datetime'])
        subset_0 = data[((data['airport']).isin(self.top_airports)) & 
                        (data['carrier'] == self.carrier)]
        
        '''~ dropping bad dates ~'''
        # if a given airport has 0 flights on some, 
        # remove all flights on that day
        bad_index = []
        for ap in subset_0['airport'].unique():
            sub = subset_0[subset_0['airport'] == ap]['avg_dep']
            nan_rows = pd.isna(sub)
            nan_inds = nan_rows[nan_rows == True].index
            bad_index.extend(list(nan_inds))

        bad_dates = subset_0['datetime'].loc[bad_index]
        inds_to_drop = subset_0[subset_0['datetime'].isin(bad_dates)].index
        subset_1 = subset_0.drop(inds_to_drop)
        #print('Number of rows dropped:',len(subset_0)-len(subset_1))
        
        '''~ log transforming ~ & ~ making delay matrix ~'''
        trans_rows = []
        delay_matr = pd.DataFrame()
        diff_matr = pd.DataFrame()
        matr = []
        for ap in self.top_airports:
            sub = subset_1[subset_1['airport'] == ap]
            # airport wise transformation, each airport X_i ~ N(0,sigma**2)
            delays = sub['avg_dep'].values
            shift = abs(delays.min())+1
            delays += shift
            delays = np.log(delays)
            mu_log = np.mean(delays)
            delays -= mu_log
            
            # making matr object (cleaned and transformed version of original input)
            # note that delay_matr is just a table of delays with no info about time etc
            sub['avg_dep'] = np.log(sub['avg_dep']+shift)-mu_log
            
            # making matrs
            delay_matr[ap] = delays
            diff_matr[ap] = pd.Series(delays).diff(1)[1:]
            # appending data
            matr.append(sub)
            trans_rows.append({'airport': ap,
                               'shift': shift,
                               'mu_log': mu_log})
                                  
        self.trans_df = pd.DataFrame(trans_rows)
        self.matr = pd.concat(matr)
        self.matr['raw_avg_dep'] = subset_1['avg_dep']
        self.diff_matr = diff_matr
        '''~ sorting based on distances ~'''
        unsorted_aps = delay_matr.columns 
        rltv_dists = []
        rfrnce_ap = unsorted_aps[0]
        for ap_i in unsorted_aps:
            if ap_i != rfrnce_ap:
                # trying to see if there is an actual flight, if not, set 1e6 distance
                try:
                    subset = AIRLINE_DATA.routes[
                        (AIRLINE_DATA.routes['ORIGIN'] == rfrnce_ap) & 
                        (AIRLINE_DATA.routes['DEST'] == ap_i)
                        ]
                    d = subset['DISTANCE'].values[0]
                    rltv_dists.append({'airport': ap_i, 'distance': d})
                except:
                    rltv_dists.append({'airport': ap_i, 'distance': 1e6})
            else:
                rltv_dists.append({'airport': rfrnce_ap, 'distance': 0})

        dist_df = pd.DataFrame(rltv_dists).sort_values(by = ['distance'])
        delay_matr = delay_matr[dist_df['airport']]
        
        return delay_matr