# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:26:29 2020

Contains functions to read data in csv files and print it with AMPL syntax in ESTD_data.dat
Also contains functions to analyse input datas

@author: Paolo Thiran
"""

import numpy as np
import pandas as pd
import csv


### Usefull functions for printing in AMPL syntax###
def ampl_syntax(df,comment) :
    # adds ampl syntax to df
    df2=df.copy()
    df2.rename(columns = {df2.columns[df2.shape[1]-1] : str(df2.columns[df2.shape[1]-1])+' '+':= '+comment}, inplace=True)
    return df2

def print_set(my_set,name, out_path) :
    with open(out_path, mode='a',newline='') as TD_file:
        writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['set ' + name +' := \t' + '\t'.join(my_set) + ';'])
        
def print_df(name, df, out_path) : # should add the comment for the param
    df.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=name, quoting=csv.QUOTE_NONE)

    with open(out_path, mode='a',newline='') as TD_file:
        writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([';'])
    
def newline(out_path):
    with open(out_path, mode='a',newline='') as TD_file:
        writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([''])
            
def print_param(name,param,comment, out_path):
    
    with open(out_path, mode='a',newline='') as TD_file:
        writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        if comment=='':
             writer.writerow(['param ' + str(name) + ' := ' + str(param) + ';'])
        else :
             writer.writerow(['param ' + str(name) + ' := ' + str(param) + '; # ' + str(comment)])


## Function to import the data from the CSV data files ##
def import_data(import_folders) :
    ## Reading CSV
    
    # Reading User CSV to build dataframes
    Eud = pd.read_csv(import_folders[0] + '\Demand.csv',sep=';',index_col=2, header=0)
    Resources = pd.read_csv(import_folders[0] + '\Resources.csv',sep=';',index_col=2, header=2)
    Technologies = pd.read_csv(import_folders[0] + '\Technologies.csv',sep=';',index_col=3, header=0)
    
    # Reading Developper CSV to build dataframes
    End_uses_categories = pd.read_csv(import_folders[1] + '\END_USES_CATEGORIES.csv',sep=';')
    Layers_in_out = pd.read_csv(import_folders[1] + '\Layers_in_out.csv',sep=';',index_col=0)
    Storage_characteristics = pd.read_csv(import_folders[1] + '\Storage_Charecteristics.csv',sep=';',index_col=0)
    Storage_eff_in = pd.read_csv(import_folders[1] + '\Storage_eff_in.csv',sep=';',index_col=0)
    Storage_eff_out = pd.read_csv(import_folders[1] + '\Storage_eff_out.csv',sep=';',index_col=0)
    Time_Series = pd.read_csv(import_folders[1] + '\Time_Series.csv',sep=';', header=0, index_col=0)
    
    
    ## Pre-pocessing
    Resources.drop(columns=['Comment'],inplace=True)
    Resources.dropna(axis=0, how='any', inplace=True)
    Technologies.drop(columns=['Comment'],inplace=True)
    Technologies.dropna(axis=0, how='any', inplace=True)
    
    
    return Eud, Resources, Technologies, \
           End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in, Storage_eff_out, Time_Series;


## Function to print the ESTD_data.dat file ##
def print_data(data, out_path, gwp_limit):
    ## Prints the data into .dat file (out_path) with the right syntaxt for AMPL
    
    out_path = out_path+'/ESTD_data.dat'
    
    Eud, Resources, Technologies, \
    End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in, \
    Storage_eff_out, Time_Series = data
    
    ## Pre-processing df
    
    # pre-processing Resources
    Resources_simple = Resources.loc[:,['avail','gwp_op','c_op']]
    Resources_simple.index.name = 'param :'
    Resources_simple = Resources_simple.astype('float')
    # pre-processing Eud
    Eud_simple = Eud.drop(columns=['Category', 'Subcategory', 'Units'])
    Eud_simple.index.name = 'param end_uses_demand_year:'
    Eud_simple = Eud_simple.astype('float')
    # pre_processing Technlogies
    Technologies_simple = Technologies.drop(columns=['Category', 'Subcategory', 'Technologies name'])
    Technologies_simple.index.name = 'param:'
    Technologies_simple = Technologies_simple.astype('float')
    
    ### Developper defined parameters ###
    # Typical days inputs
    N_TD = 12
    Weights = {'ELECTRICITY' : 3, 'HEAT_LOW_T_SH' : 3, 'PV' : 1.5, 'WIND_ONSHORE' : 0.75, 'WIND_OFFSHORE' : 0.75}
    # Economical inputs
    i_rate = 0.015 # [-]
    # Political inputs
    re_share_primary = 0 # [-] Minimum RE share in primary consumption	
    solar_area = 250 # [km^2]
    power_density_pv = 0.2367 # PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2
    power_density_solar_thermal = 0.2857  # Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2
    
    # Technologies shares
    share_mobility_public_min  = 0.199
    share_mobility_public_max  = 0.5
    share_freight_train_min  = 0.109
    share_freight_train_max  = 0.25
    share_freight_road_min  = 0
    share_freight_road_max  = 1
    share_freight_boat_min  = 0.156
    share_freight_boat_max  = 0.3
    share_heat_dhn_min  = 0.02
    share_heat_dhn_max  = 0.37
        
    # Electric vehicles : # km-pass/h/veh. : Gives the equivalence between capacity and number of vehicles.#ev_batt,size [GWh]: Size of batteries per car per technology of EV	
    EVs = pd.DataFrame({'EVs_BATT':['PHEV_BATT','BEV_BATT'], 'vehicule_capacity':[5.04E+01, 5.04E+01], 'batt_per_car':[4.40E-06, 2.40E-06]}, index=['CAR_PHEV','CAR_BEV']) 
    
    # Network
    Loss_network = {'ELECTRICITY' : 4.7E-02,'HEAT_LOW_T_DHN' : 5.0E-02}
    c_grid_extra = 367.8 # cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2	
    import_capacity = 9.72 # [GW] Maximum power of electrical interconnections		
    
    # Storage daily
    STORAGE_DAILY = ['TS_DEC_HP_ELEC', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL', 'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL', 'TS_DEC_DIRECT_ELEC', 'TS_DHN_DAILY', 'BATT_LI']
        
    
           
    
    
    ### Building SETS from data ###
    SECTORS = list(Eud_simple.columns)
    END_USES_INPUT = list(Eud_simple.index)
    END_USES_CATEGORIES = list(End_uses_categories.loc[:,'END_USES_CATEGORIES'].unique())
    RESOURCES = list(Resources_simple.index)
    BIOFUELS = list(Resources[Resources.loc[:,'Subcategory']=='Biofuels'].index)+['SNG']
    RE_RESOURCES = list(Resources.loc[(Resources['Category']=='Renewable') & (Resources['Subcategory']!='Electro-fuels'),:].index)
    EXPORT = list(Resources.loc[Resources['Category']=='Export',:].index)
    
    END_USES_TYPES_OF_CATEGORY = []
    for i in END_USES_CATEGORIES:
        l = list(End_uses_categories.loc[End_uses_categories.loc[:,'END_USES_CATEGORIES']==i,'END_USES_TYPES_OF_CATEGORY'])
        END_USES_TYPES_OF_CATEGORY.append(l)
    
    
    # TECHNOLOGIES_OF_END_USES_TYPE -> # METHOD 2 (uses layer_in_out to determine the END_USES_TYPE)
    END_USES_TYPES = list(End_uses_categories.loc[:,'END_USES_TYPES_OF_CATEGORY'])
    ALL_TECHS = list(Technologies_simple.index) 
    
    Layers_in_out_tech = Layers_in_out.loc[~Layers_in_out.index.isin(RESOURCES),:]
    TECHNOLOGIES_OF_END_USES_TYPE = [] 
    for i in END_USES_TYPES:
        l = list(Layers_in_out_tech.loc[Layers_in_out_tech.loc[:,i]==1,:].index)
        TECHNOLOGIES_OF_END_USES_TYPE.append(l)
    
    # STORAGE and INFRASTRUCTURES
    ALL_TECH_OF_EUT = [item for sublist in TECHNOLOGIES_OF_END_USES_TYPE for item in sublist]
           
    STORAGE_TECH = list(Storage_eff_in.index)
    INFRASTRUCTURE =[item for item in ALL_TECHS if item not in STORAGE_TECH and item not in ALL_TECH_OF_EUT]
    
    # EVs
    EVs_BATT = list(EVs.loc[:,'EVs_BATT'])
    V2G = list(EVs.index)
       
    ## defined nowhere
    STORAGE_DAILY = ['TS_DEC_HP_ELEC', 'TS_DEC_THHP_GAS', 'TS_DEC_COGEN_GAS', 'TS_DEC_COGEN_OIL', 'TS_DEC_ADVCOGEN_GAS', 'TS_DEC_ADVCOGEN_H2', 'TS_DEC_BOILER_GAS', 'TS_DEC_BOILER_WOOD', 'TS_DEC_BOILER_OIL', 'TS_DEC_DIRECT_ELEC', 'TS_DHN_DAILY', 'BATT_LI', 'TS_HIGH_TEMP']
    
    # STORAGE_OF_END_USES_TYPES ->  #METHOD 2 (using Storage_eff_in)
    STORAGE_OF_END_USES_TYPES_DHN = [] 
    STORAGE_OF_END_USES_TYPES_DEC = [] 
    STORAGE_OF_END_USES_TYPES_ELEC = []
    STORAGE_OF_END_USES_TYPES_HIGH_T = []
   
    for i in STORAGE_TECH :
        if Storage_eff_in.loc[i,'HEAT_LOW_T_DHN']>0 :
            STORAGE_OF_END_USES_TYPES_DHN.append(i)
        elif Storage_eff_in.loc[i,'HEAT_LOW_T_DECEN']>0 :
            STORAGE_OF_END_USES_TYPES_DEC.append(i)
        elif Storage_eff_in.loc[i,'ELECTRICITY']>0 :
            STORAGE_OF_END_USES_TYPES_ELEC.append(i)
        elif Storage_eff_in.loc[i,'HEAT_HIGH_T']>0 :
            STORAGE_OF_END_USES_TYPES_HIGH_T.append(i)
            
    
    # etc. still TS_OF_DEC_TECH and EVs_BATT_OF_V2G missing... -> hard coded !
    
    
    COGEN = []
    BOILERS = []
    
    for i in ALL_TECH_OF_EUT:
        if Layers_in_out.loc[i,'HEAT_HIGH_T']==1 or Layers_in_out.loc[i,'HEAT_LOW_T_DHN']==1 or Layers_in_out.loc[i,'HEAT_LOW_T_DECEN']==1:
            if Layers_in_out.loc[i,'ELECTRICITY']>0:
                COGEN.append(i)
            else :
                BOILERS.append(i)
    
    ### Adding AMPL syntax ###
    # creating Batt_per_Car_df for printing
    Batt_per_Car_df = EVs[['batt_per_car']]
    Vehicule_capacity_df = EVs[['vehicule_capacity']]
    Loss_network_df = pd.DataFrame(data=Loss_network.values(), index=Loss_network.keys(), columns=[' '])
    # Putting all the df in ampl syntax
    Batt_per_Car_df = ampl_syntax(Batt_per_Car_df,'# ev_batt,size [GWh]: Size of batteries per car per technology of EV')
    Vehicule_capacity_df = ampl_syntax (Vehicule_capacity_df, '# km-pass/h/veh. : Gives the equivalence between capacity and number of vehicles.')
    Eud_simple = ampl_syntax(Eud_simple,'')
    Layers_in_out = ampl_syntax(Layers_in_out,'')
    Technologies_simple = ampl_syntax(Technologies_simple,'')
    Resources_simple = ampl_syntax(Resources_simple,'')
    Storage_eff_in = ampl_syntax(Storage_eff_in,'')
    Storage_eff_out = ampl_syntax(Storage_eff_out,'')
    Storage_characteristics = ampl_syntax(Storage_characteristics,'')
    Loss_network_df = ampl_syntax(Loss_network_df,'')
        
    ### Printing data ###
    ## printing signature of data file
    with open(out_path, mode='w',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    
            TD_writer.writerow(['# -------------------------------------------------------------------------------------------------------------------------	'])
            TD_writer.writerow(['#	EnergyScope TD is an open-source energy model suitable for country scale analysis. It is a simplified representation of an urban or national energy system accounting for the energy flows'])
            TD_writer.writerow(['#	within its boundaries. Based on a hourly resolution, it optimises the design and operation of the energy system while minimizing the cost of the system.'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
            TD_writer.writerow(['#	you may not use this file except in compliance with the License.'])
            TD_writer.writerow(['#	You may obtain a copy of the License at'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
            TD_writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
            TD_writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
            TD_writer.writerow(['#	See the License for the specific language governing permissions and'])
            TD_writer.writerow(['#	limitations under the License.'])
            TD_writer.writerow(['#	'])
            TD_writer.writerow(['#	Description and complete License: see LICENSE file.'])
            TD_writer.writerow(['# -------------------------------------------------------------------------------------------------------------------------	'])
            TD_writer.writerow(['	'])
            TD_writer.writerow(['# UNIT MEASURES:'])
            TD_writer.writerow(['# Unless otherwise specified units are:'])
            TD_writer.writerow(['# Energy [GWh], Power [GW], Cost [MCHF], Time [h], Passenger transport [Mpkm], Freight Transport [Mtkm]'])
            TD_writer.writerow(['	'])
            TD_writer.writerow(['# References based on Supplementary material'])
            TD_writer.writerow(['# --------------------------	'])
            TD_writer.writerow(['# SETS not depending on TD	'])
            TD_writer.writerow(['# --------------------------	'])
            TD_writer.writerow(['	'])
            
    ## printing sets
    print_set(SECTORS,'SECTORS', out_path)
    print_set(END_USES_INPUT,'END_USES_INPUT', out_path)
    print_set(END_USES_CATEGORIES,'END_USES_CATEGORIES', out_path) 
    print_set(RESOURCES,'RESOURCES', out_path) 
    print_set(BIOFUELS,'BIOFUELS', out_path)
    print_set(RE_RESOURCES,'RE_RESOURCES', out_path)
    print_set(EXPORT,'EXPORT', out_path)
    newline(out_path)
    n=0
    for j in END_USES_TYPES_OF_CATEGORY:
        print_set(j,'END_USES_TYPES_OF_CATEGORY'+'["'+ END_USES_CATEGORIES[n] +'"]', out_path)
        n += 1    
    newline(out_path)
    n=0
    for j in TECHNOLOGIES_OF_END_USES_TYPE:
        print_set(j,'TECHNOLOGIES_OF_END_USES_TYPE'+'["'+ END_USES_TYPES[n] +'"]', out_path)
        n += 1    
    newline(out_path)
    print_set(STORAGE_TECH,'STORAGE_TECH', out_path)
    print_set(INFRASTRUCTURE,'INFRASTRUCTURE', out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage subsets'])        
    print_set(EVs_BATT,'EVs_BATT', out_path)
    print_set(V2G,'V2G', out_path)
    print_set(STORAGE_DAILY,'STORAGE_DAILY', out_path)
    newline(out_path)
    print_set(STORAGE_OF_END_USES_TYPES_DHN,'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DHN"]', out_path)
    print_set(STORAGE_OF_END_USES_TYPES_DEC,'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DECEN"]', out_path)
    print_set(STORAGE_OF_END_USES_TYPES_ELEC,'STORAGE_OF_END_USES_TYPES ["ELECTRICITY"]', out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Link between storages & specific technologies	'])
    # Hardcoded
    print_set(['TS_DEC_HP_ELEC'],'TS_OF_DEC_TECH ["DEC_HP_ELEC"]', out_path)
    print_set(['TS_DEC_DIRECT_ELEC'],'TS_OF_DEC_TECH ["DEC_DIRECT_ELEC"]', out_path)
    print_set(['TS_DEC_THHP_GAS'],'TS_OF_DEC_TECH ["DEC_THHP_GAS"]', out_path)
    print_set(['TS_DEC_COGEN_GAS'],'TS_OF_DEC_TECH ["DEC_COGEN_GAS"]', out_path)
    print_set(['TS_DEC_ADVCOGEN_GAS'],'TS_OF_DEC_TECH ["DEC_ADVCOGEN_GAS"]', out_path)
    print_set(['TS_DEC_COGEN_OIL'],'TS_OF_DEC_TECH ["DEC_COGEN_OIL"]', out_path)
    print_set(['TS_DEC_ADVCOGEN_H2'],'TS_OF_DEC_TECH ["DEC_ADVCOGEN_H2"]', out_path)
    print_set(['TS_DEC_BOILER_GAS'],'TS_OF_DEC_TECH ["DEC_BOILER_GAS"]', out_path)
    print_set(['TS_DEC_BOILER_WOOD'],'TS_OF_DEC_TECH ["DEC_BOILER_WOOD"]', out_path)
    print_set(['TS_DEC_BOILER_OIL'],'TS_OF_DEC_TECH ["DEC_BOILER_OIL"]', out_path)
    print_set(['PHEV_BATT'],'EVs_BATT_OF_V2G ["CAR_PHEV"]', out_path)
    print_set(['BEV_BATT'],'EVs_BATT_OF_V2G ["CAR_BEV"]', out_path)        
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Additional sets, just needed for printing results	'])
    print_set(COGEN,'COGEN', out_path)
    print_set(BOILERS,'BOILERS', out_path)
    newline(out_path)
    
    ## printing parameters
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# -----------------------------'])
            TD_writer.writerow(['# PARAMETERS NOT DEPENDING ON THE NUMBER OF TYPICAL DAYS : '])
            TD_writer.writerow(['# -----------------------------	'])
            TD_writer.writerow([''])
            TD_writer.writerow(['## PARAMETERS presented in Table 2.	'])
    # printing i_rate, re_share_primary,gwp_limit,solar_area
    print_param('i_rate',i_rate,'part [2.7.4]', out_path)
    print_param('re_share_primary',re_share_primary,'Minimum RE share in primary consumption', out_path)
    print_param('gwp_limit',gwp_limit,'gwp_limit [ktCO2-eq./year]: maximum GWP emissions', out_path)
    print_param('solar_area',solar_area,'', out_path)
    print_param('power_density_pv',power_density_pv,'PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2', out_path)
    print_param('power_density_solar_thermal',power_density_solar_thermal, 'Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2', out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Part [2.4]	'])        
    print_df('param:', Batt_per_Car_df, out_path)
    newline(out_path)
    print_df('param:', Vehicule_capacity_df, out_path)
    newline(out_path)
    # printing c_grid_extra and import_capacity 
    print_param('c_grid_extra',c_grid_extra,'cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2', out_path)
    print_param('import_capacity',import_capacity,'', out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# end_Uses_year see part [2.1]'])        
    print_df('param end_uses_demand_year : ', Eud_simple, out_path)
    newline(out_path)
    print_param('share_mobility_public_min',share_mobility_public_min,'', out_path)
    print_param('share_mobility_public_max',share_mobility_public_max,'', out_path)
    newline(out_path)
    print_param('share_freight_train_min',share_freight_train_min,'', out_path)
    print_param('share_freight_train_max',share_freight_train_max,'', out_path)
    newline(out_path)
    print_param('share_freight_road_min',share_freight_road_min,'', out_path)
    print_param('share_freight_road_max',share_freight_road_max,'', out_path)
    newline(out_path)
    print_param('share_freight_boat_min',share_freight_boat_min,'', out_path)
    print_param('share_freight_boat_max',share_freight_boat_max,'', out_path)
    newline(out_path)
    print_param('share_heat_dhn_min',share_heat_dhn_min,'', out_path)
    print_param('share_heat_dhn_max',share_heat_dhn_max,'', out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Link between layers  (data from Tables 19,21,22,23,25,29,30)'])        
    print_df('param layers_in_out : ', Layers_in_out, out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Technologies data from Tables (10,19,21,22,23,25,27,28,29,30) and part [2.2.1.1] for hydro'])        
    print_df('param :', Technologies_simple, out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# RESOURCES: part [2.5] (Table 26)'])        
    print_df('param :', Resources_simple, out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage inlet/outlet efficiency : part [2.6] (Table 28) and part [2.2.1.1] for hydro.	'])
    print_df('param storage_eff_in :', Storage_eff_in, out_path)
    newline(out_path)
    print_df('param storage_eff_out :', Storage_eff_out, out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage characteristics : part [2.6] (Table 28) and part [2.2.1.1] for hydro.'])
    print_df('param :', Storage_characteristics, out_path)
    newline(out_path)
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# [A.6]'])
    print_df('param loss_network ', Loss_network_df, out_path)   
    
    return

## Function to print the ESTD_12TD.dat file from timeseries and STEP1 results ##
def print_TD_data(timeseries, out_path='STEP_2_Energy_Model', step1_out='STEP_1_TD_selection\TD_of_days.out',Nbr_TD=12,
                  end_uses_reserve=pd.DataFrame(np.zeros((8760,1)), columns=['end_uses_reserve'], index=np.arange(1,8761,1))):
    
    ## DICTIONARIES TO TRANSLATE NAMES INTO AMPL SYNTAX ##
    # for EUD timeseries
    EUD_params = {'Electricity (%_elec)' : 'param electricity_time_series :',
                  'Space Heating (%_sh)' : 'param heating_time_series :',
                  'Passanger mobility (%_pass)' : 'param mob_pass_time_series :',
                  'Freight mobility (%_freight)' : 'param mob_freight_time_series :'}
    # for resources timeseries that have only 1 tech linked to it
    RES_params = {'PV' : 'PV', 'Wind_onshore': 'WIND_ONSHORE', 'Wind_offshore' : 'WIND_OFFSHORE',
                  'Hydro_river': 'HYDRO_RIVER'}
    # for resources timeseries that have several techs linked to it
    RES_mult_params = {'Solar' : ['DHN_SOLAR', 'DEC_SOLAR']}
    
    
    ## Merge end_uses__reserve to the other timeseries ##
    timeseries = timeseries.merge(end_uses_reserve,left_index=True, right_index=True)
    
    ## Redefine the output file from the out_path given ##
    out_path = out_path+'/ESTD_'+str(Nbr_TD)+'TD.dat'
    
    ## READING OUTPUT OF STEP1 ##
    TD_of_days = pd.read_csv(step1_out, names=['TD_of_days'])
    TD_of_days['day'] = np.arange(1,366,1) # putting the days of the year beside
    
    ## COMPUTING NUMBER OF DAYS REPRESENTED BY EACH TD ##
    sorted_TD = TD_of_days.groupby('TD_of_days').count()
    sorted_TD.rename(columns={'day':'#days'}, inplace=True)
    sorted_TD.reset_index(inplace=True)
    sorted_TD.set_index(np.arange(1,Nbr_TD+1), inplace=True) # adding number of TD as index
    
    ## BUILDING T_H_TD MATRICE ##
    # generate T_H_TD
    TD_and_hour_array = np.ones((24*365,2))
    for i in range(365):
        TD_and_hour_array[i*24:(i+1)*24,0] = np.arange(1,25,1)
        TD_and_hour_array[i*24:(i+1)*24,1] = TD_and_hour_array[i*24:(i+1)*24,1]*sorted_TD[sorted_TD['TD_of_days']==TD_of_days.loc[i,'TD_of_days']].index.values
    T_H_TD = pd.DataFrame(TD_and_hour_array, index = np.arange(1,8761,1), columns=['H_of_D','TD_of_day'])
    T_H_TD = T_H_TD.astype('int64')
    # giving the right syntax
    T_H_TD.reset_index(inplace=True)
    T_H_TD.rename(columns={'index':'H_of_Y'}, inplace=True)
    T_H_TD['par_g'] = '('
    T_H_TD['par_d'] = ')'
    T_H_TD['comma1'] = ','
    T_H_TD['comma2'] = ','
    # giving the right order to the columns
    T_H_TD = T_H_TD[['par_g','H_of_Y','comma1','H_of_D','comma2','TD_of_day','par_d']]
    
   

    # COMPUTING THE NORM OVER THE YEAR ##
    norm = timeseries.sum(axis=0)
    norm.index.rename('Category', inplace=True)
    norm.name = 'Norm'
    
    ## BUILDING TD TIMESERIES ##
    # creating df with 2 columns : day of the year | hour in the day
    day_and_hour_array = np.ones((24*365,2))
    for i in range(365):
        day_and_hour_array[i*24:(i+1)*24,0] = day_and_hour_array[i*24:(i+1)*24,0]*(i+1)
        day_and_hour_array[i*24:(i+1)*24,1] = np.arange(1,25,1)
    day_and_hour = pd.DataFrame(day_and_hour_array, index = np.arange(1,8761,1), columns=['D_of_H','H_of_D'])
    day_and_hour = day_and_hour.astype('int64')
    timeseries = timeseries.merge(day_and_hour,left_index=True, right_index=True)

    #selecting timeseries of TD only
    TD_ts = timeseries[timeseries['D_of_H'].isin(sorted_TD['TD_of_days'])]
    
    ## COMPUTING THE NORM_TD OVER THE YEAR FOR CORRECTION ##
    # computing the sum of ts over each TD
    agg_TD_ts = TD_ts.groupby('D_of_H').sum()
    agg_TD_ts.reset_index(inplace=True)
    agg_TD_ts.set_index(np.arange(1,Nbr_TD+1), inplace=True)
    agg_TD_ts.drop(columns=['D_of_H','H_of_D'], inplace=True)
    # multiplicating each TD by the number of day it represents
    for c in agg_TD_ts.columns:
        agg_TD_ts[c] = agg_TD_ts[c]*sorted_TD['#days']
    # sum of new ts over the whole year
    norm_TD = agg_TD_ts.sum()
    
    ## BUILDING THE DF WITH THE TS OF EACH TD FOR EACH CATEGORY ##
    # pivoting TD_ts to obtain a (24,Nbr_TD*Nbr_ts*N_c)
    all_TD_ts = TD_ts.pivot(index='H_of_D', columns='D_of_H')
    
    
    ## COMPUTE peak_sh_factor ##
    peak_sh_factor = 1
    max_sh_TD = TD_ts.loc[:,'Space Heating (%_sh)'].max()
    max_sh_all = timeseries.loc[:,'Space Heating (%_sh)'].max()
    peak_sh_factor = max_sh_all/max_sh_TD

        
    
    
    ## PRINTING ##
    # printing description of file
    with open(out_path, mode='w',newline='') as TD_file:
        TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    # Comments and license
        TD_writer.writerow(['# -------------------------------------------------------------------------------------------------------------------------	'])
        TD_writer.writerow(['#	EnergyScope TD is an open-source energy model suitable for country scale analysis. It is a simplified representation of an urban or national energy system accounting for the energy flows'])
        TD_writer.writerow(['#	within its boundaries. Based on a hourly resolution, it optimises the design and operation of the energy system while minimizing the cost of the system.'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
        TD_writer.writerow(['#	you may not use this file except in compliance with the License.'])
        TD_writer.writerow(['#	You may obtain a copy of the License at'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
        TD_writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
        TD_writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
        TD_writer.writerow(['#	See the License for the specific language governing permissions and'])
        TD_writer.writerow(['#	limitations under the License.'])
        TD_writer.writerow(['#	'])
        TD_writer.writerow(['#	Description and complete License: see LICENSE file.'])
        TD_writer.writerow(['# -------------------------------------------------------------------------------------------------------------------------	'])
        TD_writer.writerow(['	'])
    # peak_sh_factor
        TD_writer.writerow(['# SETS depending on TD	'])
        TD_writer.writerow(['# --------------------------	'])
        TD_writer.writerow(['param peak_sh_factor	:=	' + str(peak_sh_factor)])
        TD_writer.writerow([';		'])
        TD_writer.writerow(['		'])
        
    # printing T_H_TD param
        TD_writer.writerow(['#SETS [Figure 3]		'])
        TD_writer.writerow(['set T_H_TD := 		'])
    
    
    T_H_TD.to_csv(out_path,sep='\t', header=False, index=False, mode='a', quoting=csv.QUOTE_NONE)
    
    # printing interlude
    with open(out_path, mode='a',newline='') as TD_file:
        TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

        TD_writer.writerow([';'])
        TD_writer.writerow([''])
        TD_writer.writerow(['# -----------------------------'])
        TD_writer.writerow(['# PARAMETERS DEPENDING ON NUMBER OF TYPICAL DAYS : '])
        TD_writer.writerow(['# -----------------------------'])
        TD_writer.writerow([''])
        
   
    # printing EUD timeseries param
    for l in EUD_params.keys():
        ts = all_TD_ts[l]
        ts.columns = np.arange(1,Nbr_TD+1) 
        ts = ts*norm[l]/norm_TD[l]
        ts.fillna(0, inplace=True)
        
        ts = ampl_syntax(ts,'')
        print_df(EUD_params[l]+' :', ts, out_path)
        newline(out_path)

    # printing c_p_t param #
    with open(out_path, mode='a',newline='') as TD_file:
        TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        TD_writer.writerow(['param c_p_t:='])    
    # printing c_p_t part where 1 ts => 1 tech
    for l in RES_params.keys():
        ts = all_TD_ts[l]
        ts.columns = np.arange(1,Nbr_TD+1) 
        ts = ts*norm[l]/norm_TD[l]
        ts.fillna(0, inplace=True)
        
        ts = ampl_syntax(ts, '')
        s = '["'+RES_params[l]+'",*,*]:'
        print_df(s, ts, out_path)
        newline(out_path)
        
    # printing c_p_t part where 1 ts => more then 1 tech        
    for l in RES_mult_params.keys():
        for j in RES_mult_params[l]:
            ts = all_TD_ts[l]
            ts.columns = np.arange(1,Nbr_TD+1) 
            ts = ts*norm[l]/norm_TD[l]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            print_df('["'+j+'",*,*]:', ts, out_path)
            newline(out_path)
            
    # printing end_uses_reserve ts
    with open(out_path, mode='a',newline='') as TD_file:
        TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        TD_writer.writerow(['param end_uses_reserve:='])
        
    ts = all_TD_ts['end_uses_reserve']
    ts.columns = np.arange(1,Nbr_TD+1) 
    ts = ts*norm[l]/norm_TD[l]
    ts.fillna(0, inplace=True)
    ts = ampl_syntax(ts, '')
    print_df('["ELECTRICITY",*,*]:', ts, out_path)
    newline(out_path)
   
    return

if __name__ == '__main__':
    ## User defined
    gwp_limit = 150000 # [ktCO2-eq./year]	# Minimum GWP reduction 
    
    ## Path for data and outputs
    import_folders = ['Data\\User_data', 'Data\\Developper_data']
    out_path = 'STEP_2_Energy_Model'
    step1_out='STEP_1_TD_selection\TD_of_days.out'
    
    # Reading the data
    Eud, Resources, Technologies, \
    End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in, \
    Storage_eff_out, Time_Series = import_data(import_folders)
    
    
    # Printing ESTD_data.dat
    data = (Eud, Resources, Technologies, \
    End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in, \
    Storage_eff_out, Time_Series)
    print_data(data, out_path, gwp_limit)
    
    # Printing ESD_12TD.dat
    print_TD_data(Time_Series, out_path, step1_out)