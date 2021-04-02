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
    Time_Series = pd.read_csv(import_folders[1] + '\Time_Series.csv',sep=';',index_col=0)
    
    
    ## Pre-pocessing
    Resources.drop(columns=['Comment'],inplace=True)
    Resources.dropna(axis=0, how='any', inplace=True)
    Technologies.drop(columns=['Comment'],inplace=True)
    Technologies.dropna(axis=0, how='any', inplace=True)
    
    
    return Eud, Resources, Technologies, \
           End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in, Storage_eff_out, Time_Series;


def print_data(data, out_path, gwp_limit):
    ## Prints the data into .dat file (out_path) with the right syntaxt for AMPL
    
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
        
    
    ### Usefull functions ###
    def ampl_syntax(df,comment) :
        # adds ampl syntax to df
        df2=df.copy()
        df2.rename(columns = {df2.columns[df2.shape[1]-1] : str(df2.columns[df2.shape[1]-1])+' '+':= '+comment}, inplace=True)
        return df2
    
    def print_set(my_set,name) :
        with open(out_path, mode='a',newline='') as TD_file:
            writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['set ' + name +' := \t' + '\t'.join(my_set) + ';'])
            
    def print_df(name, df) : # should add the comment for the param
        df.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=name)
    
        with open(out_path, mode='a',newline='') as TD_file:
            writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([';'])
        
    def newline():
        with open(out_path, mode='a',newline='') as TD_file:
            writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([''])
            
    def print_param(name,param,comment):
        with open(out_path, mode='a',newline='') as TD_file:
            writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            if comment=='':
                 writer.writerow(['param ' + str(name) + ' := ' + str(param) + ';'])
            else :
                 writer.writerow(['param ' + str(name) + ' := ' + str(param) + '; # ' + str(comment)])
           
    
    
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
    print_set(SECTORS,'SECTORS')
    print_set(END_USES_INPUT,'END_USES_INPUT')
    print_set(END_USES_CATEGORIES,'END_USES_CATEGORIES') 
    print_set(RESOURCES,'RESOURCES') 
    print_set(BIOFUELS,'BIOFUELS')
    print_set(RE_RESOURCES,'RE_RESOURCES')
    print_set(EXPORT,'EXPORT')
    newline()
    n=0
    for j in END_USES_TYPES_OF_CATEGORY:
        print_set(j,'END_USES_TYPES_OF_CATEGORY'+'["'+ END_USES_CATEGORIES[n] +'"]')
        n += 1    
    newline()
    n=0
    for j in TECHNOLOGIES_OF_END_USES_TYPE:
        print_set(j,'TECHNOLOGIES_OF_END_USES_TYPE'+'["'+ END_USES_TYPES[n] +'"]')
        n += 1    
    newline()
    print_set(STORAGE_TECH,'STORAGE_TECH')
    print_set(INFRASTRUCTURE,'INFRASTRUCTURE')
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage subsets'])        
    print_set(EVs_BATT,'EVs_BATT')
    print_set(V2G,'V2G')
    print_set(STORAGE_DAILY,'STORAGE_DAILY')
    newline()
    print_set(STORAGE_OF_END_USES_TYPES_DHN,'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DHN"]')
    print_set(STORAGE_OF_END_USES_TYPES_DEC,'STORAGE_OF_END_USES_TYPES ["HEAT_LOW_T_DECEN"]')
    print_set(STORAGE_OF_END_USES_TYPES_ELEC,'STORAGE_OF_END_USES_TYPES ["ELECTRICITY"]')
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Link between storages & specific technologies	'])
    # Hardcoded
    print_set(['TS_DEC_HP_ELEC'],'TS_OF_DEC_TECH ["DEC_HP_ELEC"]')
    print_set(['TS_DEC_DIRECT_ELEC'],'TS_OF_DEC_TECH ["DEC_DIRECT_ELEC"]')
    print_set(['TS_DEC_THHP_GAS'],'TS_OF_DEC_TECH ["DEC_THHP_GAS"]')
    print_set(['TS_DEC_COGEN_GAS'],'TS_OF_DEC_TECH ["DEC_COGEN_GAS"]')
    print_set(['TS_DEC_ADVCOGEN_GAS'],'TS_OF_DEC_TECH ["DEC_ADVCOGEN_GAS"]')
    print_set(['TS_DEC_COGEN_OIL'],'TS_OF_DEC_TECH ["DEC_COGEN_OIL"]')
    print_set(['TS_DEC_ADVCOGEN_H2'],'TS_OF_DEC_TECH ["DEC_ADVCOGEN_H2"]')
    print_set(['TS_DEC_BOILER_GAS'],'TS_OF_DEC_TECH ["DEC_BOILER_GAS"]')
    print_set(['TS_DEC_BOILER_WOOD'],'TS_OF_DEC_TECH ["DEC_BOILER_WOOD"]')
    print_set(['TS_DEC_BOILER_OIL'],'TS_OF_DEC_TECH ["DEC_BOILER_OIL"]')
    print_set(['PHEV_BATT'],'EVs_BATT_OF_V2G ["CAR_PHEV"]')
    print_set(['BEV_BATT'],'EVs_BATT_OF_V2G ["CAR_BEV"]')        
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Additional sets, just needed for printing results	'])
    print_set(COGEN,'COGEN')
    print_set(BOILERS,'BOILERS')
    newline()
    
    ## printing parameters
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# -----------------------------'])
            TD_writer.writerow(['# PARAMETERS NOT DEPENDING ON THE NUMBER OF TYPICAL DAYS : '])
            TD_writer.writerow(['# -----------------------------	'])
            TD_writer.writerow([''])
            TD_writer.writerow(['## PARAMETERS presented in Table 2.	'])
    # printing i_rate, re_share_primary,gwp_limit,solar_area
    print_param('i_rate',i_rate,'part [2.7.4]')
    print_param('re_share_primary',re_share_primary,'Minimum RE share in primary consumption')
    print_param('gwp_limit',gwp_limit,'gwp_limit [ktCO2-eq./year]: maximum GWP emissions')
    print_param('solar_area',solar_area,'')
    print_param('power_density_pv',power_density_pv,'PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2')
    print_param('power_density_solar_thermal',power_density_solar_thermal, 'Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2')
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Part [2.4]	'])        
    print_df('param:', Batt_per_Car_df)
    newline()
    print_df('param:', Vehicule_capacity_df)
    newline()
    # printing c_grid_extra and import_capacity 
    print_param('c_grid_extra',c_grid_extra,'cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2')
    print_param('import_capacity',import_capacity,'')
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# end_Uses_year see part [2.1]'])        
    print_df('param end_uses_demand_year : ', Eud_simple)
    newline()
    print_param('share_mobility_public_min',share_mobility_public_min,'')
    print_param('share_mobility_public_max',share_mobility_public_max,'')
    newline()
    print_param('share_freight_train_min',share_freight_train_min,'')
    print_param('share_freight_train_max',share_freight_train_max,'')
    newline()
    print_param('share_freight_road_min',share_freight_road_min,'')
    print_param('share_freight_road_max',share_freight_road_max,'')
    newline()
    print_param('share_freight_boat_min',share_freight_boat_min,'')
    print_param('share_freight_boat_max',share_freight_boat_max,'')
    newline()
    print_param('share_heat_dhn_min',share_heat_dhn_min,'')
    print_param('share_heat_dhn_max',share_heat_dhn_max,'')
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Link between layers  (data from Tables 19,21,22,23,25,29,30)'])        
    print_df('param layers_in_out : ', Layers_in_out)
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Technologies data from Tables (10,19,21,22,23,25,27,28,29,30) and part [2.2.1.1] for hydro'])        
    print_df('param :', Technologies_simple)
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# RESOURCES: part [2.5] (Table 26)'])        
    print_df('param :', Resources_simple)
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage inlet/outlet efficiency : part [2.6] (Table 28) and part [2.2.1.1] for hydro.	'])
    print_df('param storage_eff_in :', Storage_eff_in)
    newline()
    print_df('param storage_eff_out :', Storage_eff_out)
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# Storage characteristics : part [2.6] (Table 28) and part [2.2.1.1] for hydro.'])
    print_df('param :', Storage_characteristics)
    newline()
    with open(out_path, mode='a',newline='') as TD_file:
            TD_writer = csv.writer(TD_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            TD_writer.writerow(['# [A.6]'])
    print_df('param loss_network ', Loss_network_df)   
    
    return
