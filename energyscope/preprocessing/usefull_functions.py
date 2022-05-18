# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:26:29 2020

Contains functions to read data in csv files and print it with AMPL syntax in ESTD_data.dat
Also contains functions to analyse input data

@author: Paolo Thiran
"""
import logging

import numpy as np
import pandas as pd
import csv
import os
import json
import shutil
from subprocess import run

from pathlib import Path

# TODO
#  add step1 and reading of weights
#  check how to include efficiency as in pathway
#  update units c_maint in data
#  check data efficiency electrolyser -> XR
#  update for nuclear up to 2035


# Useful functions for printing in AMPL syntax #
def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def ampl_syntax(df, comment):
    # adds ampl syntax to df
    df2 = df.copy()
    df2.rename(columns={df2.columns[df2.shape[1] - 1]: str(df2.columns[df2.shape[1] - 1]) + ' ' + ':= ' + comment},
               inplace=True)
    return df2


def print_set(my_set, name, out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['set ' + name + ' := \t' + '\t'.join(my_set) + ';'])


def print_df(name, df, out_path):
    df.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=name, quoting=csv.QUOTE_NONE)

    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([';'])


def newline(out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([''])


def print_param(name, param, comment, out_path):
    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        if comment == '':
            writer.writerow(['param ' + str(name) + ' := ' + str(param) + ';'])
        else:
            writer.writerow(['param ' + str(name) + ' := ' + str(param) + '; # ' + str(comment)])


def print_json(my_sets, file):  # printing the dictionary containing all the sets into directory/sets.json
    with open(file, 'w') as fp:
        json.dump(my_sets, fp, indent=4, sort_keys=True)
    return

def read_json(file):
    # reading the saved dictionary containing all the sets from directory/sets.json
    with open(file, 'r') as fp:
        data = json.load(fp)
    return data


# Function to import the data from the CSV data files #
def import_data(config):

    import_folders = config['data_dir']
    logging.info('Importing data files from '+ str(import_folders))
    # Reading CSV #
    eud = pd.read_csv(import_folders / 'Demand.csv', sep=',', index_col=2, header=0)
    resources = pd.read_csv(import_folders / 'Resources.csv', sep=',', index_col=2, header=2)
    technologies = pd.read_csv(import_folders / 'Technologies.csv', sep=',', index_col=3, header=0, skiprows=[1])
    end_uses_categories = pd.read_csv(import_folders / 'END_USES_CATEGORIES.csv', sep=',')
    layers_in_out = pd.read_csv(import_folders / 'Layers_in_out.csv', sep=',', index_col=0)
    storage_characteristics = pd.read_csv(import_folders / 'Storage_characteristics.csv', sep=',', index_col=0)
    storage_eff_in = pd.read_csv(import_folders / 'Storage_eff_in.csv', sep=',', index_col=0)
    storage_eff_out = pd.read_csv(import_folders / 'Storage_eff_out.csv', sep=',', index_col=0)
    time_series = pd.read_csv(import_folders / 'Time_series.csv', sep=',', header=0, index_col=0)

    # Reading user_defined.json
    config['user_defined'] = read_json(import_folders / 'user_defined.json')

    # Pre-processing #
    resources.drop(columns=['Comment'], inplace=True)
    resources.dropna(axis=0, how='any', inplace=True)
    technologies.drop(columns=['Comment'], inplace=True)
    technologies.dropna(axis=0, how='any', inplace=True)
    # cleaning indices and columns

    all_df = {'Demand': eud, 'Resources': resources, 'Technologies': technologies,
              'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out,
              'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in,
              'Storage_eff_out': storage_eff_out, 'Time_series': time_series}

    for key in all_df:
        if type(all_df[key].index[0]) == str:
            all_df[key].index = all_df[key].index.str.strip()
        if type(all_df[key].columns[0]) == str:
            all_df[key].columns = all_df[key].columns.str.strip()

    config['all_data'] = all_df
    return


# Function to print the ESTD_data.dat file #
def print_data(config, case = 'deter'):
    #two_up = os.path.dirname(os.path.dirname(__file__))
    two_up = Path(__file__).parents[2]
    
    if case=='deter':
        cs = os.path.join(two_up,'case_studies/')
        make_dir(cs)
    else:
        cs = os.path.join(two_up,'case_studies')
        make_dir(cs)
        cs = cs + '/' + config['UQ_case'] + '/'
        make_dir(cs)
    make_dir(cs + config['case_study'])

    data = config['all_data']

    eud = data['Demand']
    resources = data['Resources']
    technologies = data['Technologies']
    end_uses_categories = data['End_uses_categories']
    layers_in_out = data['Layers_in_out']
    storage_characteristics = data['Storage_characteristics']
    storage_eff_in = data['Storage_eff_in']
    storage_eff_out = data['Storage_eff_out']
    time_series = data['Time_series']

    if config['printing']:
        logging.info('Printing ESTD_data.dat')

        # Prints the data into .dat file (out_path) with the right syntax for AMPL
        out_path = cs + config['case_study'] + '/ESTD_data.dat'
        # config['ES_path'] + '/ESTD_data.dat'
        gwp_limit = config['GWP_limit']

        # Pre-processing df #
        # pre-processing resources
        resources_simple = resources.loc[:, ['avail', 'gwp_op', 'c_op']]
        resources_simple.index.name = 'param :'
        resources_simple = resources_simple.astype('float')
        # pre-processing eud
        eud_simple = eud.drop(columns=['Category', 'Subcategory', 'Units'])
        eud_simple.index.name = 'param end_uses_demand_year:'
        eud_simple = eud_simple.astype('float')
        # pre_processing technologies
        technologies_simple = technologies.drop(columns=['Category', 'Subcategory', 'Technologies name'])
        technologies_simple.index.name = 'param:'
        technologies_simple = technologies_simple.astype('float')

        # Economical inputs
        i_rate = config['user_defined']['i_rate']  # [-]
        # Political inputs
        re_share_primary = config['user_defined']['re_share_primary']  # [-] Minimum RE share in primary consumption
        solar_area = config['user_defined']['solar_area']  # [km^2]
        power_density_pv = config['user_defined'][
            'power_density_pv']  # PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2
        power_density_solar_thermal = config['user_defined'][
            'power_density_solar_thermal']  # Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2

        # Network
        loss_network = config['user_defined']['loss_network']
        c_grid_extra = config['user_defined'][
            'c_grid_extra']  # cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2
        import_capacity = config['user_defined']['import_capacity']  # [GW] Maximum power of electrical interconnections

        # Storage daily
        STORAGE_DAILY = config['user_defined']['STORAGE_DAILY']  # TODO automatise

        # Building SETS from data #
        SECTORS = list(eud_simple.columns)
        END_USES_INPUT = list(eud_simple.index)
        END_USES_CATEGORIES = list(end_uses_categories.loc[:, 'END_USES_CATEGORIES'].unique())
        RESOURCES = list(resources_simple.index)
        RE_RESOURCES = list(resources.loc[(resources['Category'] == 'Renewable'), :].index)
        EXPORT = list(resources.loc[resources['Category'] == 'Export', :].index)

        END_USES_TYPES_OF_CATEGORY = []
        for i in END_USES_CATEGORIES:
            li = list(end_uses_categories.loc[
                          end_uses_categories.loc[:, 'END_USES_CATEGORIES'] == i, 'END_USES_TYPES_OF_CATEGORY'])
            END_USES_TYPES_OF_CATEGORY.append(li)

        # TECHNOLOGIES_OF_END_USES_TYPE -> # METHOD 2 (uses layer_in_out to determine the END_USES_TYPE)
        END_USES_TYPES = list(end_uses_categories.loc[:, 'END_USES_TYPES_OF_CATEGORY'])

        ALL_TECHS = list(technologies_simple.index)

        layers_in_out_tech = layers_in_out.loc[~layers_in_out.index.isin(RESOURCES), :]
        TECHNOLOGIES_OF_END_USES_TYPE = []
        for i in END_USES_TYPES:
            li = list(layers_in_out_tech.loc[layers_in_out_tech.loc[:, i] == 1, :].index)
            TECHNOLOGIES_OF_END_USES_TYPE.append(li)

        # STORAGE and INFRASTRUCTURES
        ALL_TECH_OF_EUT = [item for sublist in TECHNOLOGIES_OF_END_USES_TYPE for item in sublist]

        STORAGE_TECH = list(storage_eff_in.index)
        INFRASTRUCTURE = [item for item in ALL_TECHS if item not in STORAGE_TECH and item not in ALL_TECH_OF_EUT]

        # STORAGE_OF_END_USES_TYPES ->  #METHOD 2 (using storage_eff_in)
        STORAGE_OF_END_USES_TYPES_ELEC = []

        for i in STORAGE_TECH:
            if storage_eff_in.loc[i, 'ELECTRICITY'] > 0:
                STORAGE_OF_END_USES_TYPES_ELEC.append(i)

        # Adding AMPL syntax #
        loss_network_df = pd.DataFrame(data=loss_network.values(), index=loss_network.keys(), columns=[' '])
        # Putting all the df in ampl syntax
        eud_simple = ampl_syntax(eud_simple, '')
        layers_in_out = ampl_syntax(layers_in_out, '')
        technologies_simple = ampl_syntax(technologies_simple, '')
        technologies_simple[technologies_simple > 1e+14] = 'Infinity'
        resources_simple = ampl_syntax(resources_simple, '')
        resources_simple[resources_simple > 1e+14] = 'Infinity'
        storage_eff_in = ampl_syntax(storage_eff_in, '')
        storage_eff_out = ampl_syntax(storage_eff_out, '')
        storage_characteristics = ampl_syntax(storage_characteristics, '')
        loss_network_df = ampl_syntax(loss_network_df, '')

        # Printing data #
        # printing signature of data file
        with open(out_path, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['# ------------------------------------------------------------------------------'
                             '-------------------------------------------	'])
            writer.writerow(['#	EnergyScope TD is an open-source energy model suitable for country scale analysis.'
                             ' It is a simplified representation of an urban or national energy system accounting for the'
                             ' energy flows'])
            writer.writerow(
                ['#	within its boundaries. Based on a hourly resolution, it optimises the design and operation '
                 'of the energy system while minimizing the cost of the system.'])
            writer.writerow(['#	'])
            writer.writerow(['#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), '
                             'Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
            writer.writerow(['#	'])
            writer.writerow(['#	'])
            writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
            writer.writerow(['#	you may not use this file except in compliance with the License.'])
            writer.writerow(['#	You may obtain a copy of the License at'])
            writer.writerow(['#	'])
            writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
            writer.writerow(['#	'])
            writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
            writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
            writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
            writer.writerow(['#	See the License for the specific language governing permissions and'])
            writer.writerow(['#	limitations under the License.'])
            writer.writerow(['#	'])
            writer.writerow(['#	Description and complete License: see LICENSE file.'])
            writer.writerow(
                ['# -------------------------------------------------------------------------------------------'
                 '------------------------------	'])
            writer.writerow(['	'])
            writer.writerow(['# UNIT MEASURES:'])
            writer.writerow(['# Unless otherwise specified units are:'])
            writer.writerow(
                [
                    '# Energy [GWh], Power [GW], Cost [Meuro], Time [h], Passenger transport [Mpkm], Freight Transport [Mtkm]'])
            writer.writerow(['	'])
            writer.writerow(['# References based on Supplementary material'])
            writer.writerow(['# --------------------------	'])
            writer.writerow(['# SETS not depending on TD	'])
            writer.writerow(['# --------------------------	'])
            writer.writerow(['	'])

        # printing sets
        print_set(SECTORS, 'SECTORS', out_path)
        print_set(END_USES_INPUT, 'END_USES_INPUT', out_path)
        print_set(END_USES_CATEGORIES, 'END_USES_CATEGORIES', out_path)
        print_set(RESOURCES, 'RESOURCES', out_path)
        print_set(RE_RESOURCES, 'RE_RESOURCES', out_path)
        print_set(EXPORT, 'EXPORT', out_path)

        newline(out_path)
        n = 0
        for j in END_USES_TYPES_OF_CATEGORY:
            print_set(j, 'END_USES_TYPES_OF_CATEGORY' + '["' + END_USES_CATEGORIES[n] + '"]', out_path)
            n += 1
        newline(out_path)
        n = 0
        for j in TECHNOLOGIES_OF_END_USES_TYPE:
            print_set(j, 'TECHNOLOGIES_OF_END_USES_TYPE' + '["' + END_USES_TYPES[n] + '"]', out_path)
            n += 1
        newline(out_path)
        print_set(STORAGE_TECH, 'STORAGE_TECH', out_path)
        print_set(INFRASTRUCTURE, 'INFRASTRUCTURE', out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# Storage subsets'])
        print_set(STORAGE_DAILY, 'STORAGE_DAILY', out_path)
        newline(out_path)
        print_set(STORAGE_OF_END_USES_TYPES_ELEC, 'STORAGE_OF_END_USES_TYPES ["ELECTRICITY"]', out_path)
        newline(out_path)

        # printing parameters
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# -----------------------------'])
            writer.writerow(['# PARAMETERS NOT DEPENDING ON THE NUMBER OF TYPICAL DAYS : '])
            writer.writerow(['# -----------------------------	'])
            writer.writerow([''])
            writer.writerow(['## PARAMETERS presented in Table 2.	'])
        # printing i_rate, re_share_primary,gwp_limit,solar_area
        print_param('i_rate', i_rate, 'part [2.7.4]', out_path)
        print_param('re_share_primary', re_share_primary, 'Minimum RE share in primary consumption', out_path)
        print_param('gwp_limit', gwp_limit, 'gwp_limit [ktCO2-eq./year]: maximum GWP emissions', out_path)
        print_param('solar_area', solar_area, '', out_path)
        print_param('power_density_pv', power_density_pv, 'PV : 1 kW/4.22m2   => 0.2367 kW/m2 => 0.2367 GW/km2',
                    out_path)
        print_param('power_density_solar_thermal', power_density_solar_thermal,
                    'Solar thermal : 1 kW/3.5m2 => 0.2857 kW/m2 => 0.2857 GW/km2', out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# Part [2.4]	'])

        # printing c_grid_extra and import_capacity
        print_param('c_grid_extra', c_grid_extra,
                    'cost to reinforce the grid due to intermittent renewable energy penetration. See 2.2.2', out_path)
        print_param('import_capacity', import_capacity, '', out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# end_Uses_year see part [2.1]'])
        print_df('param end_uses_demand_year : ', eud_simple, out_path)

        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# Link between layers  (data from Tables 19,21,22,23,25,29,30)'])
        print_df('param layers_in_out : ', layers_in_out, out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['# Technologies data from Tables (10,19,21,22,23,25,27,28,29,30) and part [2.2.1.1] for hydro'])
        print_df('param :', technologies_simple, out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# RESOURCES: part [2.5] (Table 26)'])
        print_df('param :', resources_simple, out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['# Storage inlet/outlet efficiency : part [2.6] (Table 28) and part [2.2.1.1] for hydro.	'])
        print_df('param storage_eff_in :', storage_eff_in, out_path)
        newline(out_path)
        print_df('param storage_eff_out :', storage_eff_out, out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# Storage characteristics : part [2.6] (Table 28) and part [2.2.1.1] for hydro.'])
        print_df('param :', storage_characteristics, out_path)
        newline(out_path)
        with open(out_path, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['# [A.6]'])
        print_df('param loss_network ', loss_network_df, out_path)

    #     return
    #
    #
    # # Function to print the ESTD_12TD.dat file from timeseries and STEP1 results #
    # def print_td_data(timeseries, out_path='STEP_2_Energy_Model', step1_out='STEP_1_TD_selection/TD_of_days.out',
    #                   nbr_td=12):
    if config['printing_td']:

        out_path = cs + config['case_study']  # config['ES_path']
        step1_out = config['step1_output']
        nbr_td = 12  # TODO add that as an argument

        logging.info('Printing ESTD_' + str(nbr_td) + 'TD.dat')

        # DICTIONARIES TO TRANSLATE NAMES INTO AMPL SYNTAX #
        # for EUD timeseries
        eud_params = {'Electricity (%_elec)': 'param electricity_time_series :'}
        # for resources timeseries that have only 1 tech linked to it
        res_params = {'PV': 'PV', 'Wind_onshore': 'WIND_ONSHORE', 'Wind_offshore': 'WIND_OFFSHORE',
                      'Hydro_river': 'HYDRO_RIVER'}
        # for parameters of costs of buying and selling and quantity selling
        cbuy_param = {'c_buy': 'ELECTRICITY'}
        csell_param = {'c_sell': 'ELECTRICITY'}
        qsell_param = {'q_sell': 'ELECTRICITY'}

        # Redefine the output file from the out_path given #
        out_path = out_path + '/ESTD_' + str(nbr_td) + 'TD.dat'

        # READING OUTPUT OF STEP1 #
        td_of_days = pd.read_csv(step1_out, names=['TD_of_days'])
        td_of_days['day'] = np.arange(1, 366, 1)  # putting the days of the year beside

        # COMPUTING NUMBER OF DAYS REPRESENTED BY EACH TD #
        sorted_td = td_of_days.groupby('TD_of_days').count()
        sorted_td.rename(columns={'day': '#days'}, inplace=True)
        sorted_td.reset_index(inplace=True)
        sorted_td.set_index(np.arange(1, nbr_td + 1), inplace=True)  # adding number of TD as index

        # BUILDING T_H_TD MATRICE #
        # generate T_H_TD
        td_and_hour_array = np.ones((24 * 365, 2))
        for i in range(365):
            td_and_hour_array[i * 24:(i + 1) * 24, 0] = np.arange(1, 25, 1)
            td_and_hour_array[i * 24:(i + 1) * 24, 1] = td_and_hour_array[i * 24:(i + 1) * 24, 1] * sorted_td[
                sorted_td['TD_of_days'] == td_of_days.loc[i, 'TD_of_days']].index.values
        t_h_td = pd.DataFrame(td_and_hour_array, index=np.arange(1, 8761, 1), columns=['H_of_D', 'TD_of_day'])
        t_h_td = t_h_td.astype('int64')
        # giving the right syntax
        t_h_td.reset_index(inplace=True)
        t_h_td.rename(columns={'index': 'H_of_Y'}, inplace=True)
        t_h_td['par_g'] = '('
        t_h_td['par_d'] = ')'
        t_h_td['comma1'] = ','
        t_h_td['comma2'] = ','
        # giving the right order to the columns
        t_h_td = t_h_td[['par_g', 'H_of_Y', 'comma1', 'H_of_D', 'comma2', 'TD_of_day', 'par_d']]

        # COMPUTING THE NORM OVER THE YEAR ##
        norm = time_series.sum(axis=0)
        norm.index.rename('Category', inplace=True)
        norm.name = 'Norm'

        # BUILDING TD TIMESERIES #
        # creating df with 2 columns : day of the year | hour in the day
        day_and_hour_array = np.ones((24 * 365, 2))
        for i in range(365):
            day_and_hour_array[i * 24:(i + 1) * 24, 0] = day_and_hour_array[i * 24:(i + 1) * 24, 0] * (i + 1)
            day_and_hour_array[i * 24:(i + 1) * 24, 1] = np.arange(1, 25, 1)
        day_and_hour = pd.DataFrame(day_and_hour_array, index=np.arange(1, 8761, 1), columns=['D_of_H', 'H_of_D'])
        day_and_hour = day_and_hour.astype('int64')
        time_series = time_series.merge(day_and_hour, left_index=True, right_index=True)

        # selecting time series of TD only
        td_ts = time_series[time_series['D_of_H'].isin(sorted_td['TD_of_days'])]

        # COMPUTING THE NORM_TD OVER THE YEAR FOR CORRECTION #
        # computing the sum of ts over each TD
        agg_td_ts = td_ts.groupby('D_of_H').sum()
        agg_td_ts.reset_index(inplace=True)
        agg_td_ts.set_index(np.arange(1, nbr_td + 1), inplace=True)
        agg_td_ts.drop(columns=['D_of_H', 'H_of_D'], inplace=True)
        # multiplicating each TD by the number of day it represents
        for c in agg_td_ts.columns:
            agg_td_ts[c] = agg_td_ts[c] * sorted_td['#days']
        # sum of new ts over the whole year
        norm_td = agg_td_ts.sum()

        # BUILDING THE DF WITH THE TS OF EACH TD FOR EACH CATEGORY #
        # pivoting TD_ts to obtain a (24,Nbr_TD*Nbr_ts*N_c)
        all_td_ts = td_ts.pivot(index='H_of_D', columns='D_of_H')

        # PRINTING #
        # printing description of file
        with open(out_path, mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

            # Comments and license
            td_writer.writerow([
                '# -------------------------------------------------------------------------------------------------------------------------	'])
            td_writer.writerow([
                '#	EnergyScope TD is an open-source energy model suitable for country scale analysis. It is a simplified representation of an urban or national energy system accounting for the energy flows'])
            td_writer.writerow([
                '#	within its boundaries. Based on a hourly resolution, it optimises the design and operation of the energy system while minimizing the cost of the system.'])
            td_writer.writerow(['#	'])
            td_writer.writerow([
                '#	Copyright (C) <2018-2019> <Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland and Université catholique de Louvain (UCLouvain), Belgium>'])
            td_writer.writerow(['#	'])
            td_writer.writerow(['#	'])
            td_writer.writerow(['#	Licensed under the Apache License, Version 2.0 (the "License");'])
            td_writer.writerow(['#	you may not use this file except in compliance with the License.'])
            td_writer.writerow(['#	You may obtain a copy of the License at'])
            td_writer.writerow(['#	'])
            td_writer.writerow(['#	http://www.apache.org/licenses/LICENSE-2.0'])
            td_writer.writerow(['#	'])
            td_writer.writerow(['#	Unless required by applicable law or agreed to in writing, software'])
            td_writer.writerow(['#	distributed under the License is distributed on an "AS IS" BASIS,'])
            td_writer.writerow(['#	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'])
            td_writer.writerow(['#	See the License for the specific language governing permissions and'])
            td_writer.writerow(['#	limitations under the License.'])
            td_writer.writerow(['#	'])
            td_writer.writerow(['#	Description and complete License: see LICENSE file.'])
            td_writer.writerow([
                '# -------------------------------------------------------------------------------------------------------------------------	'])
            td_writer.writerow(['	'])
            # peak_sh_factor
            td_writer.writerow(['# SETS depending on TD	'])
            td_writer.writerow(['# --------------------------	'])
            td_writer.writerow([';		'])
            td_writer.writerow(['		'])

            # printing T_H_TD param
            td_writer.writerow(['#SETS [Figure 3]		'])
            td_writer.writerow(['set T_H_TD := 		'])

        t_h_td.to_csv(out_path, sep='\t', header=False, index=False, mode='a', quoting=csv.QUOTE_NONE)

        # printing interlude
        with open(out_path, mode='a', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)

            td_writer.writerow([';'])
            td_writer.writerow([''])
            td_writer.writerow(['# -----------------------------'])
            td_writer.writerow(['# PARAMETERS DEPENDING ON NUMBER OF TYPICAL DAYS : '])
            td_writer.writerow(['# -----------------------------'])
            td_writer.writerow([''])

        # printing EUD timeseries param
        for k in eud_params.keys():
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            print_df(eud_params[k], ts, out_path)
            newline(out_path)

        # printing c_p_t param #
        with open(out_path, mode='a', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param c_p_t:='])
            # printing c_p_t part where 1 ts => 1 tech
        for k in res_params.keys():
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            s = '["' + res_params[k] + '",*,*]:'
            ts.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
            newline(out_path)

        out = cs + config['case_study']
        out_path_cbuy = out + '/ESTD_cbuy.txt'
        #printing param c_buy for ELECTRICITY ONLY
        with open(out_path_cbuy, mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            # td_writer.writerow([';'])
            td_writer.writerow(['param c_buy:='])
        for k in cbuy_param.keys():
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            s = '["' + cbuy_param[k] + '",*,*]:'
            ts.to_csv(out_path_cbuy, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
            newline(out_path_cbuy)

        out_path_csell = out + '/ESTD_csell.txt'
        # printing param c_sell for ELECTRICITY ONLY
        with open(out_path_csell, mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            # td_writer.writerow([';'])
            td_writer.writerow(['param c_sell:='])
        for k in csell_param.keys():
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            s = '["' + csell_param[k] + '",*,*]:'
            ts.to_csv(out_path_csell, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
            newline(out_path_csell)

        out_path_qsell = out + '/ESTD_qsell.txt'
        # printing param q_sell for ELECTRICITY ONLY
        with open(out_path_qsell, mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            # td_writer.writerow([';'])
            td_writer.writerow(['param q_sell:='])
        for k in qsell_param.keys():
            ts = all_td_ts[k]
            ts.columns = np.arange(1, nbr_td + 1)
            ts = ts * norm[k] / norm_td[k]
            ts.fillna(0, inplace=True)
            ts = ampl_syntax(ts, '')
            s = '["' + qsell_param[k] + '",*,*]:'
            ts.to_csv(out_path_qsell, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
            newline(out_path_qsell)

        # printing c_p_t part where 1 ts => more then 1 tech
        #for k in res_mult_params.keys():
        #    for j in res_mult_params[k]:
        #        ts = all_td_ts[k]
        #        ts.columns = np.arange(1, nbr_td + 1)
        #        ts = ts * norm[k] / norm_td[k]
        #        ts.fillna(0, inplace=True)
        #        ts = ampl_syntax(ts, '')
        #        s = '["' + j + '",*,*]:'
        #        ts.to_csv(out_path, sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)

    return

# def td_management(config) :
#     two_up = Path(__file__).parents[2]
#     cs = os.path.join(two_up, 'case_studies/')
#     out_path = cs + config['case_study']  # config['ES_path']
#     step1_out = config['step1_output']
#     nbr_td = 12
#     import_folders = config['data_dir']
#     time_series = pd.read_csv(import_folders / 'Time_series.csv', sep=',', header=0, index_col=0)
#
#     # READING OUTPUT OF STEP1 #
#     td_of_days = pd.read_csv(step1_out, names=['TD_of_days'])
#     td_of_days['day'] = np.arange(1, 366, 1)  # putting the days of the year beside
#
#     # COMPUTING NUMBER OF DAYS REPRESENTED BY EACH TD #
#     sorted_td = td_of_days.groupby('TD_of_days').count()
#     sorted_td.rename(columns={'day': '#days'}, inplace=True)
#     sorted_td.reset_index(inplace=True)
#     sorted_td.set_index(np.arange(1, nbr_td + 1), inplace=True)  # adding number of TD as index
#
#     # BUILDING T_H_TD MATRICE #
#     # generate T_H_TD
#     td_and_hour_array = np.ones((24 * 365, 2))
#     for i in range(365):
#         td_and_hour_array[i * 24:(i + 1) * 24, 0] = np.arange(1, 25, 1)
#         td_and_hour_array[i * 24:(i + 1) * 24, 1] = td_and_hour_array[i * 24:(i + 1) * 24, 1] * sorted_td[
#             sorted_td['TD_of_days'] == td_of_days.loc[i, 'TD_of_days']].index.values
#     t_h_td = pd.DataFrame(td_and_hour_array, index=np.arange(1, 8761, 1), columns=['H_of_D', 'TD_of_day'])
#     t_h_td = t_h_td.astype('int64')
#     # giving the right syntax
#     t_h_td.reset_index(inplace=True)
#     t_h_td.rename(columns={'index': 'H_of_Y'}, inplace=True)
#     t_h_td['par_g'] = '('
#     t_h_td['par_d'] = ')'
#     t_h_td['comma1'] = ','
#     t_h_td['comma2'] = ','
#     # giving the right order to the columns
#     t_h_td = t_h_td[['par_g', 'H_of_Y', 'comma1', 'H_of_D', 'comma2', 'TD_of_day', 'par_d']]
#
#     # COMPUTING THE NORM OVER THE YEAR ##
#     norm = time_series.sum(axis=0)
#     norm.index.rename('Category', inplace=True)
#     norm.name = 'Norm'
#
#     # BUILDING TD TIMESERIES #
#     # creating df with 2 columns : day of the year | hour in the day
#     day_and_hour_array = np.ones((24 * 365, 2))
#     for i in range(365):
#         day_and_hour_array[i * 24:(i + 1) * 24, 0] = day_and_hour_array[i * 24:(i + 1) * 24, 0] * (i + 1)
#         day_and_hour_array[i * 24:(i + 1) * 24, 1] = np.arange(1, 25, 1)
#     day_and_hour = pd.DataFrame(day_and_hour_array, index=np.arange(1, 8761, 1), columns=['D_of_H', 'H_of_D'])
#     day_and_hour = day_and_hour.astype('int64')
#     time_series = time_series.merge(day_and_hour, left_index=True, right_index=True)
#
#     # selecting time series of TD only
#     td_ts = time_series[time_series['D_of_H'].isin(sorted_td['TD_of_days'])]
#
#     # COMPUTING THE NORM_TD OVER THE YEAR FOR CORRECTION #
#     # computing the sum of ts over each TD
#     agg_td_ts = td_ts.groupby('D_of_H').sum()
#     agg_td_ts.reset_index(inplace=True)
#     agg_td_ts.set_index(np.arange(1, nbr_td + 1), inplace=True)
#     agg_td_ts.drop(columns=['D_of_H', 'H_of_D'], inplace=True)
#     # multiplicating each TD by the number of day it represents
#     for c in agg_td_ts.columns:
#         agg_td_ts[c] = agg_td_ts[c] * sorted_td['#days']
#     # sum of new ts over the whole year
#     norm_td = agg_td_ts.sum()
#
#     # BUILDING THE DF WITH THE TS OF EACH TD FOR EACH CATEGORY #
#     # pivoting TD_ts to obtain a (24,Nbr_TD*Nbr_ts*N_c)
#     all_td_ts = td_ts.pivot(index='H_of_D', columns='D_of_H')
#
#     return norm, norm_td, t_h_td, all_td_ts


# Function to run ES from python
def run_ES(config, case = 'deter'):
    two_up = Path(__file__).parents[2]

    if case == 'deter':
        cs = os.path.join(two_up,'case_studies')
        runner = 'ESTD_main_all_prints.run'
    else:
        cs = os.path.join(two_up,'case_studies',config['UQ_case'])
        runner = 'ESTD_main.run'
        #cs = cs + config['UQ_case'] + '/'

    # TODO make the case_study folder containing all runs with input, model and outputs
    shutil.copyfile(os.path.join(config['ES_path'], 'ESTD_model.mod'),
                    os.path.join(cs, config['case_study'],'ESTD_model.mod'))
    shutil.copyfile(os.path.join(config['ES_path'], runner),
                    os.path.join(cs, config['case_study'], runner))
    # creating output directory
    make_dir(os.path.join(cs,config['case_study'],'output'))
    make_dir(os.path.join(cs,config['case_study'],'output','hourly_data'))
    make_dir(os.path.join(cs,config['case_study'],'output','sankey'))
    os.chdir(os.path.join(cs,config['case_study']))
    # running ES
    logging.info('Running EnergyScope')

    if config['AMPL_path'] is None:
        #TODO add error message if ampl not found, check why doesn't print log in certain IDE
        #call('ampl '+run, shell=True)
        run(["ampl", runner])
    else:
        #TODO check about cplex call in .run if not in PATH
        run(config['AMPL_path']+'/ampl '+runner, shell=True)

    os.chdir(config['Working_directory'])

    logging.info('End of run')
    return


# Function to compute the annual average emission factors of each resource from the outputs #
def compute_gwp_op(import_folders, out_path='STEP_2_Energy_Model'):
    # import data and model outputs
    resources = pd.read_csv(import_folders[0] + '/Resources.csv', sep=';', index_col=2, header=2)
    yb = pd.read_csv(out_path + '/output/year_balance.txt', sep='\t', index_col=0)

    # clean df and get useful data
    yb.rename(columns=lambda x: x.strip(), inplace=True)
    yb.rename(index=lambda x: x.strip(), inplace=True)
    gwp_op_data = resources['gwp_op'].dropna()
    res_names = list(gwp_op_data.index)
    res_names_red = list(set(res_names) & set(list(yb.columns)))  # resources that are a layer
    yb2 = yb.drop(index='END_USES_DEMAND')
    tot_year = yb2.mul(yb2.gt(0)).sum()[res_names_red]

    # compute the actual resources used to produce each resource
    res_used = pd.DataFrame(0, columns=res_names_red, index=res_names)
    for r in res_names_red:
        yb_r = yb2.loc[yb2.loc[:, r] > 0, :]
        for i, j in yb_r.iterrows():
            if i in res_names:
                res_used.loc[i, r] = res_used.loc[i, r] + j[i]
            else:
                s = list(j[j < 0].index)[0]
                res_used.loc[s, r] = res_used.loc[s, r] - j[s]

    # differentiate the imported resources from the ones that are the mix between the imported ones and the produced ones
    gwp_op_imp = gwp_op_data.copy()
    gwp_op_imp.rename(index=lambda x: x + '_imp', inplace=True)
    gwp_op = pd.concat([gwp_op_data.copy(), gwp_op_imp])
    res_used_imp = pd.DataFrame(0, index=res_used.index, columns=res_used.columns)
    for i, j in res_used.iteritems():
        res_used_imp.loc[i, i] = j[i]
        res_used.loc[i, i] = 0
    res_used_imp.rename(index=lambda x: x + '_imp', inplace=True)
    all_res_used = pd.concat([res_used, res_used_imp])

    # compute the gwp_op of each mix through looping over the equations
    gwp_op_new = gwp_op.copy()
    conv = 100
    count = 0
    while conv > 1e-6:
        gwp_op = gwp_op_new
        gwp_op_new = pd.concat([(all_res_used.mul(gwp_op, axis=0).sum() / tot_year).fillna(0), gwp_op_imp])
        conv = (gwp_op_new - gwp_op).abs().sum()
        count += 1

    gwp_op_final = gwp_op_new[res_names_red]

    return gwp_op_final.combine_first(gwp_op_data)