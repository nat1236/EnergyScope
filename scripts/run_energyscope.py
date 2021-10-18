# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević
"""


import os
import pandas as pd
import energyscope as es


if __name__ == '__main__':

    # specify the configuration
    config = {'case_study': 'test', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'printing': True, # printing the data in ETSD_data.dat file for the optimisation problem
              'printing_td': True, # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
              'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'import_capacity': 4.5, # [GW] Electrical interconnections with neighbouring countries
              'data_folders':  ['../Data/User_data', '../Data/Developer_data'], # Folders containing the csv data files
              'ES_path': '../energyscope/STEP_2_Energy_Model', # Path to the energy model (.mod and .run files)
              'step1_output': '../energyscope/STEP_1_TD_selection/TD_of_days.out', # OUtput of the step 1 selection of typical days
              'all_data': dict(), # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
              'Working_directory': os.getcwd(),
              'AMPL_path': '/Users/xrixhon/Documents/Software/AMPL'} # PATH to AMPL licence (to adapt by the user)

    # Reading the data
    config['all_data'] = es.import_data(config['data_folders'])

    ##TODO Student work: Write the updates in data HERE
    # Example to change data: update wood availability to 23 400 GWh
    config['all_data']['Resources'].loc['WOOD','avail'] = 23400



    # Printing the .dat files for the optimisation problem
    es.print_data(config)
    # Running EnergyScope
    es.run_ES(config)

    # Example to print the sankey from this script
    sankey_path = '../case_studies/'+config['case_study']+'/output/sankey'
    es.drawSankey(path=sankey_path)
