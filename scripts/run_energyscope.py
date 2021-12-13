# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević
"""

import os
import pandas as pd
from pathlib import Path
import energyscope as es

if __name__ == '__main__':
   # define path
    path = Path(__file__).parents[1]
    user_data = path/'Data'/'User_data'
    developer_data = path/'Data'/'Developer_data'
    es_path = path/'energyscope'/'STEP_2_Energy_Model'
    step1_output = path/'energyscope'/'STEP_1_TD_selection'/'TD_of_days.out'
    # specify the configuration
    config = {'case_study': 'test', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
              'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
              'GWP_limit': 20000,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'import_capacity': 9.72,  # [GW] Electrical interconnections with neighbouring countries
              'data_folders': [user_data, developer_data],  # Folders containing the csv data files
              'ES_path': es_path,  # Path to the energy model (.mod and .run files)
              'step1_output': step1_output, # Output of the step 1 selection of typical days
              'all_data': dict(), # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
              'Working_directory': os.getcwd()}
    
   # Reading the data
    config['all_data'] = es.import_data(config['data_folders'])

    ##TODO Student work: Write the updates in data HERE
    # Example to change data: update wood availability to 23 400 GWh
    config['all_data']['Resources'].loc['WOOD', 'avail'] = 23400


    # Printing the .dat files for the optimisation problem
    es.print_data(config)

    # Running EnergyScope
    es.run_ES(config)

    # # Example to print the sankey from this script
    sankey_path = '../case_studies/' + config['case_study'] + '/output/sankey'
    es.drawSankey(path=sankey_path)

   # compute the actual average annual emission factors for each resource
    GWP_op = es.compute_gwp_op(config['data_folders'], path/'case_studies'/config['case_study'])
    GWP_op.to_csv( path/'case_studies'/config['case_study']/'output'/'GWP_op.txt', sep='\t')