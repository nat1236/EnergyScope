# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Xavier Rixhon, Gauthier Limpens
"""

import os
import pandas as pd
from pathlib import Path
import energyscope as es

if __name__ == '__main__':
    # First case run   #######################
    # define path
     path = Path(__file__).parents[1]
     data = path / 'Data' / '2040elec'
     es_path = path / 'energyscope' / 'STEP_2_Energy_Model'
     step1_output = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
     # specify the configuration
     config = {'case_study': '2040_elec_new', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
               'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
               'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
               'first_it' : True,  #First iteration of the for loop
               'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
               'data_dir': data,  # Folders containing the csv data files
               'ES_path': es_path,  # Path to the energy model (.mod and .run files)
               'step1_output': step1_output, # Output of the step 1 selection of typical days
               'all_data': dict(), # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
               'user_defined': dict(), # Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
               'Working_directory': os.getcwd(),
               'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'} # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables

    # Reading the data
     es.import_data(config)
     # Example to change data: update wood availability to 23 400 GWh (ref value here)
     #config['all_data']['Resources'].loc['WOOD', 'avail'] = 23400
     # Example to change share of public mobility into passenger mobility into 0.5 (ref value here)
     #config['user_defined']['share_mobility_public_max'] = 0.5

     config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 0
     config['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
     # Printing the .dat files for the optimisation problem
     es.print_data(config)
     # Running EnergyScope
     es.run_ES(config)
     es.scale_marginal_cost(config)
#

    # Example to print the sankey from this script
    #sankey_path = '../case_studies/' + config['case_study'] + '/output/sankey'
    #es.drawSankey(path=sankey_path)



    #Second case run #################################
    # path2 = Path(__file__).parents[1]
    # data2 = path2 / 'Data' / '2050elec'
    # es_path2 = path2 / 'energyscope' / 'STEP_2_Energy_Model'
    # step1_output2 = path2 / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
    # # specify the configuration
    # config2 = {'case_study': '2050_elec_new',
    #       # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
    #       'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
    #       'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
    #       'first_it' : True,
    #       'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
    #       'data_dir': data2,  # Folders containing the csv data files
    #       'ES_path': es_path2,  # Path to the energy model (.mod and .run files)
    #       'step1_output': step1_output2,  # Output of the step 1 selection of typical days
    #       'all_data': dict(),
    #       # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
    #       'user_defined': dict(),
    #       # Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
    #       'Working_directory': os.getcwd(),
    #       'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'}  # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables
    #
    # es.import_data(config2)
    # config2['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 0
    # config2['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    # es.print_data(config2)
    # es.run_ES(config2)
    # es.scale_marginal_cost(config2)

    #creates a dictionnary of outputs of one case study
     outputs = es.read_outputs('2040_elec_new')
    # #creates a dict of layer elec and plots it hourly
    #  dict_elec = es.read_layer('2040_elec_new','layer_ELECTRICITY')
    # es.hourly_plot(dict_elec)

    # marginal cost of config
     out = path/'case_studies'/config['case_study']
     cs = path/'case_studies'/config['case_study']/'output'
     mc = pd.read_csv(cs/'hourly_data'/'mc_scaled.txt', sep='\t', usecols = ['Hour', 'TD', 'ELECTRICITY'])
     pivot_mc = mc.pivot(index = 'Hour', columns = 'TD')
     #pivot_mc.to_csv(cs/'pivot_mc.csv')  #affiche le bon résultat

     # cbuy = pd.read_csv(out/'ESTD_cbuy.txt', sep='\t',)

    # marginal cost of config2
    #  cs2 = path2 / 'case_studies' / config2['case_study'] / 'output' / 'hourly_data'
    #  mc2 = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    #  pivot_mc2 = mc.pivot(index='Hour', columns='TD')

    # creating a dataframe for the iterations on c_buy and c_sell
    # !! refaire puisque c_buy et c_sell sont des tableaux
    #  data = {'c_buy': [c_buy], 'c_sell': [c_sell]}
    #  iter_df = pd.DataFrame(data)

    # for i in range(5):  #pour l'instant 5 itérations
    #  #just take layer ELECTRICITY et pas besoin d'itérer sur h et td
    #     # config['first_it'] = False
    #     # config2['first_it'] = False
    #     # config.c_buy = config2.mc_scaled
    #     # config.c_sell = config.mc_scaled
    #     # config.q_sell = config2.Q_buy
    #     # pt juste 12TD ? changer nom du case study ? avant ou après run ?
    #     es.print_data(config)
    #     es.print_data(config2)
    #
    #    # iter_df.append({'c_buy' : , 'c_sell' : })
    #
    #     es.run_ES(config)
    #     es.run_ES(config2)
    #     es.scale_marginal_cost(config)
#         es.scale_marginal_cost(config2)
    #     i = i+1
