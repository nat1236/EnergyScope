# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Xavier Rixhon, Gauthier Limpens
"""

import os
import pandas as pd
import csv
from pathlib import Path
import energyscope as es
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # First case run   #######################
    # define path
    path = Path(__file__).parents[1]
    data = path / 'Data' / '2040elec'
    es_path = path / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
     # specify the configuration
    config = {'case_study': 'same_PV_first', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
               'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
               'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisation problem
               'printing_params' : True,
               # 'first_it' : True,  #First iteration of the for loop
               'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
               'data_dir': data,  # Folders containing the csv data files
               'ES_path': es_path,  # Path to the energy model (.mod and .run files)
               'step1_output': step1_output, # Output of the step 1 selection of typical days
               'all_data': dict(), # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
               'user_defined': dict(), # Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
               'Working_directory': os.getcwd(),
               'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'} # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables

    es.import_data(config)
     # Example to change data: update wood availability to 23 400 GWh (ref value here)
     #config['all_data']['Resources'].loc['WOOD', 'avail'] = 23400
     # Example to change share of public mobility into passenger mobility into 0.5 (ref value here)
     #config['user_defined']['share_mobility_public_max'] = 0.5

    config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 0
    config['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    config['all_data']['Technologies'].loc['CCGT', 'f_max'] = 0
    config['all_data']['Technologies'].loc['CCGT_AMMONIA', 'f_max'] = 0
    config['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = 0
    config['all_data']['Technologies'].loc['WIND_OFFSHORE', 'f_max'] = 0
    config['all_data']['Technologies'].loc['HYDRO_RIVER', 'f_max'] = 0
    # config['all_data']['Technologies'].loc['EFFICIENCY', 'f_max'] = 0
    # config['all_data']['Technologies'].loc['GRID', 'f_max'] = 0
    config['all_data']['Technologies'].loc['BATT_LI', 'f_max'] = 0
    config['all_data']['Technologies'].loc['PHS', 'f_min'] = 0
    config['all_data']['Technologies'].loc['PHS', 'f_max'] = 0
    es.print_data(config)
    es.run_ES(config)
    es.scale_marginal_cost(config)
    outputs = es.read_outputs(config['case_study'], hourly_data=True) #layers = ['layer_ELECTRICITY'])

    thtd = es.generate_t_h_td(config)
    to_use = thtd['t_h_td']

    # mc_scaled = es.read_layer(config['case_study'], 'mc_scaled')
    # yearly_mc = es.from_td_to_year(mc_scaled, to_use)
    # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
    # yearly_mc.plot(title='Yearly marginal cost')

    # dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    # dict_elec = dict_elec.drop(columns=['Q_buy', 'q_sell'])
    # yearly = es.from_td_to_year(dict_elec, to_use)
    # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
    # yearly.plot()

    # Example to print the sankey from this script
    #sankey_path = '../case_studies/' + config['case_study'] + '/output/sankey'
    #es.drawSankey(path=sankey_path)


    #Second case run #################################
    path2 = Path(__file__).parents[1]
    data2 = path2 / 'Data' / '2050elec'
    es_path2 = path2 / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output2 = path2 / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
    # specify the configuration
    config2 = {'case_study': 'same_CCGT_first',
          # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
          'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
          'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
          'printing_params': True,
          # 'first_it' : True,
          'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
          'data_dir': data2,  # Folders containing the csv data files
          'ES_path': es_path2,  # Path to the energy model (.mod and .run files)
          'step1_output': step1_output2,  # Output of the step 1 selection of typical days
          'all_data': dict(),# Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
          'user_defined': dict(),# Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
          'Working_directory': os.getcwd(),
          'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'}  # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables

    es.import_data(config2)
    config2['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 0
    config2['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    config2['all_data']['Technologies'].loc['PV', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['CCGT_AMMONIA', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['WIND_OFFSHORE', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['HYDRO_RIVER', 'f_max'] = 0
    # config2['all_data']['Technologies'].loc['EFFICIENCY', 'f_max'] = 0
    # config2['all_data']['Technologies'].loc['GRID', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['BATT_LI', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['PHS', 'f_min'] = 0
    config2['all_data']['Technologies'].loc['PHS', 'f_max'] = 0
    es.print_data(config2)
    es.run_ES(config2)
    es.scale_marginal_cost(config2)
    outputs2 = es.read_outputs(config2['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

    # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
    # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
    # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
    # yearly_mc2.plot(title='Yearly marginal cost')

    # dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    # dict_elec2 = dict_elec2.drop(columns=['Q_buy', 'q_sell'])
    # yearly2 = es.from_td_to_year(dict_elec2, to_use)
    # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 1.0]
    # yearly2.plot()

    cs = path / 'case_studies' / config['case_study'] / 'output' / 'hourly_data'
    path_dat = path / 'case_studies' / config['case_study']
    cs2 = path2 / 'case_studies' / config2['case_study'] / 'output' / 'hourly_data'
    path_dat2 = path / 'case_studies' / config2['case_study']


    for i in range(12):  #pour l'instant 5 itérations

        mc = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
        pivot_mc = mc.pivot(index='Hour', columns='TD', values='ELECTRICITY')
        Qbuy = outputs['var_Q_buy'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        cbuy = pd.read_csv(path_dat / 'ESTD_cbuy.dat', sep='\t', skiprows=1, index_col=0)
        csell = pd.read_csv(path_dat / 'ESTD_csell.dat', sep='\t', skiprows=1, index_col=0)
        qsell = pd.read_csv(path_dat / 'ESTD_qsell.dat', sep='\t', skiprows=1, index_col=0)

        mc2 = pd.read_csv(cs2 / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
        pivot_mc2 = mc2.pivot(index='Hour', columns='TD', values='ELECTRICITY')
        cbuy2 = pd.read_csv(path_dat2 / 'ESTD_cbuy.dat', sep='\t', skiprows=1, index_col=0)
        csell2 = pd.read_csv(path_dat2 / 'ESTD_csell.dat', sep='\t', skiprows=1, index_col=0)
        qsell2 = pd.read_csv(path_dat2 / 'ESTD_qsell.dat', sep='\t', skiprows=1, index_col=0)
        Qbuy2 = outputs2['var_Q_buy'].pivot(index='Hour', columns='TD', values='ELECTRICITY')

        config['case_study'] = 'same_PV_' + str(i)
        es.print_data(config)
        cs = path / 'case_studies' / config['case_study'] / 'output' / 'hourly_data'
        path_dat = path / 'case_studies' / config['case_study']

        config2['case_study'] = 'same_CCGT_' + str(i)
        # config2['printing_params'] = False
        es.print_data(config2)
        cs2 = path2 / 'case_studies' / config2['case_study'] / 'output' / 'hourly_data'
        path_dat2 = path / 'case_studies' / config2['case_study']

        # EXCHANGE PART
        # #Exchanging cbuy and writing it in the good file to use it afterwards
        cbuy = pivot_mc2
        with open(path_dat / 'ESTD_cbuy.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param c_buy:='])
        cbuy = es.ampl_syntax(cbuy, ' ')
        s = '["' + 'ELECTRICITY' + '",*,*]:'
        cbuy.to_csv(path_dat / 'ESTD_cbuy.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        # #Doing the same for csell
        csell = pivot_mc   #de base, juste pivot_mc
        with open(path_dat / 'ESTD_csell.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param c_sell:='])
        csell = es.ampl_syntax(csell, ' ')
        csell.to_csv(path_dat / 'ESTD_csell.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        # #Doing the same for qsell
        qsell = Qbuy2
        with open(path_dat / 'ESTD_qsell.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param q_sell:='])
        qsell = es.ampl_syntax(qsell, ' ')
        qsell.to_csv(path_dat / 'ESTD_qsell.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        # #Doing the same for cbuy2
        cbuy2 = pivot_mc
        with open(path_dat2 / 'ESTD_cbuy.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param c_buy:='])
        cbuy2 = es.ampl_syntax(cbuy2, ' ')
        cbuy2.to_csv(path_dat2 / 'ESTD_cbuy.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        # #Doing the same for csell2
        csell2 = pivot_mc2  #de base, juste pivot_mc2
        with open(path_dat2 / 'ESTD_csell.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param c_sell:='])
        csell2 = es.ampl_syntax(csell2, ' ')
        csell2.to_csv(path_dat2 / 'ESTD_csell.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)
        # #Doing the same for qsell2
        qsell2 = Qbuy
        with open(path_dat2 / 'ESTD_qsell.dat', mode='w', newline='') as td_file:
            td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            td_writer.writerow(['param q_sell:='])
        qsell2 = es.ampl_syntax(qsell2, ' ')
        qsell2.to_csv(path_dat2 / 'ESTD_qsell.dat', sep='\t', mode='a', header=True, index=True, index_label=s, quoting=csv.QUOTE_NONE)

        # #Re-running on both cases

        es.run_ES(config)
        es.run_ES(config2)
        es.scale_marginal_cost(config)
        es.scale_marginal_cost(config2)
        outputs = es.read_outputs(config['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])
        outputs2 = es.read_outputs(config2['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

        #Plots
        # mc_scaled = es.read_layer(config['case_study'], 'mc_scaled')
        # yearly_mc = es.from_td_to_year(mc_scaled, to_use)
        # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
        # yearly_mc.plot(title='Yearly marginal cost')

        # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
        # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
        # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
        # yearly_mc2.plot(title='Yearly marginal cost')

        # dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
        # dict_elec = dict_elec.drop(columns=['Q_buy', 'q_sell'])
        # yearly = es.from_td_to_year(dict_elec, to_use)
        # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
        # yearly.plot()

        # dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
        # dict_elec2 = dict_elec2.drop(columns=['Q_buy', 'q_sell'])
        # yearly2 = es.from_td_to_year(dict_elec2, to_use)
        # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 1.0]
        # yearly2.plot()

        i = i+1


    #Plots
    dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    dict_elec = dict_elec.drop(columns=['Q_buy','q_sell'])
    # es.hourly_plot(dict_elec)

    dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    dict_elec2 = dict_elec2.drop(columns=['Q_buy','q_sell'])
    # es.hourly_plot(dict_elec2)

    # mc_scaled = es.read_layer(config['case_study'],'mc_scaled')
    # yearly_mc = es.from_td_to_year(mc_scaled,to_use)
    # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
    # yearly_mc.plot(title='Yearly marginal cost')

    # yearly = es.from_td_to_year(dict_elec, to_use)
    # yearly = yearly.loc[:,yearly.sum().abs() > 1.0]
    # # yearly = yearly.drop(columns=['Q_buy','q_sell']) #loc[:,['PV','Q_buy-q_sell','END_USE']]
    # yearly.plot()

    # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
    # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
    # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
    # yearly_mc2.plot(title='Yearly marginal cost')

    # yearly2 = es.from_td_to_year(dict_elec2, to_use)
    # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 1.0]
    # # yearly2 = yearly2.drop(['Q_buy','q_sell'], axis=1)
    # yearly2.plot()

    # pd.plotting.lag_plot(yearly)


    # pd.plotting.autocorrelation_plot(yearly_mc)

    # plt.figure()
    # heat_map = sns.heatmap(cbuy, linewidth=1, annot=True)
    # plt.title("HeatMap using Seaborn Method")
    # plt.show()