 # -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Xavier Rixhon, Gauthier Limpens
"""

import os
import pandas as pd
import numpy as np
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
    config = {'case_study': 'changed_PV_first', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
               'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
               'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisation problem
               'printing_params' : True,
               'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
               'data_dir': data,  # Folders containing the csv data files
               'ES_path': es_path,  # Path to the energy model (.mod and .run files)
               'step1_output': step1_output, # Output of the step 1 selection of typical days
               'all_data': dict(), # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
               'user_defined': dict(), # Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
               'Working_directory': os.getcwd(),
               'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'} # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables

    es.import_data(config)
     # Example to change share of public mobility into passenger mobility into 0.5 (ref value here)
     #config['user_defined']['share_mobility_public_max'] = 0.5

    config['all_data']['Resources'].loc['ELECTRICITY', 'c_op'] = 5
    config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 1000000000000  # 0
    config['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    config['all_data']['Technologies'].loc['PV', 'f_max'] = 100000000000000
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
    outputs = es.read_outputs(config['case_study'], hourly_data=True)

    cs = path / 'case_studies' / config['case_study'] / 'output' / 'hourly_data'
    path_dat = path / 'case_studies' / config['case_study']

    mc = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    pivot_mc = mc.pivot(index='Hour', columns='TD', values='ELECTRICITY')
    pivot_mc = es.check_mc(pivot_mc)
    Qimp = outputs['var_Q_imp'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
    qexp = pd.read_csv(path_dat / 'ESTD_qexp.dat', sep='\t', skiprows=1, index_col=0)

    EUD = outputs['var_END_USES'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
    F_PV = outputs['F_PV']
    # alpha = outputs['param_alpha'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
    cpt_PV = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8822, nrows=24)
    #
    # Sto_in_PHS = pd.read_csv(cs / 'layer_ELECTRICITY.txt',sep='\t', usecols=[' Time', 'Td ', 'PHS_Pin'])
    # Sto_in_PHS = Sto_in_PHS.pivot(index=' Time', columns='Td ', values='PHS_Pin')
    # Sto_in_BATT = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pin'])
    # Sto_in_BATT = Sto_in_BATT.pivot(index=' Time', columns='Td ', values='BATT_LI_Pin')
    # Sto_out_PHS = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'PHS_Pout'])
    # Sto_out_PHS = Sto_out_PHS.pivot(index=' Time', columns='Td ', values='PHS_Pout')
    # Sto_out_BATT = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pout'])
    # Sto_out_BATT = Sto_out_BATT.pivot(index=' Time', columns='Td ', values='BATT_LI_Pout')
    #
    alpha = cpt_PV * F_PV.iloc[0, 0] - EUD.values #- qexp.values + Qimp.values #+ Sto_out_PHS.values + Sto_out_BATT.values + Sto_in_PHS.values + Sto_in_BATT.values
    alpha = es.check_alpha(alpha)

    # config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 100000000000

    thtd = es.generate_t_h_td(config)
    to_use = thtd['t_h_td']

    #computing lcoe
    # lcoe_pv = es.compute_lcoe(config)

    # #2 different plots
    # mc_scaled = es.read_layer(config['case_study'], 'mc_scaled')
    # yearly_mc = es.from_td_to_year(mc_scaled, to_use)
    # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
    # yearly_mc.plot(title='Yearly marginal cost')

    dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    dict_elec = dict_elec.loc[:, ['PV', 'Q_imp', 'END_USE']]
    # es.hourly_plot(dict_elec)
    # yearly = es.from_td_to_year(dict_elec, to_use)
    # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
    # yearly = yearly.loc[:, ['PV', 'Q_imp', 'END_USE']]
    #
    # firstExch = es.hourly_plot(dict_elec) #dict_elec.plot(title='Layer ELEC')


    xticks = np.arange(0, dict_elec.shape[0] + 1, 8)
    fig, ax = plt.subplots(figsize=(13,7))
    dict_elec.plot(kind='bar', stacked = True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap='tab20')
    ax.set_title('Layer elec - first exchange')
    ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel('Hour')
    ax.set_ylabel('Power [GW]')
    fig.tight_layout()
    fig.show()

    # Example to print the sankey from this script
    #sankey_path = '../case_studies/' + config['case_study'] + '/output/sankey'
    #es.drawSankey(path=sankey_path)


    #Second case run #################################
    data2 = path / 'Data' / '2050elec'
    es_path2 = path / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output2 = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
    # specify the configuration
    config2 = {'case_study': 'changed_CCGT_0',# Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
          'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
          'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
          'printing_params': True,
          'GWP_limit': 1e+7,  # [ktCO2-eq./year]	# Minimum GWP reduction
          'data_dir': data2,  # Folders containing the csv data files
          'ES_path': es_path2,  # Path to the energy model (.mod and .run files)
          'step1_output': step1_output2,  # Output of the step 1 selection of typical days
          'all_data': dict(),# Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
          'user_defined': dict(),# Dictionnary with user_defined parameters from user_defined.json, see definition into Data/user_defined_doc.json
          'Working_directory': os.getcwd(),
          'AMPL_path': '/home/natacha/Downloads/ampl_linux-intel64'}  # PATH to AMPL licence (to adapt by the user), set to None if AMPL is in your PATH variables

    es.import_data(config2)
    config2['all_data']['Resources'].loc['ELECTRICITY', 'c_op'] = 5 #config['all_data']['Resources'].loc['ELECTRICITY', 'c_op']
    config2['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 100000000000
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

    cs2 = path / 'case_studies' / config2['case_study'] / 'output' / 'hourly_data'
    path_dat2 = path / 'case_studies' / config2['case_study']
    s = '["' + 'ELECTRICITY' + '",*,*]:'

    cimp2 = pivot_mc
    with open(path_dat2 / 'ESTD_cimp.dat', mode='w', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        td_writer.writerow(['param c_imp:='])
    cimp2 = es.ampl_syntax(cimp2, ' ')
    cimp2.to_csv(path_dat2 / 'ESTD_cimp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

    qexp2 = Qimp
    with open(path_dat2 / 'ESTD_qexp.dat', mode='w', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        td_writer.writerow(['param q_exp:='])
    qexp2 = es.ampl_syntax(qexp2, ' ')
    qexp2.to_csv(path_dat2 / 'ESTD_qexp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

    alpha2 = alpha
    with open(path_dat2 / 'ESTD_alpha.dat', mode='w', newline='') as td_file:
        td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        td_writer.writerow(['param alpha:='])
    # alpha2 = es.ampl_syntax(alpha2, ' ')
    alpha2.to_csv(path_dat2 / 'ESTD_alpha.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

    es.run_ES(config2)
    es.scale_marginal_cost(config2)
    outputs2 = es.read_outputs(config2['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

    cpt_CCGT = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8822, nrows=24)
    cpt_CCGT.iloc[:] = 1

    # #2 different plots
    # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
    # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
    # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
    # yearly_mc2.plot(title='Yearly marginal cost')

    dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    dict_elec2 = dict_elec2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
    dict_elec2 = dict_elec2.rename(columns={"Q_imp":"Q_imp_from_PV", "q_exp":"q_exp_from_PV", "END_USE":"END_USE_CCGT"})

    both = pd.merge(dict_elec,dict_elec2,right_index =True, left_index=True)
    both.plot(kind='bar',stacked=True,position=0,width=1.0,legend=True,colormap='tab20',xticks=xticks)
    # dict_elec2 = dict_elec2.drop(columns='alpha')
    # es.hourly_plot(dict_elec2)
    # yearly2 = es.from_td_to_year(dict_elec2, to_use)
    # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 1.0]
    # yearly2 = yearly2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
    # dict_elec2.plot(kind='bar', stacked = True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap='tab20')

    # xticks2 = np.arange(0, dict_elec2.shape[0] + 1, 8)
    # fig2, ax2 = plt.subplots(figsize=(13, 7))
    # dict_elec2.plot(kind='line', ax=ax2, legend=True, xticks=xticks2, colormap='tab20')
    # ax2.set_title('Layer elec - second exchange')
    # ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax2.set_xlabel('Hour')
    # ax2.set_ylabel('Power [GW]')
    # fig2.tight_layout()
    # fig2.show()

    # creating data frames to store the mean of marginal costs
    # mc_debut = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    # marginal_cost = mc_debut.pivot(index='Hour', columns='TD', values='ELECTRICITY')
    # marginal_cost.iloc[:] = 0

    # mc2_debut = pd.read_csv(cs2 / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    # marginal_cost2 = mc2_debut.pivot(index='Hour', columns='TD', values='ELECTRICITY')
    # marginal_cost2.iloc[:] = 0

    for i in range(1,3):  #pour l'instant 5 itérations

        mc = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
        pivot_mc = mc.pivot(index='Hour', columns='TD', values='ELECTRICITY')
        pivot_mc = es.check_mc(pivot_mc)
        cimp = pd.read_csv(path_dat / 'ESTD_cimp.dat', sep='\t', skiprows=1, index_col=0)
        Qimp = outputs['var_Q_imp'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        qexp = pd.read_csv(path_dat / 'ESTD_qexp.dat', sep='\t', skiprows=1, index_col=0)

        mc2 = pd.read_csv(cs2 / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
        pivot_mc2 = mc2.pivot(index='Hour', columns='TD', values='ELECTRICITY')
        pivot_mc2 = es.check_mc(pivot_mc2)
        cimp2 = pd.read_csv(path_dat2 / 'ESTD_cimp.dat', sep='\t', skiprows=1, index_col=0)
        Qimp2 = outputs2['var_Q_imp'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        qexp2 = pd.read_csv(path_dat2 / 'ESTD_qexp.dat', sep='\t', skiprows=1, index_col=0)

        EUD = outputs['var_END_USES'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        F_PV = outputs['F_PV']
        # alpha = outputs['param_alpha'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        cpt_PV = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8822, nrows=24)
        # Sto_in_PHS = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'PHS_Pin'])
        # Sto_in_PHS = Sto_in_PHS.pivot(index=' Time', columns='Td ', values='PHS_Pin')
        # Sto_in_PHS = es.check_nan(Sto_in_PHS)
        # Sto_in_BATT = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pin'])
        # Sto_in_BATT = Sto_in_BATT.pivot(index=' Time', columns='Td ', values='BATT_LI_Pin')
        # Sto_in_BATT = es.check_nan(Sto_in_BATT)
        # Sto_out_PHS = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'PHS_Pout'])
        # Sto_out_PHS = Sto_out_PHS.pivot(index=' Time', columns='Td ', values='PHS_Pout')
        # Sto_out_PHS = es.check_nan(Sto_out_PHS)
        # Sto_out_BATT = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pout'])
        # Sto_out_BATT = Sto_out_BATT.pivot(index=' Time', columns='Td ', values='BATT_LI_Pout')
        # Sto_out_BATT = es.check_nan(Sto_out_BATT)
        #
        EUD2 = outputs2['var_END_USES'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        F_CCGT = outputs2['F_CCGT']
        # Sto_in_PHS2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'PHS_Pin'])
        # Sto_in_PHS2 = Sto_in_PHS2.pivot(index=' Time', columns='Td ', values='PHS_Pin')
        # Sto_in_PHS2 = es.check_nan(Sto_in_PHS2)
        # Sto_in_BATT2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pin'])
        # Sto_in_BATT2 = Sto_in_BATT2.pivot(index=' Time', columns='Td ', values='BATT_LI_Pin')
        # Sto_in_BATT2 = es.check_nan(Sto_in_BATT2)
        # Sto_out_PHS2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'PHS_Pout'])
        # Sto_out_PHS2 = Sto_out_PHS2.pivot(index=' Time', columns='Td ', values='PHS_Pout')
        # Sto_out_PHS2 = es.check_nan(Sto_out_PHS2)
        # Sto_out_BATT2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'BATT_LI_Pout'])
        # Sto_out_BATT2 = Sto_out_BATT2.pivot(index=' Time', columns='Td ', values='BATT_LI_Pout')
        # Sto_out_BATT2 = es.check_nan(Sto_out_BATT2)
        # # alpha2 = outputs2['param_alpha'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        #
        alpha = cpt_PV * F_PV.iloc[0, 0] - EUD.values #- qexp.values + Qimp.values #+ Sto_out_PHS.values + Sto_out_BATT.values + Sto_in_PHS.values + Sto_in_BATT.values
        alpha = es.check_alpha(alpha)

        alpha2 = cpt_CCGT * F_CCGT.iloc[0, 0] - EUD2.values #- qexp2.values + Qimp2.values #+ Sto_out_PHS2.values + Sto_out_BATT2.values + Sto_in_PHS2.values + Sto_in_BATT2.values
        alpha2 = es.check_alpha(alpha2)

        s = '["' + 'ELECTRICITY' + '",*,*]:'

        if i == 0 or i % 2 == 0 :  #si itération paire
            config2['case_study'] = 'changed_CCGT_' + str(i)
            es.print_data(config2)
            path_dat2 = path / 'case_studies' / config2['case_study']
            cs2 = path_dat2 / 'output' / 'hourly_data'

            cimp2 = pivot_mc #es.which_cost(Qimp2, pivot_mc2,pivot_mc)  #  #marginal_cost
            with open(path_dat2 / 'ESTD_cimp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param c_imp:='])
            cimp2 = es.ampl_syntax(cimp2, ' ')
            cimp2.to_csv(path_dat2 / 'ESTD_cimp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            qexp2 = Qimp
            with open(path_dat2 / 'ESTD_qexp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param q_exp:='])
            qexp2 = es.ampl_syntax(qexp2, ' ')
            qexp2.to_csv(path_dat2 / 'ESTD_qexp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            alpha2 = alpha
            with open(path_dat2 / 'ESTD_alpha.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param alpha:='])
            # alpha2 = es.ampl_syntax(alpha2, ' ')
            alpha2.to_csv(path_dat2 / 'ESTD_alpha.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            es.run_ES(config2)
            es.scale_marginal_cost(config2)
            outputs2 = es.read_outputs(config2['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

            dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
            dict_elec2 = dict_elec2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
            dict_elec2 = dict_elec2.rename(columns={"Q_imp": "Q_imp_from_PV", "q_exp": "q_exp_from_PV", "END_USE": "END_USE_CCGT"})

            both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
            both.plot(kind='bar', stacked=True, position=0, width=1.0, legend=True, colormap='tab20', xticks=xticks)
            # es.hourly_plot(dict_elec2)
            # yearly2 = es.from_td_to_year(dict_elec2, to_use)
            # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 0.0]
            # yearly2 = yearly2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
            # # yearly2 = yearly2.loc[:, ['CCGT', 'Q_exch', 'END_USE']]
            # yearly2.plot(title='Yearly layer ELEC (>0)')
            # es.hourly_plot(dict_elec2)

        else :
            config['case_study'] = 'changed_PV_' + str(i)
            es.print_data(config)
            path_dat = path / 'case_studies' / config['case_study']
            cs = path_dat / 'output' / 'hourly_data'

            cimp = pivot_mc2 # es.which_cost(Qimp,pivot_mc, pivot_mc2)  #pivot_mc2
            with open(path_dat / 'ESTD_cimp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param c_imp:='])
            cimp = es.ampl_syntax(cimp, ' ')
            cimp.to_csv(path_dat / 'ESTD_cimp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            qexp = Qimp2
            with open(path_dat / 'ESTD_qexp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param q_exp:='])
            qexp = es.ampl_syntax(qexp, ' ')
            qexp.to_csv(path_dat / 'ESTD_qexp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            alpha = alpha2
            with open(path_dat / 'ESTD_alpha.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param alpha:='])
            # alpha = es.ampl_syntax(alpha, ' ')
            alpha.to_csv(path_dat / 'ESTD_alpha.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            es.run_ES(config)
            es.scale_marginal_cost(config)
            outputs = es.read_outputs(config['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

            dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
            dict_elec = dict_elec.loc[:, ['ELECTRICITY', 'PV', 'Q_imp', 'q_exp', 'END_USE']]

            dict_elec = dict_elec.rename(columns={"Q_imp": "Q_imp_from_CCGT", "q_exp": "q_exp_from_CCGT", "END_USE": "END_USE_PV"})

            both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
            both.plot(kind='bar', stacked=True, position=0, width=1.0, legend=True, colormap='tab20', xticks=xticks)
            # es.hourly_plot(dict_elec)
            # yearly = es.from_td_to_year(dict_elec, to_use)
            # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
            # yearly = yearly.loc[:,['ELECTRICITY','PV','Q_imp','q_exp','END_USE']]
            # yearly = yearly.loc[:, ['PV','Q_exch','END_USE']]
            # yearly.plot(title='Yearly layer ELEC')
            # dict_elec2.plot(ax=ax2)

        #Plots
        # mc_scaled = es.read_layer(config['case_study'], 'mc_scaled')
        # yearly_mc = es.from_td_to_year(mc_scaled, to_use)
        # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
        # yearly_mc.plot(title='Yearly marginal cost')
        #
        # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
        # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
        # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
        # yearly_mc2.plot(title='Yearly marginal cost')

        i = i+1


    #Plots
    # dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    # # dict_elecbis = dict_elec.drop(columns=['Q_buy','q_sell'])
    # es.hourly_plot(dict_elec)

    # dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    # # dict_elec2 = dict_elec2.drop(columns=['Q_buy','q_sell'])
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

    # x = ['first',1,3,5,7,9,11,13]
    # y = [173.685496,177.878811,178.031713,178.038324,181.408444,181.574659,181.575528,184.748782]
    # plt.figure()
    # plt.plot(x, y, linestyle='solid')
    # plt.title('Evolution of installed quantity of PV')
    # plt.xlabel('Iterations')
    # plt.ylabel('Power [GW]')
    # plt.show()

    # x = [0,2,4,6,8,10,12,14]
    # y = [32.402582,32.402577,32.402577,32.402577,32.402577,32.402577,32.402577,32.402577]
    # plt.figure()
    # plt.plot(x, y, linestyle='solid')
    # plt.title('Evolution of installed quantity of CCGT')
    # plt.xlabel('Iterations')
    # plt.ylabel('Power [GW]')
    # plt.show()