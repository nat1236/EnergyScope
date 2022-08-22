 # -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Xavier Rixhon, Gauthier Limpens
"""

import os
import re
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import energyscope as es
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcmp
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)

if __name__ == '__main__':
    # First case run   #######################
    #dictionary for the colors
    dictColor = {'curtailment':'gold','PV': 'yellow','CCGT':'goldenrod', 'END_USE':'lightblue', 'Q_imp':'rebeccapurple','q_exp':'mediumblue','q_exp_from_PV':'mediumblue',
                 'Outside_elec':'deepskyblue','ELECTRICITY': 'deepskyblue', 'WIND_ONSHORE':'yellowgreen','WIND_OFFSHORE':'forestgreen','HYDRO_RIVER':'skyblue',
                 'CCGT_AMMONIA':'darkgoldenrod'}

    # define path
    path = Path(__file__).parents[1]
    data = path / 'Data' / '2040elec'
    es_path = path / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
     # specify the configuration
    config = {'case_study': 'new_PV_first', # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
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

    config['all_data']['Resources'].loc['ELECTRICITY', 'c_op'] = 0.5
    config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 1000000000000  # 0
    config['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    config['all_data']['Technologies'].loc['PV', 'f_max'] = 100000000000000
    config['all_data']['Technologies'].loc['CCGT', 'f_max'] = 0
    config['all_data']['Technologies'].loc['CCGT_AMMONIA', 'f_max'] = 0
    config['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = 0
    config['all_data']['Technologies'].loc['WIND_OFFSHORE', 'f_max'] = 0
    config['all_data']['Technologies'].loc['HYDRO_RIVER', 'f_max'] = 0
    config['all_data']['Technologies'].loc['BATT_LI', 'f_max'] = 0
    config['all_data']['Technologies'].loc['PHS', 'f_min'] = 0
    config['all_data']['Technologies'].loc['PHS', 'f_max'] = 0

    # config['all_data']['Technologies'].loc['PV', 'c_inv'] = 1982.459017
    # config['all_data']['Technologies'].loc['PV', 'c_maint'] = 22.63983434
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
    cpt_PV = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8822, nrows=24)

    # cpt_WON = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8847, nrows=24)
    # F_WON = outputs['F_WIND_ONSHORE']
    # cpt_WOFF = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8873, nrows=24)
    # F_WOFF = outputs['F_WIND_OFFSHORE']
    # cpt_HR = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8899, nrows=24)
    # F_HR = outputs['F_HYDRO_RIVER']

    alpha = cpt_PV * F_PV.iloc[0, 0] - EUD.values #+ cpt_WON * F_WON.iloc[0, 0] + cpt_WOFF * F_WOFF.iloc[0, 0] + cpt_HR * F_HR.iloc[0, 0]
    alpha = es.check_alpha(alpha)

    thtd = es.generate_t_h_td(config)
    to_use = thtd['t_h_td']

    dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    dict_elec = dict_elec.loc[:, ['ELECTRICITY','PV', 'END_USE','Q_imp']] #['ELECTRICITY','PV', 'END_USE','WIND_ONSHORE','WIND_OFFSHORE','HYDRO_RIVER','Q_imp']

    ## Plot of layer electricity PV_first
    # names = dict_elec.columns.values
    # cmap = []
    # for i in names:
    #     cmap += [dictColor[i]]
    # cmap = lcmp(cmap)
    # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # dict_elec.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    # plt.xticks(fontsize=16,rotation=0)
    # plt.yticks(fontsize=16)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # # plt.ylim([-17, 130])  # à changer pour chaque graphe en fonction min et max
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -17, 17, linestyle='-', color='white', linewidth=2.5)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # # ax.set_title(r'$Layer elec$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=20)
    # ax.set_ylabel('Power [GW]', fontsize=20)
    # fig.tight_layout()
    # fig.show()

    # Example to print the sankey from this script
    #sankey_path = '../case_studies/' + config['case_study'] + '/output/sankey'
    #es.drawSankey(path=sankey_path)

    #Second case run #################################
    data2 = path / 'Data' / '2050elec'
    es_path2 = path / 'energyscope' / 'STEP_2_Energy_Model'
    step1_output2 = path / 'energyscope' / 'STEP_1_TD_selection' / 'TD_of_days.out'
    # specify the configuration
    config2 = {'case_study': 'new_CCGT_0',# Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
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
    config2['all_data']['Resources'].loc['ELECTRICITY', 'c_op'] = 0.5 #config['all_data']['Resources'].loc['ELECTRICITY', 'c_op']
    config2['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = 100000000000
    config2['all_data']['Resources'].loc['ELEC_EXPORT', 'avail'] = 0
    config2['all_data']['Technologies'].loc['PV', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['CCGT_AMMONIA', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['WIND_OFFSHORE', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['HYDRO_RIVER', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['BATT_LI', 'f_max'] = 0
    config2['all_data']['Technologies'].loc['PHS', 'f_min'] = 0
    config2['all_data']['Technologies'].loc['PHS', 'f_max'] = 0

    # config2['all_data']['Technologies'].loc['CCGT', 'c_inv'] = 382.4761605
    # config2['all_data']['Technologies'].loc['CCGT', 'c_maint'] = 9.686869965
    # config2['all_data']['Resources'].loc['GAS', 'c_op'] = 0.026649335
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

    qexp2 = Qimp #+ elec_ext.values
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

    # elec2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
    # elec_ext2 = elec2.pivot(index=' Time', columns='Td ', values='ELECTRICITY')

    dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    dict_alpha = dict_elec2.loc[:, ['alpha']]
    yearly_alpha = es.from_td_to_year(dict_alpha,to_use)
    #To get values of curtailment and all possible production
    prodPV = dict_elec.loc[:, ['PV']]
    all = prodPV + dict_alpha.values
    yearly_all = es.from_td_to_year(all, to_use)
    # with pd.ExcelWriter(path / 'curtailment.xlsx',mode='a') as writer:
    #     yearly_alpha.to_excel(writer, sheet_name='curtailment_02_more')
    #     yearly_all.to_excel(writer, sheet_name='allprod_02_low_more')

    dict_elec2 = dict_elec2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']] #,'CCGT_AMMONIA'

    ## Plot with curtailment of PV_first
    elec1_alpha = pd.merge(dict_elec, dict_alpha, right_index=True, left_index=True)
    elec1_alpha = elec1_alpha.rename(columns={"alpha":"curtailment"})
    names = elec1_alpha.columns.values
    cmap = []
    for i in names:
        cmap += [dictColor[i]]
    cmap = lcmp(cmap)
    xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    fig, ax = plt.subplots(figsize=(13, 7))
    elec1_alpha.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    plt.xticks(fontsize=16,rotation=0)
    plt.yticks(fontsize=16)
    xlab = [str(i) for i in np.arange(1, 13)]
    ax.set_xticklabels(labels=xlab)
    plt.vlines(np.arange(24, 24 * 11 + 1, 24), -17, 15, linestyle='-', color='white', linewidth=2.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel('Typical days', fontsize=20)
    ax.set_ylabel('Power [GW]', fontsize=20)
    fig.tight_layout()
    fig.show()

    ## Plot of layer electricity CCGT_0
    names = dict_elec2.columns.values
    cmap = []
    for j in names:
        cmap += [dictColor[j]]
    cmap = lcmp(cmap)
    fig, ax = plt.subplots(figsize=(13, 7))
    dict_elec2.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    plt.xticks(fontsize=16,rotation=0)
    plt.yticks(fontsize=16)
    xlab = [str(i) for i in np.arange(1, 13)]
    ax.set_xticklabels(labels=xlab)
    plt.ylim([-35, 35])  # à changer pour chaque graphe en fonction min et max
    plt.vlines(np.arange(24, 24 * 11 + 1, 24), -35, 35, linestyle='-', color='white', linewidth=2) #linestyle='--', color='0.5')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel('Typical days', fontsize=20)
    ax.set_ylabel('Power [GW]', fontsize=20)
    fig.tight_layout()
    fig.show()

    ## Plot of marginal cost PV_first/import cost CCGT_0
    # mc_PV = es.read_layer(config['case_study'], 'mc_scaled')
    # mc_PV = mc_PV.loc[:,['ELECTRICITY']]
    ##these two are the same
    mcPV = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'], index_col=[0, 1])
    xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    fig, ax = plt.subplots(figsize=(13, 7))
    mcPV.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0,xticks=xticks, legend=False)
    plt.xticks(fontsize=16,rotation=0)
    plt.yticks(fontsize=16)
    xlab = [str(i) for i in np.arange(1, 13)]
    ax.set_xticklabels(labels=xlab)
    plt.xlim([0, 288])
    plt.vlines(np.arange(24, 24 * 11 + 1, 24), 0, 0.1, linestyle='dashed', color='white',linewidth=2) #linestyle='--', color='0.5'
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #ax.set_title(r'$Marginal cost$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    ax.set_xlabel('Typical days', fontsize=20)
    ax.set_ylabel('Marginal cost [M€/GWh]', fontsize=20)
    fig.tight_layout()
    fig.show()

    for i in range(1,201):
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
        cpt_PV = pd.read_csv(path_dat / 'ESTD_12TD.dat', sep='\t', index_col=0, skiprows=8822, nrows=24)

        EUD2 = outputs2['var_END_USES'].pivot(index='Hour', columns='TD', values='ELECTRICITY')
        F_CCGT = outputs2['F_CCGT']

        alpha = cpt_PV * F_PV.iloc[0, 0] - EUD.values #+ Sto_out_BATT.values + Sto_in_BATT.values #+ Sto_out_PHS.values + Sto_in_PHS.values - qexp.values + Qimp.values
        alpha = es.check_alpha(alpha)

        alpha2 = cpt_CCGT * F_CCGT.iloc[0, 0] - EUD2.values #- qexp2.values + Qimp2.values
        alpha2 = es.check_alpha(alpha2)

        s = '["' + 'ELECTRICITY' + '",*,*]:'

        if i == 0 or i % 2 == 0 :  #even iteration -> case CCGT
            config2['case_study'] = 'new_CCGT_' + str(i)
            es.print_data(config2)
            path_dat2 = path / 'case_studies' / config2['case_study']
            cs2 = path_dat2 / 'output' / 'hourly_data'

            cimp2 = pivot_mc
            with open(path_dat2 / 'ESTD_cimp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param c_imp:='])
            cimp2 = es.ampl_syntax(cimp2, ' ')
            cimp2.to_csv(path_dat2 / 'ESTD_cimp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            qexp2 = Qimp #+ elec_ext.values
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
            # dict_alpha = dict_elec2.loc[:, ['alpha']]
            dict_elec2 = dict_elec2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]

            elec2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
            elec_ext2 = elec.pivot(index=' Time', columns='Td ', values='ELECTRICITY')

            # yearly_alpha = es.from_td_to_year(dict_alpha, to_use)
            ## To get values of curtailment and all possible production
            # prodPV = dict_elec.loc[:, ['PV']]
            # all = prodPV + dict_alpha.values
            # yearly_all = es.from_td_to_year(all, to_use)
            # with pd.ExcelWriter(path / 'final.xlsx',mode='a') as writer:
            #     yearly_alpha.to_excel(writer, sheet_name='curtail_'+str(i))
            #     yearly_all.to_excel(writer, sheet_name='allprod'+str(i))

            # dict_elec2 = dict_elec2.rename(columns={"Q_imp": "Q_imp_from_PV", "q_exp": "q_exp_from_PV", "END_USE": "END_USE_CCGT"})

            # both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
            # both.plot(kind='bar', stacked=True, position=0, width=1.0, legend=True, colormap='tab20', xticks=xticks)
            # es.hourly_plot(dict_elec2)
            # yearly2 = es.from_td_to_year(dict_elec2, to_use)
            # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 0.0]
            # yearly2 = yearly2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
            # # yearly2 = yearly2.loc[:, ['CCGT', 'Q_exch', 'END_USE']]
            # yearly2.plot(title='Yearly layer ELEC (>0)')
            # es.hourly_plot(dict_elec2)

            # names = dict_elec2.columns.values
            # cmap = []
            # for j in names:
            #     cmap += [dictColor[j]]
            # cmap = lcmp(cmap)
            # fig, ax = plt.subplots(figsize=(13, 7))
            # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
            # dict_elec2.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
            # plt.xticks(rotation=0)
            # xlab = [str(i) for i in np.arange(1, 13)]
            # ax.set_xticklabels(labels=xlab)
            # plt.ylim([-35, 35])  # à changer pour chaque graphe en fonction min et max
            # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -35, 35, linestyle='-', color='white', linewidth=2) #linestyle='--', color='0.5')
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
            # ax.set_xlabel('Typical days', fontsize=16)
            # ax.set_ylabel('Power [GW]', fontsize=16)
            # fig.tight_layout()
            # fig.show()

        else : #odd iteration -> case PV
            config['case_study'] = 'new_PV_' + str(i)
            es.print_data(config)
            path_dat = path / 'case_studies' / config['case_study']
            cs = path_dat / 'output' / 'hourly_data'

            cimp = pivot_mc2
            with open(path_dat / 'ESTD_cimp.dat', mode='w', newline='') as td_file:
                td_writer = csv.writer(td_file, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
                td_writer.writerow(['param c_imp:='])
            cimp = es.ampl_syntax(cimp, ' ')
            cimp.to_csv(path_dat / 'ESTD_cimp.dat', sep='\t', mode='a', header=True, index=True, index_label=s,quoting=csv.QUOTE_NONE)

            qexp = Qimp2 #+ elec_ext2.values
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
            outputs = es.read_outputs(config['case_study'], hourly_data=True)

            dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
            dict_elec = dict_elec.loc[:, ['ELECTRICITY', 'PV', 'Q_imp', 'q_exp', 'END_USE']]

            elec = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
            elec_ext = elec.pivot(index=' Time', columns='Td ', values='ELECTRICITY')

            # dict_elec = dict_elec.rename(columns={"Q_imp": "Q_imp_from_CCGT", "q_exp": "q_exp_from_CCGT", "END_USE": "END_USE_PV"})

            # both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
            # both.plot(kind='bar', stacked=True, position=0, width=1.0, legend=True, colormap='tab20', xticks=xticks)
            # es.hourly_plot(dict_elec)
            # yearly = es.from_td_to_year(dict_elec, to_use)
            # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
            # yearly = yearly.loc[:,['ELECTRICITY','PV','Q_imp','q_exp','END_USE']]
            # yearly = yearly.loc[:, ['PV','Q_exch','END_USE']]
            # yearly.plot(title='Yearly layer ELEC')
            # dict_elec2.plot(ax=ax2)

            # names = dict_elec.columns.values
            # cmap = []
            # for j in names:
            #     cmap += [dictColor[j]]
            # cmap = lcmp(cmap)
            # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
            # fig, ax = plt.subplots(figsize=(13, 7))
            # dict_elec.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
            # plt.xticks(rotation=0)
            # xlab = [str(i) for i in np.arange(1, 13)]
            # ax.set_xticklabels(labels=xlab)
            # # plt.ylim([-17, 130])  # à changer pour chaque graphe en fonction min et max
            # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -20, 20, linestyle='-', color='white', linewidth=2.5)
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # #ax.set_title(r'$Layer elec$')
            # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
            # ax.set_xlabel('Typical days', fontsize=16)
            # ax.set_ylabel('Power [GW]', fontsize=16)
            # fig.tight_layout()
            # fig.show()

            ##Plot with curtailment
            # elec1_alpha = pd.merge(dict_elec, dict_alpha, right_index=True, left_index=True)
            # elec1_alpha = elec1_alpha.rename(columns={"alpha":"curtailment"})
            # names = elec1_alpha.columns.values
            # cmap = []
            # for j in names:
            #     cmap += [dictColor[j]]
            # cmap = lcmp(cmap)
            # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
            # fig, ax = plt.subplots(figsize=(13, 7))
            # elec1_alpha.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
            # plt.xticks(rotation=0)
            # xlab = [str(i) for i in np.arange(1, 13)]
            # ax.set_xticklabels(labels=xlab)
            # # plt.ylim([-17, 130])  # à changer pour chaque graphe en fonction min et max
            # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -20, 100, linestyle='-', color='white', linewidth=2.5)
            # plt.gca().spines['top'].set_visible(False)
            # plt.gca().spines['right'].set_visible(False)
            # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
            # ax.set_xlabel('Typical days', fontsize=14)
            # ax.set_ylabel('Power [GW]', fontsize=14)
            # fig.tight_layout()
            # fig.show()

        i = i + 1

    ##Evolution graphs installed quantity
    # itebatt = file_tracker(path / 'case_studies' / 'cimp=0.2&both_batt')
    # ite,totPV = file_tracker(path / 'case_studies' / 'cimp=0.05')
    # allCCGT,totC = track_CCGT(path / 'case_studies' / 'cimp=0.05')
    # fig = plt.figure(figsize=(6, 9))
    # ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
    # x = ['first']+list(np.array(np.arange(1,200,2),dtype=str))
    # x = np.append([0],np.arange(1,200,2))
    # ax1.plot(x, totPV, linestyle='solid', marker='.')
    # ax1.set_xticks(x[::10],labels=np.append(['first'], x[::10][1:]),fontsize=16)
    # # ax1.set_ylim([25, 30])  # CHANGER À CHAQUE FOIS
    # ax1.set_yticks([min(totPV), (min(totPV)+max(totPV))/2,max(totPV)], labels=[np.round(min(totPV)), np.round((min(totC)+max(totC))/2), max(totPV)], fontsize=16)
    # # ax1.set_yticks([0,min(ite), (min(ite) + max(ite)) / 2,max(ite)],labels=[0,np.round(min(ite)), (min(ite) + max(ite)) / 2, max(ite)], fontsize=16) # (min(ite) + max(ite)) / 2
    # xpair = np.arange(0, 201, 2)
    # ax2.plot(xpair, totC, linestyle='solid', marker='.')
    # ax2.set_xticks(xpair[::10], labels=xpair[::10],fontsize=16)
    # # ax2.set_ylim([0, 14])
    # ax2.set_yticks([min(totC), (min(totC)+max(totC))/2, max(totC)], labels=[np.round(min(totC)), np.round((min(totC)+max(totC))/2), np.round(max(totC))], fontsize=16)
    # # ax2.set_yticks([0,min(allCCGT)-1, max(allCCGT)],labels=[0,np.round(min(allCCGT)-1), np.round(max(allCCGT))],fontsize=16)
    # ax2.set_xlabel('Iterations', fontsize = 20)
    # for ax in [ax1,ax2]:
    #     # ax.set_ylabel('Power [GW]', fontsize=20)
    #     ax.set_ylabel('Total Cost [M€]', fontsize=20)
    #     plt.sca(ax)
    #     plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d")) #%.2f  ou   d
    #     plt.gca().spines['top'].set_color('none')
    #     plt.gca().spines['right'].set_color('none')

    # qimp = track_Qimp_C(path / 'case_studies' / 'cimp=0.05')
    # qimppv = track_Qimp_PV(path / 'case_studies' / 'cimp=0.05')
    # ite,tot = file_tracker(path / 'case_studies' / 'cimp=0.05')
    # allCCGT = track_CCGT(path / 'case_studies' / 'cext=0.5&cimp=0.05&no_new_qexp')
    # plt.figure()
    # x = ['first'] + list(np.arange(1, 200, 2))
    # plt.plot([0]+list(x[1:]), ite, linestyle='solid',marker='.')
    # plt.xticks([0]+x[::10][1:],labels=x[::10],fontsize=16)
    # # xpair = np.arange(0, 201, 2)
    # # plt.plot(xpair, qimp, linestyle='solid', marker='.')
    # # plt.xticks(xpair[::10], labels=xpair[::10], fontsize=16)
    # # plt.ylim([0, 100])  # CHANGER À CHAQUE FOIS
    # # yfin = [173.685496,211.493637,254.754118,287.052940]
    # # plt.yticks(yfin,fontsize=14)
    # plt.yticks(fontsize=16)
    # plt.xlabel('Iterations', fontsize=20)
    # # plt.ylabel('Total Cost [M€]', fontsize=16)
    # plt.ylabel('Power [GW]', fontsize=20)
    # plt.gca().spines['top'].set_color('none')
    # plt.gca().spines['right'].set_color('none')
    # plt.show()