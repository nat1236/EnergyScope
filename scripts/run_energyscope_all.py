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


def PV_finder(path):
     f = open(path, 'r')
     line = f.readline()
     t = line.split('\t')
     idx = 0
     for elem in t:
         if elem == ' f':
             break
         idx += 1
     while 'PV' not in line:
         line = f.readline()
     split = line.split('\t')
     return float(split[idx])


def file_tracker(Base):
     vec = []
     l = []
     for dirr in os.listdir(Base):
         if 'new_PV' in dirr and 'first' not in dirr:
             l.append(dirr)
     l.sort(key=lambda x: int(re.search(r'\d+', x).group()))
     l.insert(0, 'new_PV_first')
     for dirr in l:
         if 'first' in dirr:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/assets.txt')):
                 vec.append(PV_finder(os.path.join(path, 'output/assets.txt')))
         elif int(re.search(r'\d+', dirr).group()) % 2 == 1:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/assets.txt')):
                 vec.append(PV_finder(os.path.join(path, 'output/assets.txt')))
     return vec


def CCGT_find(path):
     f = open(path, 'r')
     line = f.readline()
     t = line.split('\t')
     idx = 0
     for elem in t:
         if elem == ' f':
             break
         idx += 1
     while 'CCGT' not in line:
         line = f.readline()
     split = line.split('\t')
     return float(split[idx])


def track_CCGT(Base):
     vec = []
     l = []
     for dirr in os.listdir(Base):
         if 'new_CCGT' in dirr:
             l.append(dirr)
     l.sort(key=lambda x: int(re.search(r'\d+', x).group()))
     for dirr in l:
         if int(re.search(r'\d+', dirr).group()) % 2 == 0:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/assets.txt')):
                 vec.append(CCGT_find(os.path.join(path, 'output/assets.txt')))
     return vec


 ###################
def Qimp_finder(path):
     f = open(path, 'r')
     line = f.readline()
     t = line.split('\t')
     idx = 0
     for elem in t:
         if 'ELECTRICITY' in elem:
             print('break')
             break
         idx += 1
     while 'Q_imp' not in line:
         line = f.readline()
     split = line.split('\t')
     return float(split[idx])


def track_Qimp_PV(Base):
     vec = []
     l = []
     for dirr in os.listdir(Base):
         if 'new_PV' in dirr and 'first' not in dirr:
             l.append(dirr)
     l.sort(key=lambda x: int(re.search(r'\d+', x).group()))
     l.insert(0, 'new_PV_first')
     for dirr in l:
         if 'first' in dirr:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/year_balance.txt')):
                 vec.append(Qimp_finder(os.path.join(path, 'output/year_balance.txt')))
         elif int(re.search(r'\d+', dirr).group()) % 2 == 1:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/year_balance.txt')):
                 vec.append(Qimp_finder(os.path.join(path, 'output/year_balance.txt')))
     return vec


def track_Qimp_C(Base):
     vec = []
     l = []
     for dirr in os.listdir(Base):
         if 'new_CCGT' in dirr:
             l.append(dirr)
     l.sort(key=lambda x: int(re.search(r'\d+', x).group()))
     for dirr in l:
         if int(re.search(r'\d+', dirr).group()) % 2 == 0:
             path = os.path.join(Base, dirr)
             if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'output/year_balance.txt')):
                 vec.append(Qimp_finder(os.path.join(path, 'output/year_balance.txt')))
     return vec


if __name__ == '__main__':
    # First case run   #######################
    #dictionary for the colors
    dictColor = {'curtailment':'gold','PV': 'yellow','CCGT':'goldenrod', 'END_USE':'lightblue', 'Q_imp':'rebeccapurple','q_exp':'mediumblue','q_exp_from_PV':'mediumblue','Outside_elec':'deepskyblue','ELECTRICITY': 'deepskyblue'}

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

    # config['all_data']['Technologies'].loc['BATT_LI', 'f_max'] = 0
    # config['all_data']['Technologies'].loc['PHS', 'f_min'] = 0
    # config['all_data']['Technologies'].loc['PHS', 'f_max'] = 0

    # config['all_data']['Technologies'].loc['PV', 'f_min'] = 50

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

    alpha = cpt_PV * F_PV.iloc[0, 0] - EUD.values  #+ Sto_out_BATT.values + Sto_in_BATT.values #+ Sto_out_PHS.values + Sto_in_PHS.values - qexp.values + Qimp.values
    alpha = es.check_alpha(alpha)

    thtd = es.generate_t_h_td(config)
    to_use = thtd['t_h_td']

    #computing lcoe
    # lcoe_pv = es.comp_lcoe_pv(outputs)
    # lcoe_tab = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # lcoe_tab.iloc[:] = lcoe_pv

    # #2 different plots
    # mc_scaled = es.read_layer(config['case_study'], 'mc_scaled')
    # yearly_mc = es.from_td_to_year(mc_scaled, to_use)
    # yearly_mc = yearly_mc.loc[:, yearly_mc.sum().abs() > 1.0]
    # yearly_mc.plot(title='Yearly marginal cost')

    dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
    dict_elec = dict_elec.loc[:, ['ELECTRICITY','PV', 'Q_imp', 'END_USE']]

    elec = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
    elec_ext = elec.pivot(index=' Time', columns='Td ', values='ELECTRICITY')
    # es.hourly_plot(dict_elec)
    # yearly = es.from_td_to_year(dict_elec, to_use)
    # yearly = yearly.loc[:, yearly.sum().abs() > 1.0]
    # yearly = yearly.loc[:, ['PV', 'Q_imp', 'END_USE']]

    # dict_elec1 = dict_elec.loc[:,['Q_imp','PV']]  #pour le graphe avec juste échange
    # firstExch = es.hourly_plot(dict_elec1) #dict_elec.plot(title='Layer ELEC')

    ## Plot of layer electricity PV_first
    # names = dict_elec.columns.values
    # cmap = []
    # for i in names:
    #     cmap += [dictColor[i]]
    # cmap = lcmp(cmap)
    # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # dict_elec.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    # plt.xticks(rotation=0)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # # plt.ylim([-17, 130])  # à changer pour chaque graphe en fonction min et max
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -17, 17, linestyle='-', color='white', linewidth=2.5)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # ax.set_title(r'$Layer elec$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=16)
    # ax.set_ylabel('Power [GW]', fontsize=16)
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

    cimp2 = pivot_mc #lcoe_tab
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

    # lcoe_ccgt = es.comp_lcoe_ccgt(outputs2)
    # lcoe_tab2 = pd.DataFrame(index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # lcoe_tab2.iloc[:] = lcoe_ccgt

    # #2 different plots
    # mc_scaled2 = es.read_layer(config2['case_study'], 'mc_scaled')
    # yearly_mc2 = es.from_td_to_year(mc_scaled2, to_use)
    # yearly_mc2 = yearly_mc2.loc[:, yearly_mc2.sum().abs() > 1.0]
    # yearly_mc2.plot(title='Yearly marginal cost')

    elec2 = pd.read_csv(cs2 / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
    elec_ext2 = elec2.pivot(index=' Time', columns='Td ', values='ELECTRICITY')

    dict_elec2 = es.read_layer(config2['case_study'], 'layer_ELECTRICITY')
    dict_alpha = dict_elec2.loc[:, ['alpha']]
    yearly_alpha = es.from_td_to_year(dict_alpha,to_use)
    #To get values of curtailment and all possible production
    prodPV = dict_elec.loc[:, ['PV']]
    all = prodPV + dict_alpha.values
    yearly_all = es.from_td_to_year(all, to_use)
    # with pd.ExcelWriter(path / 'final.xlsx',mode='a') as writer:
    #     yearly_alpha.to_excel(writer, sheet_name='curtailment_PV24')
    #     yearly_all.to_excel(writer, sheet_name='allprod_PV24')

    dict_elec2 = dict_elec2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
    dict_elec2_1 = dict_elec2.loc[:,['ELECTRICITY','q_exp','CCGT']]
    dict_elec2_1 = dict_elec2_1.rename(columns={"ELECTRICITY":"Outside_elec", "q_exp":"q_exp_from_PV"})

    ## Plot with curtailment of PV_first
    # elec1_alpha = pd.merge(dict_elec, dict_alpha, right_index=True, left_index=True)
    # elec1_alpha = elec1_alpha.rename(columns={"alpha":"curtailment"})
    # names = elec1_alpha.columns.values
    # cmap = []
    # for i in names:
    #     cmap += [dictColor[i]]
    # cmap = lcmp(cmap)
    # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # elec1_alpha.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    # plt.xticks(rotation=0)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # # plt.ylim([-17, 130])  # à changer pour chaque graphe en fonction min et max
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -17, 90, linestyle='-', color='white', linewidth=2.5)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # #ax.set_title(r'$Layer elec$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=16)
    # ax.set_ylabel('Power [GW]', fontsize=16)
    # fig.tight_layout()
    # fig.show()

    #Try plot with exchange
    # both = pd.merge(dict_elec1,dict_elec2_1,right_index =True, left_index=True)
    # names = both.columns.values
    # cmap = []
    # for i in names:
    #     cmap += [dictColor[i]]
    # cmap = lcmp(cmap)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # both.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    # plt.xticks(rotation=0)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # plt.ylim([-17, 50])  # à changer pour chaque graphe en fonction min et max
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -17, 50, linestyle='-', color='white',
    #            linewidth=2)  # linestyle='--', color='0.5')
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # #ax.set_title(r'$Layer elec$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=14)
    # ax.set_ylabel('Power [GW]', fontsize=14)
    # fig.tight_layout()
    # fig.show()

    # dict_elec2 = dict_elec2.drop(columns='alpha')
    # es.hourly_plot(dict_elec2)
    # yearly2 = es.from_td_to_year(dict_elec2, to_use)
    # yearly2 = yearly2.loc[:, yearly2.sum().abs() > 1.0]
    # yearly2 = yearly2.loc[:, ['ELECTRICITY', 'CCGT', 'Q_imp', 'q_exp', 'END_USE']]
    # dict_elec2.plot(kind='bar', stacked = True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap='tab20')

    ## Plot of layer electricity CCGT_0
    # names = dict_elec2.columns.values
    # cmap = []
    # for j in names:
    #     cmap += [dictColor[j]]
    # cmap = lcmp(cmap)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # dict_elec2.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0, legend=True, xticks=xticks, colormap=cmap)
    # plt.xticks(rotation=0)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # plt.ylim([-35, 35])  # à changer pour chaque graphe en fonction min et max
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), -35, 35, linestyle='-', color='white', linewidth=2) #linestyle='--', color='0.5')
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # #ax.set_title(r'$Layer elec$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=16)
    # ax.set_ylabel('Power [GW]', fontsize=16)
    # fig.tight_layout()
    # fig.show()

    ## Plot of marginal cost PV_first/import cost CCGT_0
    # mc_PV = es.read_layer(config['case_study'], 'mc_scaled')
    # mc_PV = mc_PV.loc[:,['ELECTRICITY']]
    ##these two are the same
    # mcPV = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'], index_col=[0, 1])
    # xticks = np.arange(12, dict_elec.shape[0] + 1, 24)
    # fig, ax = plt.subplots(figsize=(13, 7))
    # mcPV.plot(xticks=xticks,ax=ax,legend=False)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # plt.xlim([0, 288])
    # plt.ylim([0,2.1])
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), 0, 2.2, linestyle='--', color='0.5')
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # #ax.set_title(r'$Marginal cost$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=16)
    # ax.set_ylabel('Cost [M€]', fontsize=16)
    # fig.tight_layout()
    # fig.show()

    # fig, ax = plt.subplots(figsize=(13, 7))
    # mcPV.plot(kind='bar', stacked=True, ax=ax, position=0, width=1.0,xticks=xticks, legend=False)
    # xlab = [str(i) for i in np.arange(1, 13)]
    # ax.set_xticklabels(labels=xlab)
    # plt.xlim([0, 288])
    # plt.vlines(np.arange(24, 24 * 11 + 1, 24), 0, 2.2, linestyle='--', color='0.5')
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # #ax.set_title(r'$Marginal cost$')
    # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
    # ax.set_xlabel('Typical days', fontsize=16)
    # ax.set_ylabel('Cost [M€]', fontsize=16)
    # fig.tight_layout()
    # fig.show()

    # creating data frames to store the mean of marginal costs
    # mc_debut = pd.read_csv(cs / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    # marginal_cost = mc_debut.pivot(index='Hour', columns='TD', values='ELECTRICITY')
    # marginal_cost.iloc[:] = 0
    #
    # mc2_debut = pd.read_csv(cs2 / 'mc_scaled.txt', sep='\t', usecols=['Hour', 'TD', 'ELECTRICITY'])
    # marginal_cost2 = mc2_debut.pivot(index='Hour', columns='TD', values='ELECTRICITY')
    # marginal_cost2.iloc[:] = 0

    for i in range(1,1):  #pour l'instant 5 itérations

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

        # updating the mean of marginal costs
        # marginal_cost.iloc[:] = (i / (i + 1)) * marginal_cost.iloc[:] + (1 / (i + 1)) * pivot_mc  # stockage de la moyenne
        # marginal_cost2.iloc[:] = (i / (i + 1)) * marginal_cost2.iloc[:] + (1 / (i + 1)) * pivot_mc2

        if i == 0 or i % 2 == 0 :  #even iteration -> case CCGT
            # marginal_cost.iloc[:] = (i / (i + 1)) * marginal_cost.iloc[:] + (1 / (i + 1)) * pivot_mc  # stockage de la moyenne

            config2['case_study'] = 'new_CCGT_' + str(i)
            es.print_data(config2)
            path_dat2 = path / 'case_studies' / config2['case_study']
            cs2 = path_dat2 / 'output' / 'hourly_data'

            cimp2 = pivot_mc #es.which_cost(Qimp2, pivot_mc2,pivot_mc)  #  #marginal_cost
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
            dict_alpha = dict_elec2.loc[:, ['alpha']]
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

            both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
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
            # #ax.set_title(r'$Layer elec$')
            # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
            # ax.set_xlabel('Typical days', fontsize=16)
            # ax.set_ylabel('Power [GW]', fontsize=16)
            # fig.tight_layout()
            # fig.show()

        else : #odd iteration -> case PV
            # marginal_cost2.iloc[:] = (i / (i + 1)) * marginal_cost2.iloc[:] + (1 / (i + 1)) * pivot_mc2

            config['case_study'] = 'new_PV_' + str(i)
            es.print_data(config)
            path_dat = path / 'case_studies' / config['case_study']
            cs = path_dat / 'output' / 'hourly_data'

            cimp = pivot_mc2 # lcoe_tab2 # es.which_cost(Qimp,pivot_mc, pivot_mc2)
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
            outputs = es.read_outputs(config['case_study'], hourly_data=True)  # layers = ['layer_ELECTRICITY'])

            dict_elec = es.read_layer(config['case_study'], 'layer_ELECTRICITY')
            dict_elec = dict_elec.loc[:, ['ELECTRICITY', 'PV', 'Q_imp', 'q_exp', 'END_USE']]

            elec = pd.read_csv(cs / 'layer_ELECTRICITY.txt', sep='\t', usecols=[' Time', 'Td ', 'ELECTRICITY'])
            elec_ext = elec.pivot(index=' Time', columns='Td ', values='ELECTRICITY')

            # dict_elec = dict_elec.rename(columns={"Q_imp": "Q_imp_from_CCGT", "q_exp": "q_exp_from_CCGT", "END_USE": "END_USE_PV"})

            both = pd.merge(dict_elec, dict_elec2, right_index=True, left_index=True)
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
            # ax.set_title(r'$Layer elec$')
            # ax.legend(ncol=2)  # ax.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
            # ax.set_xlabel('Typical days', fontsize=14)
            # ax.set_ylabel('Power [GW]', fontsize=14)
            # fig.tight_layout()
            # fig.show()

        i = i + 1
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

    ##Graphes d'évolution quantité installée
    # ycostPV = [123621, 22516, 22519, 22519, 22707, 22913, 22706, 22907, 22882, 22681, 21971] #19iterations
    ## prochains sont les quantités intallées de PV
    # y = [173.685496,319.890316,293.351660,263.236101,456.392331,332.359575,319.890303,287.052951,456.392309,263.236110,456.392316,287.052951,456.392309]

    # yalpha11 = [173.685496,185.983834,194.377080,203.178284,211.493637,220.513519,229.574508,238.674014,254.754118,264.994212,270.102803,274.914158,287.052940]

    # yalpha = [173.685496,177.878811,178.031713,178.038324,181.408444,181.574659,181.575528,184.748782,184.905289,176.544910,165.803486] #,169.740115,169.899523,
    # 169.906977,170.564510,174.570378,174.766334,167.853065,171.890769,172.019797,172.024909,172.025050,172.025022,167.853065,171.890769,172.062425,172.069628,176.315290,
    # 176.524677,176.526534,176.526634,167.853065,170.175749,172.585024,172.673131,167.853065,170.175749,172.585024,172.673131,167.853065,170.175749,172.585024,172.673131,
    # 167.853065,170.175749,172.585024,172.673131,167.853065,170.175749,172.585024,172.673131]

    # ycimp1 = [121.323858,124.359732,124.473525,124.727569,124.923396,124.932452,124.932448,124.932451,127.966994,130.916088,131.060463,134.162109,137.211662,137.354569,133.893947,
    #           130.324292,133.596134,133.734222,133.893948,133.901714,134.723824,134.764372,134.765566,137.755839,140.820819,140.931998,144.226385,144.630588,144.650281,144.650665,
    #           144.650681,144.650678,148.128458,148.577424,152.248867,152.429115,152.429129,154.691698,143.843690,138.070901,133.893947,130.324292,133.596134,133.734222,133.893948,
    #           133.901714,134.723824,134.764372,134.765566,137.755839,140.820819]

    # ycimp3 = [222.258382,227.236263,227.356998,227.360833,227.870896,227.896032,220.268280,224.979674,225.204104,230.158939,230.402826,231.062907,220.268303,209.726669,214.815031,
    # 214.924415,214.927622,220.268308,220.484654,220.487595,209.726669,214.114474,216.892868,217.029396,217.029543,221.777489,225.242026,225.401337,229.661246,220.268300,224.979697,
    # 225.014438,224.308980,209.726669,214.815031,214.924415,214.927622,220.268308,220.484654,220.487595,209.726669,214.114474,216.892868,217.029396,217.029543,221.777489,225.242026,
    # 225.401337, 229.661246,220.268300, 224.979697]

    # yPVplus = [117.232666,120.066737,120.155790,123.134028,123.280822,123.280831,118.989656,121.837607,121.872541,121.873424,116.816301,119.629847,119.718902,122.675599,122.821319,
    #            122.821326,122.821319,124.954286,125.059477,125.059725,118.989656,115.907304,118.707822,118.798448,118.802433,119.908052,119.962576,118.989656,121.910390,121.968544,
    #            121.970973,124.068347,124.171784,124.174844,124.174880,115.907318,118.676035,118.780678,118.785402,121.696043,121.839588,124.721602,127.595553,127.730407,127.736132,130.813216,
    #            130.964958,129.182841,124.068347,127.060278,127.158581]

    # yCCGTlessbad = [173.685496,177.878817,178.031713,178.038324,181.408442,185.601628,185.801095,185.801149,185.801120,187.491236,191.834577,192.048678,192.050731,192.050819,181.408453,
    #              167.853071,170.175765,170.220198,174.374649,175.501366,175.555678,175.555915,177.357564,177.446409,167.853071,171.890782,172.062468,172.069312,173.082331,173.132275,
    #              167.853072,159.555815,163.298100,163.432738,163.438694,163.438881,163.438883,156.396858,159.869568,159.998587,160.003336,163.653917,163.833939,167.673436,167.862669,
    #              156.396865,152.562572,156.357815,156.504780,156.512009,157.522115]

    # yCCGTless = [173.685496,177.845313,177.996546,178.003161,178.003358,179.309025,179.373419,179.373364,176.544909,167.853065,170.175740,165.803486,156.396871,143.843641,142.946296,
    #              146.071787,146.173335,149.457996,149.619876,140.652516,133.893944,130.324297,126.292684,129.394313,129.533797,130.324433,130.363278,130.363281,124.727559,124.068348,
    #              116.816301,115.907315,114.826577,100.758324,103.019426,103.094994,103.098529,105.475043,105.592249,105.592673,107.779924,107.814855,110.423971,110.552626,110.558930,
    #              112.884598,114.527519,114.602878,106.120868,100.758317,98.867112]

    # yPV_BATTLI = [382.422101,110.320429,63.280675,51.664058,51.678651,56.485043,56.484957,57.399596,57.399527,57.564940,57.564872,57.593408,57.593337,57.595492,57.595416,57.595654,
    #               57.595580,57.595637,57.595561,57.595684,57.595609,56.287127,53.613485,55.709215,55.709203,57.259479,57.259411,57.539753,57.539683,57.589927,57.589858,57.595248,
    #               57.595178,57.595626,57.595554,57.590346,53.613485,55.709221,55.709208,57.259457,57.259386,57.539221,57.539153,57.589841,57.589773,57.595165,57.595096,57.594367,
    #               57.594353,57.595615,57.595545]
    # yBATTLI = [213.612634,105.067814,40.375724,40.202191,26.884020,58.592388,28.174775,58.592388,27.750597,58.592388,27.673911,58.592389,27.660708,58.592389,27.659744,58.592388,
    #            27.659668,58.592388,27.659677,58.592388,27.659655,58.592388,29.506563,58.592373,28.534569,58.592389,27.815583,58.592388,27.685593,58.592388,27.662322,58.592388,
    #            27.659855,58.592388,27.659680,58.592389,29.506563,58.592377,28.534567,58.592388,27.815595,58.592388,27.685839,58.592388,27.662362,58.592388,27.659893,58.592388,
    #            27.660237,58.592389,27.659684]

    # yPV_PHS = [190.172981,181.611663,157.961540,147.929050,139.910718,134.003197,126.405839,120.024634,112.149644,107.741979,101.964394,97.381753,91.282820,87.259613,81.589488,78.262699,
    #            73.913225,70.724226,67.236187,64.990202,61.630692,59.154695,57.786859,54.439840,53.007696,49.631109,47.986862,45.601113,43.840463,41.677294,41.651673,41.595858,41.686970,
    #            41.686968,41.971407,42.175267,42.394854,42.394848,42.394844,42.394823,42.394813,42.394795,42.394772,42.394770,42.394765,42.394750,42.394737,42.394711,42.394706,
    #            42.394678,42.394675]

    # yPV_both = [382.272609,112.558208,63.880006,51.509289,54.351357,55.248945,55.248927,57.304279,57.304227,57.675874,57.675814,57.743072,57.743025,52.294596,54.684971,54.684948,56.362327,
    #             56.487107,56.487100,57.528142,57.528083,57.716362,57.716310,57.749646,57.749589,52.294597,54.684769,54.684742,56.362322,56.487108,56.487099,57.528138,57.528081,57.716362,
    #             57.716309,57.749648,57.749585,52.294599,54.684823,54.684796,56.362322,56.487108,56.487099,57.528142,57.528086,57.716363,57.716310,57.749638,57.749573,52.294597,54.684907]
    # yBATTLI_both = [208.309050,95.927123,35.280008,32.386840,33.743332,42.772964,22.248047,52.092388,21.294811,52.092388,39.244739,52.092388,39.234289,23.618264,52.092388,22.509619,48.089496,
    #             39.429540,21.673790,52.092388,21.190992,52.092388,39.238436,52.092388,39.233260,23.618263,52.092388,22.509714,48.124969,39.429540,21.673790,52.092388,21.190985,52.092388,
    #             39.238438,52.092388,39.233258,23.618262,52.092388,22.509689,48.102806,39.429540,21.673799,52.092388,21.190987,52.092388,39.238431,52.092388,39.233266,23.618263,52.092388]

    # ycostc1 = [69333.1006,19724.7989,19730.3465,19778.1772,19767.6833,19752.3946,19752.3866,19816.3997,19907.5516,20059.4197,20065.0603] #19iterations
    # ycostc3 = [176077.1617,25137.4778,25146.4262,25141.8971,25409.81495,25170.7034,24912.7135,25022.9807,25119.9499,25302.1353,25305.8669] #19iterations
    # hello = file_tracker(path / 'case_studies' / '200iterations')
    # qimp = track_Qimp_C(path / 'case_studies' / 'cext=0.5&cimp=0.05&no_new_qexp')
    # qimppv = track_Qimp_PV(path / 'case_studies' / 'cext=0.5&cimp=0.05&no_new_qexp')
    # ite = file_tracker(path / 'case_studies' / 'cext=0.5&cimp=0.1&no_new_qexp')
    # allCCGT = track_CCGT(path / 'case_studies' / 'cimp=0.2&both_batt')
    # ite = file_tracker(path / 'case_studies' / 'cimp=0.2&both_batt')
    # xpair = np.arange(0,201,2)
    # x = ['first']+list(np.arange(1,200,2))
    # plt.figure()
    # plt.plot([0]+list(x[1:]), qimppv, linestyle='solid',marker='.')
    # plt.xticks([0]+x[::10][1:],labels=x[::10])
    #
    # # plt.plot(xpair, qimp, linestyle='solid', marker='.')
    # # plt.xticks(xpair[::10], labels=xpair[::10])
    # # plt.xlim([-10,401])
    # # yfin = [173.685496,211.493637,254.754118,287.052940]
    # # plt.yticks(yfin,fontsize=14)
    # # plt.title('Evolution of installed quantity of PV', fontsize = 22)
    # plt.xlabel('Iterations', fontsize = 16)
    # # plt.ylabel('Total Cost [M€]', fontsize=16)
    # plt.ylabel('Power [GW]', fontsize = 16)
    # plt.gca().spines['top'].set_color('none')
    # plt.gca().spines['right'].set_color('none')
    # plt.show()



    # x = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    # y = [32.402582,32.402577,32.402545,32.402579,32.402578,32.402576,32.402577,32.402539,32.402577,32.402579,32.402577,32.402539,32.402577]
    # yCCGT_BATTLI = [30.363199,31.357281,32.324739,32.794023,32.402601,32.826227,32.402613,33.220123,32.402643,33.220552,32.402631,33.219352,32.402603,33.219768,32.402606,33.220252] #30it
    # ycostCCGT = [19701, 19761, 19781, 19760, 19743, 20023, 19776, 19829, 20032, 20157]
    # yalpha11 = [32.402583,32.402582,32.402582,32.402573,32.402580,32.402580,32.402573,32.402573,32.402583,32.402578,32.402580,32.402578,32.402581]
    # yalpha = [32.402582,32.402577,32.402577,32.402577,32.402577,32.402577,32.402577,32.402577,32.402574,32.402572,32.402570,32.402565,32.402557]
    # plt.figure()
    # plt.plot(x, yCCGT_BATTLI, linestyle='solid',marker='.')
    # # plt.title('Evolution of installed quantity of CCGT',fontsize = 20)
    # plt.xlabel('Iterations',fontsize =14)
    # plt.ylabel('Power [GW]',fontsize =14)
    # #Hide the right and top spines
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.show()