import pandas as pd
import os
import energyscope as es
from pathlib import Path

def run_ESTD_UQ(sample, AMPL_path, gwp_limit):

    path = Path(__file__).parents[2]

    user_data = os.path.join(path,'Data','User_data')
    developer_data = os.path.join(path,'Data','Developer_data')
    es_path = os.path.join(path,'energyscope','STEP_2_Energy_Model')
    step1_output = os.path.join(path,'energyscope','STEP_1_TD_selection','TD_of_days.out')
    AMPL_path = r''+ AMPL_path

    s = sample[0]
    name = sample[1]
    sample_index = s[0]
    sample_dict = s[1]
    # specify the configuration
    config = {'UQ_case': name,
              'case_study': 'Run_{}'.format(sample_index),
              # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
              'printing': True,  # printing the data in ETSD_data.dat file for the optimisation problem
              'printing_td': True,  # printing the time related data in ESTD_12TD.dat for the optimisaiton problem
              'GWP_limit': gwp_limit,  # [ktCO2-eq./year]	# Minimum GWP reduction
              'import_capacity': 9.72,  # [GW] Electrical interconnections with neighbouring countries
              'data_folders': [user_data, developer_data],  # Folders containing the csv data files
              'ES_path': es_path,  # Path to the energy model (.mod and .run files)
              'step1_output': step1_output,
              # OUtput of the step 1 selection of typical days
              'all_data': dict(),
              # Dictionnary with the dataframes containing all the data in the form : {'Demand': eud, 'Resources': resources, 'Technologies': technologies, 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out, 'Storage_characteristics': storage_characteristics, 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
              'Working_directory': os.getcwd(),
              'AMPL_path': AMPL_path} # PATH to AMPL licence (to adapt by the user)

    # Reading the data
    config['all_data'] = es.import_data(config['data_folders'])



    # # Test to update uncertain parameters
    uncer_params = sample_dict
    config['all_data'] =  es.transcript_uncertainties(uncer_params,config)


    # Printing the .dat files for the optimisation problem
    es.print_data(config, 'uq')

    # Running EnergyScope
    es.run_ES(config, 'uq')

    # Example to get total cost
    total_cost = es.get_total_cost(config,'uq')

    return total_cost
    

def transcript_uncertainties(uncer_params, config):
    #TODO update with *=

    # update all_data with ref values
    config['all_data'] = es.import_data(config['data_folders'])

    # to fill the undefined uncertainty parameters
    up = {'avail_elec': 27567.3,
          'avail_waste': 17800,
          'avail_coal': 33355,
          'avail_biomass': 1,
          'c_op_electricity': 0.08433,
          'c_op_coal': 0.017657892,
          'c_op_biomass': 1,
          'c_op_biofuels': 1,
          'c_op_syn_fuels': 1,
          'c_op_hydrocarbons': 1,
          'gwp_op_ELECTRICITY': 0.206485714,
          'c_inv_pv': 870,
          'c_inv_wind_onshore': 1040,
          'c_inv_wind_offshore': 4975,
          'c_inv_dhn_hp_elec': 344.76,
          'c_inv_dec_hp_elec': 492,
          'c_inv_h2_electrolysis': 696,
          'f_max_nuc': 0,
          'f_max_pv': 59.2,
          'f_max_windon': 10,
          'f_max_windoff': 6,
          'f_max_geoelec': 0,
          'f_max_geodhn': 0,
          'elec_extra': 1,
          'ht_extra': 1,
          'sh_extra': 1,
          'ned_extra': 1,
          'freight_extra': 1,
          'passenger_extra': 1,
          'c_inv_bus': 1.0,
          'c_inv_car': 1.0,
          'c_inv_truck': 1.0,
          'c_inv_ic_prop': 1.0,
          'c_inv_e_prop': 1.0,
          'c_inv_fc_prop': 1.0,
          'cpt_pv': 1.0,
          'cpt_winds': 1.0,
          }
    for key in uncer_params:
        up[key] = uncer_params[key]

    # changing absolute value
    config['all_data']['Resources'].loc['ELECTRICITY', 'avail'] = up['avail_elec']
    config['all_data']['Resources'].loc['WASTE', 'avail'] = up['avail_waste']
    config['all_data']['Resources'].loc['COAL', 'avail'] = up['avail_coal']
    config['all_data']['Resources'].loc['WOOD', 'avail'] *= up[
        'avail_biomass']
    config['all_data']['Resources'].loc['WET_BIOMASS', 'avail'] *= up['avail_biomass']

    # Changing cost of operating:
    config['all_data']['Resources'].loc['ELECTRICITY', 'c_op'] = up['c_op_electricity']
    config['all_data']['Resources'].loc['COAL', 'c_op'] = up['c_op_coal']
    # c_op biomass
    config['all_data']['Resources'].loc['WOOD', 'c_op'] *= up['c_op_biomass']
    config['all_data']['Resources'].loc['WET_BIOMASS', 'c_op'] *= up['c_op_biomass']
    # c_op_biofuels
    config['all_data']['Resources'].loc['BIODIESEL', 'c_op'] *= up['c_op_biofuels']
    config['all_data']['Resources'].loc['BIOETHANOL', 'c_op'] *= up['c_op_biofuels']
    # c_op_syn_fuels
    config['all_data']['Resources'].loc['H2_RE', 'c_op'] *= up['c_op_syn_fuels']
    config['all_data']['Resources'].loc['GAS_RE', 'c_op'] *= up['c_op_syn_fuels']
    config['all_data']['Resources'].loc['METHANOL_RE', 'c_op'] *= up['c_op_syn_fuels']
    config['all_data']['Resources'].loc['AMMONIA_RE', 'c_op'] *= up['c_op_syn_fuels']
    # c_op_ hydrocarbons
    config['all_data']['Resources'].loc['GASOLINE', 'c_op'] *= up['c_op_hydrocarbons']
    config['all_data']['Resources'].loc['DIESEL', 'c_op'] *= up['c_op_hydrocarbons']
    config['all_data']['Resources'].loc['H2', 'c_op'] *= up['c_op_hydrocarbons']
    config['all_data']['Resources'].loc['GAS', 'c_op'] *= up['c_op_hydrocarbons']
    config['all_data']['Resources'].loc['METHANOL', 'c_op'] *= up['c_op_hydrocarbons']
    config['all_data']['Resources'].loc['AMMONIA', 'c_op'] *= up['c_op_hydrocarbons']

    config['all_data']['Resources'].loc['ELECTRICITY', 'gwp_op'] = up['gwp_op_ELECTRICITY']

    config['all_data']['Technologies'].loc['PV', 'c_inv'] = up['c_inv_pv']
    config['all_data']['Technologies'].loc['WIND_ONSHORE', 'c_inv'] = up['c_inv_wind_onshore']
    config['all_data']['Technologies'].loc['WIND_OFFSHORE', 'c_inv'] = up['c_inv_wind_offshore']
    config['all_data']['Technologies'].loc['DHN_HP_ELEC', 'c_inv'] = up['c_inv_dhn_hp_elec']
    config['all_data']['Technologies'].loc['DEC_HP_ELEC', 'c_inv'] = up['c_inv_dec_hp_elec']

    config['all_data']['Technologies'].loc['H2_ELECTROLYSIS', 'c_inv'] = up['c_inv_h2_electrolysis']

    config['all_data']['Technologies'].loc['NUCLEAR', 'f_max'] = up['f_max_nuc']
    config['all_data']['Technologies'].loc['PV', 'f_max'] = up['f_max_pv']
    config['all_data']['Technologies'].loc['WIND_ONSHORE', 'f_max'] = up['f_max_windon']
    config['all_data']['Technologies'].loc['WIND_OFFSHORE', 'f_max'] = up['f_max_windoff']
    config['all_data']['Technologies'].loc['GEOTHERMAL', 'f_max'] = up['f_max_geoelec']
    config['all_data']['Technologies'].loc['DHN_DEEP_GEO', 'f_max'] = up['f_max_geodhn']

    # demand
    config['all_data']['Demand'].loc[
        'ELECTRICITY', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'elec_extra']
    config['all_data']['Demand'].loc[
        'LIGHTING', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'elec_extra']
    config['all_data']['Demand'].loc[
        'HEAT_HIGH_T', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'ht_extra']
    config['all_data']['Demand'].loc[
        'HEAT_LOW_T_SH', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'sh_extra']
    config['all_data']['Demand'].loc[
        'MOBILITY_PASSENGER', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'passenger_extra']
    config['all_data']['Demand'].loc[
        'MOBILITY_FREIGHT', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up[
        'freight_extra']
    config['all_data']['Demand'].loc[
        'NON_ENERGY', config['all_data']['Demand'].select_dtypes(include=['number']).columns] *= up['ned_extra']

    # hourly capacity factors of RE
    config['all_data']['Time_series'].loc[:, 'PV'] *= up['cpt_pv']
    config['all_data']['Time_series'].loc[:, 'Wind_onshore'] *= up['cpt_winds']
    config['all_data']['Time_series'].loc[:, 'Wind_offshore'] *= up['cpt_winds']

    # update mobility costs
    config['all_data']['Technologies'].loc['BUS_COACH_DIESEL', 'c_inv'] *= up['c_inv_bus'] * up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['BUS_COACH_HYDIESEL', 'c_inv'] *= up['c_inv_bus'] * 0.5 * (
                up['c_inv_ic_prop'] + up['c_inv_e_prop'])
    config['all_data']['Technologies'].loc['BUS_COACH_CNG_STOICH', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                                  'BUS_COACH_CNG_STOICH', 'c_inv'] * up[
                                                                                  'c_inv_bus'] * up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['BUS_COACH_FC_HYBRIDH2', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                                   'BUS_COACH_FC_HYBRIDH2', 'c_inv'] * \
                                                                               up['c_inv_bus'] * up['c_inv_fc_prop']

    config['all_data']['Technologies'].loc['CAR_GASOLINE', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                          'CAR_GASOLINE', 'c_inv'] * up['c_inv_car'] * \
                                                                      up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['CAR_DIESEL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                        'CAR_DIESEL', 'c_inv'] * up['c_inv_car'] * up[
                                                                        'c_inv_ic_prop']
    config['all_data']['Technologies'].loc['CAR_NG', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                    'CAR_NG', 'c_inv'] * up['c_inv_car'] * up[
                                                                    'c_inv_ic_prop']
    config['all_data']['Technologies'].loc['CAR_METHANOL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                          'CAR_METHANOL', 'c_inv'] * up['c_inv_car'] * \
                                                                      up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['CAR_HEV', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                     'CAR_HEV', 'c_inv'] * up['c_inv_car'] * 0.5 * (
                                                                         up['c_inv_ic_prop'] + up['c_inv_e_prop'])
    config['all_data']['Technologies'].loc['CAR_PHEV', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                      'CAR_PHEV', 'c_inv'] * up['c_inv_car'] * 0.5 * (
                                                                          up['c_inv_ic_prop'] + up['c_inv_e_prop'])
    config['all_data']['Technologies'].loc['CAR_BEV', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                     'CAR_BEV', 'c_inv'] * up['c_inv_car'] * up[
                                                                     'c_inv_e_prop']
    config['all_data']['Technologies'].loc['CAR_FUEL_CELL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                           'CAR_FUEL_CELL', 'c_inv'] * up['c_inv_car'] * \
                                                                       up['c_inv_fc_prop']

    config['all_data']['Technologies'].loc['BOAT_FREIGHT_DIESEL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                                 'BOAT_FREIGHT_DIESEL', 'c_inv'] * up[
                                                                                 'c_inv_ic_prop']
    config['all_data']['Technologies'].loc['BOAT_FREIGHT_NG', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                             'BOAT_FREIGHT_NG', 'c_inv'] * up[
                                                                             'c_inv_ic_prop']
    config['all_data']['Technologies'].loc['BOAT_FREIGHT_METHANOL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                                   'BOAT_FREIGHT_METHANOL', 'c_inv'] * \
                                                                               up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['TRUCK_DIESEL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                          'TRUCK_DIESEL', 'c_inv'] * up['c_inv_truck'] * \
                                                                      up['c_inv_ic_prop']
    config['all_data']['Technologies'].loc['TRUCK_FUEL_CELL', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                             'TRUCK_FUEL_CELL', 'c_inv'] * up[
                                                                             'c_inv_truck'] * up['c_inv_fc_prop']
    config['all_data']['Technologies'].loc['TRUCK_ELEC', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                        'TRUCK_ELEC', 'c_inv'] * up['c_inv_truck'] * up[
                                                                        'c_inv_e_prop']
    config['all_data']['Technologies'].loc['TRUCK_NG', 'c_inv'] = config['all_data']['Technologies'].loc[
                                                                      'TRUCK_NG', 'c_inv'] * up['c_inv_truck'] * up[
                                                                      'c_inv_ic_prop']
    config['all_data']['Technologies'].loc['TRUCK_METHANOL', 'c_inv'] *= up['c_inv_truck'] * up['c_inv_ic_prop']

    return config['all_data']

