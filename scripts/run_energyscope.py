# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević
"""


import os
import pandas as pd
import energyscope as es


if __name__ == '__main__':

    # To run ES
    run_ES = True
    import_reserves = False

    # User defined
    GWP_limit = 70000  # [ktCO2-eq./year]	# Minimum GWP reduction

    # Path for data and outputs
    data_folders = ['..\\Data\\User_data', '..\\Data\\Developer_data']
    ES_path = '..\\STEP_2_Energy_Model'
    step1_output = '..\\STEP_1_TD_selection\\TD_of_days.out'

    # Reading the data
    (Eud, Resources, Technologies, End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in,
     Storage_eff_out, Time_Series) = es.import_data(data_folders)

    # Data changes
    Resources.loc['ELECTRICITY', 'avail'] = 0

    # Printing ESTD_data.dat
    all_df = (Eud, Resources, Technologies, End_uses_categories, Layers_in_out, Storage_characteristics, Storage_eff_in,
              Storage_eff_out, Time_Series)

    es.print_data(all_df, ES_path, GWP_limit)

    # Printing ESD_12TD.dat
    if import_reserves:
        # TODO: Make more elegant without so much processing and generalize country names (not only ES)
        reserves = pd.read_csv('Reserves.csv')
        reserves.index = range(1, len(reserves) + 1)
        reserves = reserves.loc[:, 'ES']
        reserves = pd.DataFrame(reserves/1000 + 3)
        reserves.rename(columns={'ES': 'end_uses_reserve'}, inplace=True)
        es.print_td_data(Time_Series, ES_path, step1_output, end_uses_reserve=reserves)
    else:
        es.print_td_data(Time_Series, ES_path, step1_output)

    # run the energy system optimisation model with AMPL
    if run_ES:
        os.chdir('..\\STEP_2_Energy_Model')
        os.system('cmd /c "ampl ESTD_main.run"')
        os.chdir('..\\scripts')

        # compute the actual average annual emission factors for each resource
        GWP_op = es.compute_gwp_op(data_folders, ES_path)
        GWP_op.to_csv('..\\STEP_2_Energy_Model\output\GWP_op.txt', sep='\t')
