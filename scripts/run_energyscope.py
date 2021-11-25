# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Antoine Dubois
"""


from pathlib import Path
import energyscope as es
from energyscope.preprocessing.print_data import print_12td, print_estd, newline, print_param, print_set
from subprocess import CalledProcessError, run
import shutil

import os

if __name__ == '__main__':

    # define path
    path = Path(__file__).parents[1]
    user_data = os.path.join(path, 'Data', 'User_data')
    developer_data = os.path.join(path, 'Data', 'Developer_data')
    es_path = os.path.join(path, 'STEP_2_Energy_Model')
    step1_output = os.path.join(path, 'STEP_1_TD_selection', 'TD_of_days.out')
    temp_dir = Path(__file__).parent.absolute().joinpath('../temp')

    # specify the configuration
    # TODO: add this in a config file ?
    config = {
        'case_studies_dir': Path(__file__).parent.absolute().joinpath('../case_studies/'),
        # Name of the case study. The outputs will be printed into : config['ES_path']+'\output_'+config['case_study']
        'case_study_name': 'test_test',
        # printing the data in ETSD_data.dat file for the optimisation problem
        'printing': True,
        # printing the time related data in ESTD_12TD.dat for the optimisation problem
        'printing_td': True,
        # [ktCO2-eq./year]	# Minimum GWP reduction
        'GWP_limit': 1e+7,
        # [GW] Electrical interconnections with neighbouring countries
        'import_capacity': 9.72,
        # Folders containing the csv data files
        'data_folders': [user_data, developer_data],
        # Directory to save temporary output files
        'temp_dir': temp_dir,
        # Path to the energy model (.mod and .run files)
        'ES_path': es_path,
        # Output of the step 1 selection of typical days
        'step1_output': step1_output,
        # Dictionnary with the dataframes containing all the data in the form :
        # {'Demand': eud, 'Resources': resources, 'Technologies': technologies,
        # 'End_uses_categories': end_uses_categories, 'Layers_in_out': layers_in_out,
        # 'Storage_characteristics': storage_characteristics,
        # 'Storage_eff_in': storage_eff_in, 'Storage_eff_out': storage_eff_out, 'Time_series': time_series}
        'all_data': dict(),
        # 'Working_directory': os.getcwd()
        # PATH to AMPL licence (to adapt by the user)
        'AMPL_path': '/home/duboisa1/ampl_linux-intel64'
    }

    # Reading the data
    config['all_data'] = es.import_data(config['data_folders'])

    # Optimal solution
    # Printing the .dat files for the optimisation problem
    if not os.path.isdir(f"{config['case_studies_dir']}/{config['case_study_name']}"):
        out_path = f"{config['temp_dir']}/ESTD_data.dat"
        print_estd(out_path, config['all_data'], config["import_capacity"], config["GWP_limit"])
        out_path = f"{config['temp_dir']}/ESTD_12TD.dat"
        print_12td(out_path, config['all_data']['Time_series'], config["step1_output"])
        # Running EnergyScope
        es.run_ES(config)

        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}/output/"
        es.drawSankey(path=f"{output_dir}/sankey")

    if 0:
        # Optimal solution in terms of CO2
        # Printing the .dat files for the optimisation problem
        out_path = f"{config['temp_dir']}/ESTD_data.dat"
        print_estd(out_path, config['all_data'], config["import_capacity"], config["GWP_limit"])
        # Run the model
        try:
            run(f"ampl {config['ES_path']}/ESTD_main_gwp.run", shell=True, check=True)
        except CalledProcessError as e:
            print("The run didn't end normally.")
            print(e)
            exit()
        # Copy temporary results to case studies directory
        case_study_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_gwp/"
        shutil.copytree('../temp', case_study_dir)
        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_gwp/output/"
        es.drawSankey(path=f"{output_dir}/sankey")

    # Get total cost
    cost = es.get_total_cost(f"{config['case_studies_dir']}/{config['case_study_name']}")

    # Get epsilon invariant
    epsilons = []
    for epsilon in epsilons:

        # Printing the .dat files for the optimisation problem
        out_path = f"{config['temp_dir']}/ESTD_data_epsilon.dat"
        print_estd(out_path, config['all_data'], config["import_capacity"], config["GWP_limit"])
        # Add specific elements
        newline(out_path)
        print_param("TOTAL_COST_OP", cost, "Optimal cost of the system", out_path)
        newline(out_path)
        print_param("EPSILON", epsilon, "Epsilon value", out_path)
        # newline(out_path)
        # technologies_to_minimize = ["WIND_ONSHORE", "WIND_OFFSHORE"]
        # print_set(technologies_to_minimize, "TECHNOLOGIES_TO_MINIMIZE", out_path)

        # Run the model
        # running ES
        try:
            run(f"ampl {config['ES_path']}/ESTD_main_epsilon.run", shell=True, check=True)
        except CalledProcessError as e:
            print("The run didn't end normally.")
            print(e)
            exit()

        # Copy temporary results to case studies directory
        case_study_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_epsilon_{epsilon}/"
        shutil.copytree('../temp', case_study_dir)

        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_epsilon_{epsilon}/output/"
        es.drawSankey(path=f"{output_dir}/sankey")
