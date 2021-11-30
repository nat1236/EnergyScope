# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model

@author: Paolo Thiran, Matija Pavičević, Antoine Dubois
"""


import yaml
import os

import energyscope as es


def load_config(config_fn: str):

    # Load parameters
    cfg = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Extend path
    for param in ['case_studies_dir', 'user_data', 'developer_data', 'temp_dir', 'ES_path', 'step1_output']:
        cfg[param] = os.path.join(cfg['energyscope_dir'], cfg[param])
    return cfg


if __name__ == '__main__':

    # Load configuration
    config = load_config('config.yaml')

    # config['all_data'] = es.import_data(config['data_folders'])
    all_data = import_data(config['user_data'], config['developer_data'])

    # Optimal solution
    if not os.path.isdir(f"{config['case_studies_dir']}/{config['case_study_name']}"):

        # Saving .dat files
        out_path = f"{config['temp_dir']}/ESTD_data.dat"
        es.print_estd(out_path, all_data, config["import_capacity"], config["GWP_limit"])
        out_path = f"{config['temp_dir']}/ESTD_12TD.dat"
        es.print_12td(out_path, all_data['Time_series'], config["step1_output"])

        # Running EnergyScope
        cs = f"{config['case_studies_dir']}/{config['case_study_name']}"
        run_fn = f"{config['ES_path']}/optimal_cost.run"
        es.run_energyscope(cs, run_fn, config['AMPL_path'], config['temp_dir'])
        exit()
        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}/output/"
        es.drawSankey(path=f"{output_dir}/sankey")

    if 1:
        # Optimal solution in terms of EINV
        # Printing the .dat files for the optimisation problem
        out_path = f"{config['temp_dir']}/ESTD_data.dat"
        es.print_estd(out_path, all_data, config["import_capacity"], config["GWP_limit"])

        # Run the model
        cs = f"{config['case_studies_dir']}/{config['case_study_name']}_einv/"
        run_fn = f"{config['ES_path']}/optimal_einv.run"
        es.run_energyscope(cs, run_fn, config['AMPL_path'], config['temp_dir'])

        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_einv/output/"
        es.drawSankey(path=f"{output_dir}/sankey")

    # Get total cost
    cost = es.get_total_cost(f"{config['case_studies_dir']}/{config['case_study_name']}")

    # Get epsilon invariant
    epsilons = [0.0125]
    for epsilon in epsilons:

        print("Run for epsilon", epsilon)

        # Printing the .dat files for the optimisation problem
        out_path = f"{config['temp_dir']}/ESTD_data_epsilon.dat"
        es.print_estd(out_path, all_data, config["import_capacity"], config["GWP_limit"])
        # Add specific elements
        es.newline(out_path)
        es.print_param("TOTAL_COST_OP", cost, "Optimal cost of the system", out_path)
        es.newline(out_path)
        es.print_param("EPSILON", epsilon, "Epsilon value", out_path)

        # newline(out_path)
        # technologies_to_minimize = ["WIND_ONSHORE", "WIND_OFFSHORE"]
        # print_set(technologies_to_minimize, "TECHNOLOGIES_TO_MINIMIZE", out_path)

        # Run the model
        cs = f"{config['case_studies_dir']}/{config['case_study_name']}_epsilon_{epsilon}/"
        run_fn = f"{config['ES_path']}/epsilon.run"
        es.run_energyscope(cs, run_fn, config['AMPL_path'], config['temp_dir'])

        # Example to print the sankey from this script
        output_dir = f"{config['case_studies_dir']}/{config['case_study_name']}_epsilon_{epsilon}/output/"
        es.drawSankey(path=f"{output_dir}/sankey")
