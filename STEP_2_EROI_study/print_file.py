# -*- coding: utf-8 -*-
"""
This script modifies the input data and runs the EnergyScope model.

@author: Paolo Thiran, Matija Pavičević, Antoine Dubois
"""

import yaml
import os

import pandas as pd
import energyscope as es
from STEP_2_EROI_study.run_eroi import load_config

from energyscope.misc.utils import make_dir
from energyscope.postprocessing.utils import get_total_einv


if __name__ == '__main__':

    # Get the current working directory
    cwd = os.getcwd()
    # Print the current working directory
    print("Current working directory: {0}".format(cwd))

    # Load configuration into a dict
    config = load_config(config_fn='config.yaml')

    # Printing data #
    header = ['################################################################################',
             '##																			  ##',
             '##                     			MASTER RUN								      ##',
             '##																			  ##',
             '################################################################################',
              '## WARNING: when executed from a working directory, it is required to specify  #',
              '## the path of the .mod, .dat, and .run files from the working directory.      #',
              '################################################################################',
              '#',
              '# 1. Load standard model',
              'model '+config['ES_path']+'/ESTD_model.mod;',
              '#',
              '# 2. specify the path of the temp folder',
              'param PathName symbolic default "'+config['temp_dir']+'/output/";',
              'print PathName;'
              ]
    out_path_master_run = 'ampl_model/test.run'
    with open(out_path_master_run, mode='w', newline='') as f:
        for line in header:
            f.write(line)
            f.write('\n')
