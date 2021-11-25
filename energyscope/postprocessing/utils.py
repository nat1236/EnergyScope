import pandas as pd
import os


def get_total_cost(output_path):
    costs = pd.read_csv(f"{output_path}/output/cost_breakdown.txt", index_col=0, sep='\t')

    return costs.sum().sum()


def get_total_gwp(output_path):
    gwp = pd.read_csv(f"{output_path}/output/gwp_breakdown.txt", index_col=0, sep='\t')

    return gwp.sum().sum()
