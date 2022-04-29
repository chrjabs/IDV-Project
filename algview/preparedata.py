# Script to prepare the data to be used by the visualization from the data exported from the cluster

import pandas as pd
import numpy as np

def prepare_data(mlic_data, scep_data, scsc_data):
    run_data = pd.concat([mlic_data, scep_data, scsc_data])

    table_data = pd.DataFrame()
    for alg in run_data['alg'].unique():
        for inst in run_data['instance'].unique():
            try:
                table_data.loc[inst, alg] = run_data.loc[(
                    run_data['alg'] == alg) & (run_data['instance'] == inst), 'time'].iloc[0]
            except IndexError:
                table_data.loc[inst, alg] = np.nan
    table_data['instance'] = table_data.index
    table_data['instance type'] = 'Decision Rule Learning'
    table_data.loc[table_data['instance'].str.startswith(
        'fixed-element-prob'), 'instance type'] = 'SetCovering-EP'
    table_data.loc[table_data['instance'].str.startswith(
        'fixed-set-card'), 'instance type'] = 'SetCovering-SC'

    return run_data, table_data


if __name__ == '__main__':
    import sys

    path = './'
    if len(sys.argv) > 1:
        path = sys.argv[1] + '/'

    mlic_data = pd.read_pickle(path + 'mlic_data.pkl.gz')
    scep_data = pd.read_pickle(path + 'scep_data.pkl.gz')
    scsc_data = pd.read_pickle(path + 'scsc_data.pkl.gz')

    run_data, table_data = prepare_data(mlic_data, scep_data, scsc_data)

    run_data.to_pickle(path + 'run_data.pkl.gz')
    table_data.to_pickle(path + 'table_data.pkl.gz')
