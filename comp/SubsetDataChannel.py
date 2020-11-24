# -*- coding: utf-8 -*-
"""
Subset a datachannel for all samples found in LAM's Samples-folder to create a new datachannel. The output datafile
is saved to the respective Sample-folders and should be immediately usable in LAM-analysis.

Vars:
----
    analysis_path : pathlib.Path
        Full path to the root of the analysis directory.

    data_channel : str
        Name of the file that will be filtered for all samples.

    filtering : [str]
        Expression patterns to filter the data. If multiple expressions are given, all expressions are performed
        consecutively, i.e. all are performed to produce the final data subset.
        The expressions should be given in pattern: '{column_label} {operator} {value}'
        E.g. 'Area > 100', 'Nearest_Dist_DAPI <= 10', 'Volume != 200'
        For additional information, see:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html

    output_file : str
        The name of the data file. The file will be saved in the sample's folder in "Analysis Data\Samples".


Created on Mon Mar 16 16:34:33 2020
@author: Arto I. Viitanen
"""

import pandas as pd
import pathlib as pl

# Give path to the analysis root directory:
analysis_path = pl.Path(r'E:\Code_folder\ALLSTATS')

# Give name of data-file to subset
data_channel = 'DAPI.csv'

# Give filtering expressions to subset the data.
filtering = ['Area > 100', 'Nearest_Dist_DAPI <= 10']

# Give name for output-file.
output_file = 'DAPI_subset.csv'


def subset_data(paths: [pl.Path], file: str, filters: [str], filename: str):
    """Create a subset of a datachannel based on filter expressions."""
    for path in paths:  # Loop all sample-folders
        print(path.name)
        save_path = path.joinpath(filename)
        filepath = path.joinpath(file)
        data = pd.read_csv(filepath)
        # Perform subsetting
        subdata = data_query(data, filters)
        if subdata.empty:
            continue
        # Save subset
        subdata.to_csv(save_path, index=False)


def data_query(data: pd.DataFrame, filters: [pl.Path]) -> pd.DataFrame:
    """Successively query all given expressions to produce a subset of a DataFrame."""
    for filter_expr in filters:
        try:
            data = data.query(filter_expr)
        except (KeyError, ValueError, pd.core.computation.ops.UndefinedVariableError) as err:
            print(f'Error: {err}')
            return pd.DataFrame()
    return data


if __name__ == '__main__':
    samples_dir = analysis_path.joinpath('Analysis Data', 'Samples')
    sample_paths = [p for p in samples_dir.iterdir() if p.is_dir()]
    subset_data(sample_paths, data_channel, filtering, output_file)
