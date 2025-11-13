######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


""" This module contains functions that can be applied
to a DataFrame contained known EAR data. """

import re
import numpy as np
import pandas as pd

from returns.result import Result, Failure, Success

from .utils import join_metric_node
from .metrics import read_metrics_configuration, metric_regex
from .console import warning


def df_filter_invalid_gpu_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing EAR data, returns a copy of it without all
    those invalid GPU columns.

    An invalid GPU column is a column with data of a GPU x, for which EAR did
    not report any GPUx_POWER_W reading.
    """
    # The regex is precompiled since it is searched multiple times.
    gpu_colname_pattern = re.compile(r'GPU(\d)_POWER_W')

    def return_gpupwr_index(gpupwr_colname: str) -> str:
        """
        Given a str of the form r'GPUx_POWER_W', returns the x part, if
        found. Otherwise, returns None.

        The regular expression pattern is taken from gpu_colname_pattern.
        """
        match = re.fullmatch(gpu_colname_pattern, gpupwr_colname)
        if match:
            try:
                return match.group(1)
            except IndexError:
                return None
        else:
            return None

    invalid_gpu_indices = filter(None,  # Filter all elements which are false
                                 map(return_gpupwr_index,
                                     df_get_invalid_gpupower_cols(df))
                                 )

    indices_or = '|'.join(invalid_gpu_indices)
    filter_str = fr'GPU({indices_or})_\w+'
    return df.drop(columns=df.filter(regex=filter_str).columns)


def df_get_invalid_gpupower_cols(df: pd.DataFrame) -> pd.Index:
    """
    Given a pd.DataFrame with EAR data, returns those columns which are
    actually invalid GPU Power data.

    Invalid GPU power data is all those GPUx_POWER_W columns of the DataFrame
    that are full of zero values.
    """
    return (df
            .filter(regex=r'GPU\d_POWER_W')
            .mask(lambda x: x != 0)  # All non-zero as nan
            .dropna(axis=1, how='all')  # Drop nan columns
            .columns
            )


def df_get_valid_gpu_data(df, gpu_metrics_regex):
    """
    Returns a DataFrame with only valid GPU data.

    Valid GPU data is all those GPU columns of the DataFrame
    that are not full of zeroes values.
    """
    return (df
            .filter(regex=gpu_metrics_regex)
            .mask(lambda x: x == 0)  # All 0s as nan
            .dropna(axis=1, how='all')  # Drop nan columns
            .mask(lambda x: x.isna(), other=0))  # Return to 0s


def df_has_gpu_data(df, gpu_metrics_regex):
    """
    Returns whether the DataFrame df has valid GPU data.
    """
    return not df.pipe(df_get_valid_gpu_data, gpu_metrics_regex).empty


def filter_invalid_gpu_series(df, gpu_metrics_regex):
    """
    Given a DataFrame with EAR data, filters those GPU
    columns that not contain some of the job's GPUs used.
    """
    return (df
            .drop(df  # Erase GPU columns
                  .filter(regex=gpu_metrics_regex).columns, axis=1)
            .join(df  # Join with valid GPU columns
                  .pipe(df_get_valid_gpu_data, gpu_metrics_regex),
                  validate='one_to_one'))  # Validate the join operation


# TODO: This function is not called anywhere
def df_gpu_node_metrics(df, conf_fn):
    """
    Given a DataFrame `df` with EAR data and a configuration filename `conf_fn`
    Returns a copy of the DataFrame with new columns showing node-level GPU
    metrics.
    """
    metrics_conf = read_metrics_configuration(conf_fn)

    gpu_pwr_regex = metric_regex('gpu_power', metrics_conf)
    gpu_freq_regex = metric_regex('gpu_freq', metrics_conf)
    gpu_memfreq_regex = metric_regex('gpu_memfreq', metrics_conf)
    gpu_util_regex = metric_regex('gpu_util', metrics_conf)
    gpu_memutil_regex = metric_regex('gpu_memutil', metrics_conf)

    gr_active_regex = metric_regex('dcgmi_gr_engine_active', metrics_conf)

    return (df
            .assign(
                tot_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_regex)
                                        .sum(axis=1)),  # Agg. GPU power

                avg_gpu_pwr=lambda x: (x.filter(regex=gpu_pwr_regex)
                                        .mean(axis=1)),  # Avg. GPU power

                avg_gpu_freq=lambda x: (x.filter(regex=gpu_freq_regex)
                                        .mean(axis=1)),  # Avg. GPU freq

                avg_gpu_memfreq=lambda x: (x.filter(regex=gpu_memfreq_regex)
                                           .mean(axis=1)),  # Avg. GPU mem freq

                avg_gpu_util=lambda x: (x.filter(regex=gpu_util_regex)
                                        .mean(axis=1)),  # Avg. % GPU util

                avg_gpu_memutil=lambda x: (x.filter(regex=gpu_memutil_regex)
                                           .mean(axis=1)),  # Avg %GPU mem util
                avg_gr_engine_active=lambda x: (x.filter(regex=gr_active_regex)
                                                .mean(axis=1))
            ))


def metric_agg_timeseries(df, metric):
    """
    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    return (df
            .pivot_table(values=metric,
                         index='TIMESTAMP', columns='NODENAME')
            .bfill()
            .ffill()
            .pipe(join_metric_node)
            .agg(np.sum, axis=1)
            )


def filter_batch_step(ear_df: pd.DataFrame) -> Result[pd.DataFrame, str]:
    """
    This function returns the DataFrame `ear_df` without any SLURM batch step
    if it has some. It spects the DataFrame containing a column called
    'STEPID'. If not encountered, returns a copy of the input argument.

    Parameters
    ----------
    ear_df: A DataFrame containing EAR signature data. It must have a column
            named 'STEPID'.

    Return
    ------
    A Result type with a Success(pd.DataFrame) with the passed DataFrame
    filtered or a Failure(str) indicating that the STEPID column does not exist
    in the passed argument.
    """
    if 'STEPID' in ear_df.columns:
        return Success(ear_df.loc[ear_df['STEPID'] != 4294967291])
    else:
        return Failure('STEPID not in data.')


def filter_and_query(df, rules):
    """
    Returns the resulting DataFrame of applying filtering rules to the passed
    dataframe `df`. The function first performs a pre-filtering of the
    dataframe based on column labels and then uses the pd.DataFrame.query
    method to query for specific row values.

    Rules are configured in `rules` as a dict with the following
    <key, value> pairs:
        - 'filter': <A dictionary with a pd.DataFrame.filter's kwarg. This
          key is optional and it is used to call the function to the passed
          dataframe before querying.
        - 'expr': 'A valid string to be passed to pd.DataFrame.query
          function called on the filtered dataframe. This field is required
          if and only if the next key is not found.
        - 'criteria': 'A string with a valid query operation to be concatenated
          with every column of the pre-filtered dataframe.'
        - 'join': 'A string with conditional operator, e.g., and, or.'

    (Optional) Pre-filtering consists of calling pd.DataFrame.filter on
    the passed dataframe and using rules' 'filter' dictionary as kwarg,
    i.e., df.filter(**rules['filter']).

    If `rules` contains 'expr' string, pd.DataFrame.query is called
    directly. Otherwise, the expression is build as:
        <column..0> <criteria> [<join> <column..1> <criteria>]*
    where 'join' operator is used just when more than one column is found in
    (maybe pre-filtered) dataframe and it is the 'or' string if `rules` does
    not provide it.
    """
    # If the configuration does not have the 'filter' field, we apply
    # the filter which returns the identical df
    prefilter = rules.get('filter', {'items': df.columns})
    df_filtered = df.filter(**prefilter)

    if not df_filtered.empty:
        expr = create_ear_dataframe_query(df_filtered, rules)
        return df_filtered.query(expr), expr
    return df_filtered, None


def create_ear_dataframe_query(df, rules):
    """Support function for creating the query usied by
    ear_dataframe_filter_and_query"""
    expr = rules.get('expr', None)
    if expr is None:
        try:
            criteria = rules['criteria']
        except KeyError as e:
            warning(f'The rule has not {e} field.')
            return None
        # Create the query to check whether some row matches the
        # alert criteria
        # Format: <column> <criteria> <join> <column> <criteria>...
        join = rules.get('join', 'or')
        expr = (f' {join} '
                .join([f'`{col}` {criteria}'
                       for col in df.columns])
                )
    return expr

