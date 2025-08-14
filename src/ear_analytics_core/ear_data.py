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

from .utils import join_metric_node
from .metrics import read_metrics_configuration, metric_regex
from .console import warning, error


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
    return(df
           .pivot_table(values=metric,
                        index='TIMESTAMP', columns='NODENAME')
           .bfill()
           .ffill()
           .pipe(join_metric_node)
           .agg(np.sum, axis=1)
           )


def filter_and_query(df, rules):
    """
    Returns the resulting DataFrame of applying filtering rules to the passed
    dataframe `df`. The function first performs a pre-filtering of the
    dataframe based on column labels and then uses the pandas.DataFrame.query
    method to query for specific row values.

    Rules are configured in `rules` as a dict with the following
    <key, value> pairs:
        - 'filter': <A dictionary with a pandas.DataFrame.filter's kwarg. This
          key is optional and it is used to call the function to the passed
          dataframe before querying.
        - 'expr': 'A valid string to be passed to pandas.DataFrame.query
          function called on the filtered dataframe. This field is required
          if and only if the next key is not found.
        - 'criteria': 'A string with a valid query operation to be concatenated
          with every column of the pre-filtered dataframe.'
        - 'join': 'A string with conditional operator, e.g., and, or.'

    (Optional) Pre-filtering consists of calling pandas.DataFrame.filter on
    the passed dataframe and using rules' 'filter' dictionary as kwarg,
    i.e., df.filter(**rules['filter']).

    If `rules` contains 'expr' string, pandas.DataFrame.query is called
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
    else:
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
        else:
            # Create the query to check whether some row matches the
            # alert criteria
            # Format: <column> <criteria> <join> <column> <criteria>...
            join = rules.get('join', 'or')
            expr = (f' {join} '
                    .join([f'`{col}` {criteria}'
                           for col in df.columns])
                    )
    return expr


def read_apps_data(data_file: str)-> pd.DataFrame:
    """
    Read application signatures from a CSV file.

    Parameters:
        data_file (str): Path to the app CSV file, which can be generated by:
            - the command "eacct -j job_id -l -c"
            - or by using the tag "--ear-user-db"

    Returns:
        pd.DataFrame: DataFrame containing a selection of the application
        signatures tha be used in cpu models, projections, polices, etc..
    """
    df = pd.read_csv(data_file, sep=';')

    # don't take into account the bash step
    # UINT_MAX = ctypes.c_uint(-5).value
    # TODO verify it in other systems (Lluis?)
    UINT_MAX = 4294967291
    df = df.loc[df.STEPID != UINT_MAX]

    # compute node_non_gpu_power if GPUs are used ()
    ngpus = 0
    for c in df.columns.to_list():
        if re.match("GPU._UTIL.", c) != None:
            ngpus += 1

    # add extra cols
    df["POWER_W"] = 0
    df["GPU_POWER_W"] = 0
    for i in range(ngpus):
        df["GPU_POWER_W"] += df[f"GPU{i}_POWER_W"]
    df["POWER_W"] = df["DC_NODE_POWER_W"] - df["GPU_POWER_W"]

    df["ENERGY_J"] = df["POWER_W"]*df["TIME_SEC"]

    # rename the field CPU-GFLOPS (old DB version)
    if "CPU-GFLOPS" in df.columns:
        df.rename(columns={'CPU-GFLOPS': 'CPU_GFLOPS'}, inplace=True)

    # To compute VPI
    df.rename(columns={"SPOPS_256": "FLOPS_256#F"}, inplace=True)
    df.rename(columns={"SPOPS_512": "FLOPS_512#F"}, inplace=True)
    df.rename(columns={"DP_256"   : "FLOPS_256#D"}, inplace=True)
    df.rename(columns={"DPOPS_512": "FLOPS_512#D"}, inplace=True)

    # signatures we intersred in (can be expanded)
    sigs_list = [
        "NODENAME", "JOBID", "STEPID", "APPID", "JOBNAME","POLICY", "POLICY_TH",
        "DEF_FREQ_KHZ", "AVG_CPUFREQ_KHZ", "AVG_IMCFREQ_KHZ",
        "TIME_SEC","POWER_W","ENERGY_J", "DC_NODE_POWER_W", "GPU_POWER_W",
        "DRAM_POWER_W","PCK_POWER_W", "CPI", "TPI", "MEM_GBS", "IO_MBS",
        "PERC_MPI", "CPU_GFLOPS", "INSTRUCTIONS",
        "FLOPS_256#F", "FLOPS_512#F", "FLOPS_256#D", "FLOPS_512#D"
    ]
    df = df[sigs_list]
    df = df.reset_index(drop=True)

    return df
