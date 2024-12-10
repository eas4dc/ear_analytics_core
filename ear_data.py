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

import numpy as np

from .utils import join_metric_node
from .metrics import read_metrics_configuration, metric_regex


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


def filter_invalid_gpu_series(df, gpu_metrics_regex):
    """
    Given a DataFrame with EAR data, filters those GPU
    columns that not contain some of the job's GPUs used.

    TODO: Pay attention here because this function depends directly
    on EAR's output.
    """
    return (df
            .drop(df  # Erase GPU columns
                  .filter(regex=gpu_metrics_regex).columns, axis=1)
            .join(df  # Join with valid GPU columns
                  .pipe(df_get_valid_gpu_data, gpu_metrics_regex),
                  validate='one_to_one'))  # Validate the join operation


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


def metric_timeseries_by_node(df, df_job, metric):
    """
    """
    columns = ['JOBID', 'STEPID', 'APPID', 'NODENAME']
    df_job = df_job.set_index(columns)[['START_TIME', 'END_TIME']]
    return (df
            .pivot_table(columns=columns, values=metric, index='TIMESTAMP')
            )


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
