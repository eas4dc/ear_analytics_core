######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################

"""This module has methods related to work with EARL metrics."""

from .io_api import read_configuration


def read_metrics_configuration(filename):
    """
    Return metrics configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration(filename)['runtime']['metrics']


def read_gpu_metrics_configuration(filename):
    """
    Return GPU metrics configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration(filename)['runtime']['gpu_metrics']


def metric_regex(metric, metrics_conf):
    """
    This function returns the metric's column name
    regex to be used then in a filtering action .
    """

    return metrics_conf[metric]['column_name']


def metric_step(metric, metrics_conf):
    """
    This function returns the metric's step value to be used for value
    discretisation when building a gradient timeline.
    """
    return metrics_conf[metric]['step']


def get_plottable_metrics(metrics_conf):
    """
    Filters just those metrics that can be plotted.
    """
    print(metrics_conf.keys())
    return {k: v for (k, v) in metrics_conf.items()
            if k not in ['job_step', 'node_count', 'energy', 'cpu_flops']}


def print_runtime_metrics(filename):
    runtime_config = read_configuration(filename)['runtime']

    node_metrics = runtime_config['metrics'].keys()
    print(f'Available Node metrics: {" ".join(node_metrics)}')

    gpu_metrics = runtime_config['gpu_metrics'].keys()
    print(f'Available GPU metrics: {" ".join(gpu_metrics)}')

    socket_metrics = runtime_config['socket_metrics'].keys()
    print(f'Available socket metrics: {" ".join(socket_metrics)}')
