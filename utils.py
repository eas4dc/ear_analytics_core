######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


""" Util functions. """


from functools import reduce

from .io_api import read_configuration


def filter_df(data_f, **kwargs):
    """
    Filters the DataFrame `data_f`. **kwargs keys indicate the DataFrame
    columns you want to filter by, and keys are values.
    """

    expr = ' and '.join([f'{k} == @kwargs.get("{k}")'
                         for k in kwargs if kwargs[k] is not None
                         and k in data_f.columns])
    if expr == '':
        return data_f

    return data_f.query(expr)


def list_str(values):
    """
    Split the string `values` using comma as a separator.
    """
    return values.split(',')


def join_metric_node(df):
    "Given a DataFrame df, returns it flattening it's columns MultiIndex."
    df.columns = df.columns.to_flat_index()
    return df


def read_job_data_config(filename):
    return read_configuration(filename)['job']


def read_loop_data_config(filename):
    return read_configuration(filename)['loop']


def function_compose(*funcs):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, funcs)
