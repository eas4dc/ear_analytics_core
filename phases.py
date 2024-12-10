######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


from numpy import around
from itertools import repeat
from pylatex import Tabular, MultiColumn
from pylatex.utils import bold

from .io_api import read_configuration


def read_phases_configuration(filename):
    """
    Return phases configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration(filename)['phases']


def phases_all_phases(phases_conf):
    """
    Given a phase configuration dict, returns a generator of tuple with the
    form (display_name, column_name). All phases are at the same level in the
    resulting generator.
    """
    for phase in phases_conf:

        if 'sub-phases' in phases_conf[phase]:
            yield from phases_all_phases(phases_conf[phase]['sub-phases'])

        yield (phases_conf[phase]['display_name'],
               phases_conf[phase]['column_name'])


def df_phases_total_time(df, phases_conf):
    """
    Returns the DataFrame `df` with an addidtional column with the total
    execution time. The DataFrame must contain EAR phase data, with involved
    nodes as Index and different phases as columns.

    `phases_conf` must be a dict with main phases configuration, i.e., keys
    must have a dict as values containing the key 'column_name', expressing
    column names of `df`.

    The total time is computed by summing the value of `df`'s columns matching
    all column names found in `phases_conf`.

    The new column is called total_time.
    """

    def get_column_name(key):
        """
        Returns the 'column_name' value of the phase keyed as `key`.
        """
        return phases_conf[key]['column_name']

    return (df
            .assign(
             total_time=lambda x: (x
                                   .filter(map(get_column_name, phases_conf))
                                   .sum(axis=1)
                                   )
             )
            )


def df_phases_phase_time_ratio(df, phases_conf):
    """
    Compute the % of time (% of the total) being on each phase of the given
    DataFrame containing phase data stored at phases_conf.
    """

    # We get a DataFrame with total exec. time
    df = df_phases_total_time(df, phases_conf)

    def compute_phase_ratio(phase):
        """
        phase is a tuple of "display_name" and df's column_name
        """

        timeperc_vs_total = (df.get(phase[1], default=0) / df['total_time']) * 100

        return (phase[0], timeperc_vs_total)

    new_columns = dict(map(compute_phase_ratio,
                           phases_all_phases(phases_conf)))

    return (df
            .assign(**new_columns)
            .pipe(lambda df: df[new_columns.keys()])
            .transform(around, decimals=2)
            )


def df_phases_to_tex_tabular(df, filepath, **kwargs):
    """
    Generates and stores a LaTeX tabular environment.

    The snippet is stored at `filepath`.tex.
    kwargs are passed at the creation of Tabular environment.

    The DataFrame is expected to have involved nodes as index, and
    phases time as columns.
    """
    header_str = '|'.join(repeat('l', df.shape[1] + 1))
    tabular_params = ''.join(['|', header_str, '|'])

    tabular = Tabular(tabular_params, **kwargs)

    tabular.add_hline()

    phase_row = ('', MultiColumn(df.shape[1], align='c|', data=bold('Phase')))
    tabular.add_row(phase_row)

    node_row = (bold('Node'),) + tuple(df.columns.values)
    tabular.add_hline()

    tabular.add_row(node_row)
    tabular.add_hline()

    for node, data in zip(df.index, df.to_numpy()):
        data_form_as_perc = map(lambda x: f'{x} %', data)  # We want %
        tabular.add_row((node,) + tuple(data_form_as_perc))

    tabular.add_hline()

    tabular.generate_tex(filepath)
