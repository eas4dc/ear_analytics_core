######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


from .io_api import read_configuration


def ear2prv_config(filename):
    return read_configuration(filename)['ear2prv']


def ear2prv_job_config(ear2prv_config):
    return ear2prv_config['job']


def ear2prv_loop_config(ear2prv_config):
    return ear2prv_config['loop']
