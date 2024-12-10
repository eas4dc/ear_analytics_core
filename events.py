######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################
"""This module contains methods related to EAR special events."""


from .io_api import read_configuration


def read_events_configuration(filename):
    """
    Returns events configuration stored in `filename`,
    which is a file in JSON format.
    """
    return read_configuration(filename)['events']
