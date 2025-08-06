######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################



from rich.console import Console

console = Console()


def warning(*args, **kwargs):
    console.print('[magenta][WARNING][/]', *args, **kwargs)


def error(*args, **kwargs):
    console.print('[bright_red][ERROR][/]', *args, **kwargs)


def info(*args, **kwargs):
    console.print('[cyan][INFO][/]', *args, **kwargs)
