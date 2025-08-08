######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################


import logging
from rich.console import Console
from rich.logging import RichHandler

# Create a shared Console instance
console = Console()

# Set up the RichHandler
handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    show_path=False,    # Hide file:line
    show_time=False,    # Hide timestamp
    markup=True
)

# Set up the logger
logging.basicConfig(
    level=logging.INFO,      # Only logs INFO and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(message)s",    # You can customize this further if needed
    datefmt="[%X]",
    handlers=[handler]
)

# Create a named logger instance
logger = logging.getLogger("richLogger")
