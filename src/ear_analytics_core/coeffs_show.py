######################################################################
# Copyright (c) 2024 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################

import csv
import struct
import pandas as pd

from .logger import logger

__all__ = ['cpu_coeffs']


def cpu_coeffs(binary_file: str, save_csv: bool = False) -> pd.DataFrame:
    """
    Converting the binary CPU coefficients into pandas DataFrame and save file to a CSV file.

    Parameters
    ----------
        binary_file : str
            Path to the binary file containing CPU coefficients.
        save_csv : bool, optional.
            If true saves the DataFrame to a CSV file named "<binary_file>.csv".
            Default: False.

    Returns
    -------
        pd.DataFrame: DataFrame containing the CPU coefficients.
    """

    # Define binary record structure: 2 longs (8 bytes), 1 int  and 4 doublse (8 bytes)
    # <=> to 'typedef struct coefficient' in the EAR Library
    """
    // from <common/types/coefficient.h>
    typedef struct coefficient
    {
        unsigned long pstate_ref;
        unsigned long pstate;
        unsigned int available;
        /* For power projection */
        double A;
        double B;
        double C;
        /* For CPI projection */
        double D;
        double E;
        double F;
    } coefficient_t;
    """
    record_format = 'QQidddddd'
    record_size = struct.calcsize(record_format)

    print(f"Reading the binary file: {binary_file}")

    df = None
    columns = ['FROM', 'TO', 'Available', 'A', 'B', 'C', 'D', 'E', 'F']
    with open(binary_file, 'rb') as bin_file:
        while True:
            chunk = bin_file.read(record_size)
            if not chunk:
                print('No more chunks, ending.')
                break

            if len(chunk) < record_size:
                print(
                    f'Incomplete chunk of size {
                        len(chunk)} found, skipping.')
                break

            coeffs_row = struct.unpack(record_format, chunk)
            new_row_df = pd.DataFrame([coeffs_row], columns=columns)
            if df is None:
                df = new_row_df
            else:
                df = pd.concat([df, new_row_df], ignore_index=True)

    # Save in the csv file
    if save_csv:
        csv_file = f"{binary_file}.csv"
        df.to_csv(csv_file, sep=";", index=False)
        logger.info(f"(The csv file for the coefficients: {csv_file} is saved.")

    return df
