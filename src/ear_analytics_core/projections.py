######################################################################
# Copyright (c) 2025 Energy Aware Solutions, S.L
#
# This program and the accompanying materials are made
# available under the terms of the Eclipse Public License 2.0
# which is available at https://www.eclipse.org/legal/epl-2.0/
#
# SPDX-License-Identifier: EPL-2.0
#######################################################################

import os
import pandas as pd
import numpy as np
from pathlib import Path

from .logger import logger
from .ear_data import read_apps_data

__all__ = ["compute_projections"]


def apply_projection(
        dfc_ref: pd.DataFrame,
        sig_ref: dict[str, float],
        model_name: str
    ) -> tuple[list[float], list[float]]:
    """
    Computes power and time projections using a specified model.

    Parameters
    ----------
        dfc_ref : pd.DataFrame)
            Model coefficients for one source frequency.
        sig_ref : dict[str, float]
            Signature values keyed by name.
        model_name : str
            Model type from the list 'basic', 'basic_perc', 'spr_imc_perc'

    Returns
    -------
        tuple[list[float], list[float]]: Projected CPU Power and Time.
    """
    model_name = model_name.lower()

    power_proj = None
    time_proj = None

    if model_name in {"basic", "basic_perc"}:
        power_ref = sig_ref["POWER_W"]
        time_ref = sig_ref["TIME_SEC"]
        tpi_ref = sig_ref["TPI"]
        cpi_ref = sig_ref["CPI"]

        # Power projection (same for both models)
        power_proj = dfc_ref.A * power_ref + dfc_ref.B * tpi_ref + dfc_ref.C

        if model_name == "basic":
            cpi_proj = dfc_ref.D * cpi_ref + dfc_ref.E * tpi_ref + dfc_ref.F
            cpi_norm = cpi_proj / cpi_ref
            freq_norm = dfc_ref.FROM / dfc_ref.TO
            time_proj = cpi_norm.mul(freq_norm) * time_ref
        else:  # basic_perc
            time_proj = time_ref * (dfc_ref.D * cpi_ref + dfc_ref.E * tpi_ref + dfc_ref.F)

    elif model_name == "spr_imc_perc":
        power_ref = sig_ref["POWER_W"]
        time_ref = sig_ref["TIME_SEC"]
        imcf_ref = sig_ref["AVG_IMCFREQ_KHZ"]
        power_proj = dfc_ref.A * power_ref + dfc_ref.B * imcf_ref + dfc_ref.C
        time_proj = time_ref * (dfc_ref.E * imcf_ref + dfc_ref.F)
    else:
        logger.error(f"Unsupported model: '{model_name}'")

    return power_proj.to_list(), time_proj.to_list()


def compute_basic_projections(
        df_apps: pd.DataFrame,
        df_coeffs: pd.DataFrame,
        model_name: str
    )-> pd.DataFrame:
    """
    compute all projections (time and power) using the basic or basic_perc model
    """

    freqs_list = df_coeffs.FROM.unique()

    df_apps["FROM_FREQ_KHZ"] = df_apps["DEF_FREQ_KHZ"]

    # Only update FROM_FREQ_KHZ rows where POLICY == "min_energy"
    # Get the coloset DEF FREQUECNY to AVG CPU FREQ
    df_apps.loc[df_apps["POLICY"] == "min_energy", "FROM_FREQ_KHZ"] = df_apps.loc[
        df_apps["POLICY"] == "min_energy", "AVG_CPUFREQ_KHZ"
    ].apply(lambda x: freqs_list[np.abs(freqs_list - x).argmin()])

    # target dataframe containg projections
    cols = ["NODENAME", "JOBID", "STEPID", "JOBNAME", "POLICY", "POLICY_TH",
            "DEF_FREQ_KHZ", "AVG_CPUFREQ_KHZ", "CPI", "TPI", "MEM_GBS",
            "FROM_FREQ_KHZ", "TO_FREQ_KHZ",
            "TIME_SEC", "PROJ_TIME_SEC", "POWER_W", "PROJ_POWER_W",
            "ENERGY_J", "PROJ_ENERGY_J"]

    # Dataframe to return
    df = pd.DataFrame(columns=cols)
    shared_cols = df.columns.intersection(df_apps.columns)

    dg = df_apps.groupby(["FROM_FREQ_KHZ"])
    for _, group in dg:
        from_freq = group.FROM_FREQ_KHZ.unique().item()
        dfc =  df_coeffs.query(f"FROM=={from_freq} and Available==1")
        dfc.reset_index(drop=True)

        if dfc.empty:
            logger.warning(f"No available coeffs. from the frequecy {freq}")
        else:
            for idx, row in group.iterrows():
                sig_ref = {"CPI": row.CPI,
                           "TPI": row.TPI,
                           "POWER_W": row.POWER_W,
                           "TIME_SEC": row.TIME_SEC
                           }
                power_proj, time_proj = apply_projection(dfc, sig_ref, model_name)
                dfi = pd.DataFrame(columns=cols, index=dfc.index)
                for col in shared_cols:
                    dfi[col] = group.loc[idx, col]
                dfi.TO_FREQ_KHZ = dfc.TO
                dfi.PROJ_TIME_SEC = time_proj
                dfi.PROJ_POWER_W = power_proj
                dfi.PROJ_ENERGY_J = dfi.PROJ_POWER_W*dfi.PROJ_TIME_SEC
                if df.empty:
                    df = dfi
                else:
                    df = pd.concat([df, dfi], ignore_index=True)

    return df


def add_variations(df: pd.DataFrame)-> None:
    """
    Add new columns (Power reduction, Time impact and Evergy savings) to the DataFrame in place.
    """
    df["POWER_RED_PERC"] = 0.0
    df["TIME_IMPCAT_PERC"] = 0.0
    df["ENERGY_SAVE_PERC"] = 0.0
    dg = df.groupby(["JOBID", "NODENAME", "STEPID"])

    for _, group in dg:
        idx = group.index
        fref = group.DEF_FREQ_KHZ.unique().item()
        fref_idx = group.loc[df["TO_FREQ_KHZ"] == fref].index

        ref_power = (df.loc[fref_idx, "PROJ_POWER_W"]).item()
        ref_time = (df.loc[fref_idx, "PROJ_TIME_SEC"]).item()
        ref_energy = (df.loc[fref_idx, "PROJ_ENERGY_J"]).item()

        df.loc[idx, "POWER_RED_PERC"] = (
            100 * (ref_power - group.PROJ_POWER_W) / ref_power
        )
        df.loc[idx, "TIME_IMPCAT_PERC"] = (
            100 * (group.PROJ_TIME_SEC - ref_time) / ref_time
        )
        df.loc[idx, "ENERGY_SAVE_PERC"] = (
            100 * (ref_energy - group.PROJ_ENERGY_J) / ref_energy
        )


def compute_projections(
        apps_file: str,
        coeffs_file: str,
        model_name:str,
        save_results: bool = False,
        verbose: bool = False
        ) -> pd.DataFrame:
    """
    Compute Power, Time and Energy projections using the given energy model

    Parameters
    ----------
        apps_file : str
            Path to the application signatures file.
        coeffs_file : str, optional
            Path to a csv coefficients file (genetated by coeffs_show.cpu_coeffs)
        model_name : str
            CPU model to choose from the following list: ['basic', 'basic_perc']
        save_results : bool, optional.
            If true saves the results into a CSV file named "*.SAVE.csv", default: False.
        verbose : bool, optional
            If True show detailed logs, default: False.

    Returns
    -------
        pd.DataFrame: DataFrame containing the applications signature with projections results.

    Side effect
    -----------
        Saves the DataFrame to a CSV file named "<apps_file>.PROJ.csv"
    """
    if apps_file and os.path.exists(apps_file):
        if verbose:
            logger.info(f"Reading applications signature from file: {apps_file}")
        apps_df = read_apps_data(apps_file)
    else:
        logger.error(f"The application signatures file {apps_file} not found")
        return

    if coeffs_file and os.path.exists(coeffs_file):
        if verbose:
            logger.info(f"Reading coefficients from file: {coeffs_file}")
        coeffs_df = pd.read_csv(coeffs_file, sep=";")
    else:
        logger.error(f"The coefficients file {apps_file} not found")
        return

    df_apps = read_apps_data(apps_file)
    df_coeffs = pd.read_csv(coeffs_file, sep=";")

    if model_name in ["basic", "basic_perc"]:
        df = compute_basic_projections(df_apps, df_coeffs, model_name)
        add_variations(df)
    elif model_name == "spr_imc_perc":
        # TODO
        pass
    else:
        logger.error(f"Unsupported model: '{model_name}'")

    # Store the results in a csv file
    if save_results:
        data_filename = Path(apps_file)
        output_filename = data_filename.stem + ".PROJ"
        df.to_csv(output_filename+ ".csv", header=True, index=False, sep=';')
        if verbose:
            logger.info(f"The output file {output_filename}.csv containg the "
                "projections results is stored")

    return df

