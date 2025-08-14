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

from .projections import compute_projections
from .logger import logger
from .ear_data import read_apps_data

__all__ = ["compute_policy_savings"]


def simulate_cpu_policy(
        df_proj: pd.DataFrame,
        thresholds: list[float] = None,
        verbose: bool = False
    ) ->  pd.DataFrame:
    """
    Estimates the potential energy savings from runs executed with monitoring.
    """
    if df_proj.empty:
        if verbose:
            logger.warning("Dataset does not contain runs with monitoring: "
                "potential savings will not be evaluated")
        return pd.DataFrame()
    else:
        if verbose:
            logger.info("Dataset contains runs with monitoring: "
                "potential savings will be evaluated")

    if thresholds == None:
        thresholds = [0.05]
    if verbose:
        logger.info(f"The target thresholds {thresholds} will be used")


    df = pd.DataFrame(columns=df_proj.columns)

    dg = df_proj.groupby(["JOBID", "NODENAME", "STEPID"])
    for _, group in dg:
        for th in thresholds:
            dfi = group[group.TIME_IMPCAT_PERC <= 100*th]
            selected_row = dfi.loc[dfi["PROJ_ENERGY_J"].idxmin()].copy()
            selected_row["POLICY_TH"] = th
            df.loc[-1] = selected_row
            df.index = df.index + 1
            df = df.sort_index()

    return df


def evaluate_cpu_policy(df_proj: pd.DataFrame, verbose: bool = False):
    """
    Evaluates the energy savings achieved from the runs with min-energy policy
    """
    if df_proj.empty:
        if verbose:
            logger.warning("Dataset does not contain runs with min-energy: "
                "policy savings will not be evaluated")
        return pd.DataFrame()
    else:
        if verbose:
            logger.info("Dataset contains runs with min-energy, "
                "policy savings will be evaluated")

    df = df_proj[df_proj.TO_FREQ_KHZ == df_proj.DEF_FREQ_KHZ]
    df.reset_index(drop=True)

    # Update perf. gains
    df.loc[:, "TIME_IMPCAT_PERC"] = (
        100 * (df["TIME_SEC"] - df["PROJ_TIME_SEC"]) / df["PROJ_TIME_SEC"]
    )

    df.loc[:, "POWER_RED_PERC"] = (
        100 * (df["PROJ_POWER_W"] - df["POWER_W"]) / df["PROJ_POWER_W"]
    )

    df.loc[:, "ENERGY_SAVE_PERC"] = (
        100 * (df["PROJ_ENERGY_J"] - df["ENERGY_J"]) / df["PROJ_ENERGY_J"]
    )

    return df


def savings_summarize(df_mon: pd.DataFrame, df_pol:pd.DataFrame):
    """
    """
    if df_mon is not None:
        if not df_mon.empty:
            logger.info("[bold magenta]Potential savings summary[/bold magenta] (monitoring runs)")

            logger.info("{:<12} {:>20} {:>28} {:>12}".format(
                "Threashold %",
                "Energy used (kJ)",
                "Potential Energy to save (kJ)",
                "Savings %"
            ))
            logger.info("-" * 76)
            dg = df_mon.groupby(["POLICY_TH"])
            for _, group in dg:
                idx = group.index
                th = group.POLICY_TH.unique()[0]
                tot_energy = group.ENERGY_J.sum()
                sav_energy = tot_energy - group.PROJ_ENERGY_J.sum()
                sav_perc = 100*sav_energy/tot_energy
                sel_freq_avg = group.TO_FREQ_KHZ.mean()
                logger.info("{:<12.1f} {:>20.2f} {:>28.2f} {:>12.2f}".format(
                    100*th, tot_energy/1e3, sav_energy/1e3, sav_perc
                ))
            logger.info("-" * 76)
    if df_pol is not None:
        if not df_pol.empty:
            logger.info("")
            logger.info("[bold magenta]Estimated savings summary[/bold magenta] (min-energy runs)")

            logger.info("{:<12} {:>20} {:>28} {:>12}".format(
                "Threashold %",
                "Energy used (kJ)",
                "Estimated Energy saved (kJ)",
                "Savings %"
            ))
            logger.info("-" * 76)
            dg = df_pol.groupby(["POLICY_TH"])
            for _, group in dg:
                idx = group.index
                th = group.POLICY_TH.unique()[0]
                used_energy = group.ENERGY_J.sum()
                total_energy = group.PROJ_ENERGY_J.sum()
                sav_energy = total_energy - used_energy
                sav_perc = 100*sav_energy/total_energy
                sel_freq_avg = group.FROM_FREQ_KHZ.mean()
                logger.info("{:<12.1f} {:>20.2f} {:>28.2f} {:>12.2f}".format(
                    100*th, used_energy/1e3, sav_energy/1e3, sav_perc
                ))
            used_energy = df_pol.ENERGY_J.sum()
            total_energy = df_pol.PROJ_ENERGY_J.sum()
            sav_energy = total_energy - used_energy
            sav_perc = 100*sav_energy/total_energy
            sel_freq_avg = df_pol.FROM_FREQ_KHZ.mean()
            logger.info("{:<12} {:>20.2f} {:>28.2f} {:>12.2f}".format(
                "Total", used_energy/1e3, sav_energy/1e3, sav_perc
            ))
            logger.info("-" * 76)


def compute_policy_savings(
        apps_file: str,
        coeffs_file: str,
        model_name: str,
        thresholds: list[float] = None,
        summary: bool = False,
        save_results: bool = True,
        verbose: bool = True,
    ) ->  pd.DataFrame:
    """
    Estimates
        1. The energy savings achieved from the runs with min-energy policy.
        2. The potential energy savings from the runs with monitoring.

    Parameters
    ----------
        apps_file : str
            Path to the application signatures file.
        coeffs_file : str
            Path to a csv coefficients file,
            genetated by coeffs_show.cpu_coeffs2csv.
        model_name : str
            CPU model to choose from the following list: ['basic', 'basic_perc']
        thresholds : list of float, optional
            Desireds maximum performance penalties, default value is [5%].
            It will be used with monitoring runs
        summury : bool, optional
            If true, print to the consol the savings results. Default: False.
        save_results : bool, optional.
            If true saves the results into a CSV file named "*.SAVE.csv".
            Default: True.
        verbose : bool, optional
            If True, show detailed logs.

    Returns
    -------
        pd.DataFrame: DataFrame containing the applications signature with saving results.

    Side effect
    -----------
        Saves the DataFrame to a CSV file named "<apps_file>.SAVE.csv"
    """

    if apps_file and os.path.exists(apps_file):
        if verbose:
            logger.info(f"Reading applications signature from file: {apps_file}")
    else:
        logger.error(f"The application signatures file {apps_file} not found")
        return pd.DataFrame()

    if coeffs_file and os.path.exists(coeffs_file):
        if verbose:
            logger.info(f"Reading coefficients from file: {coeffs_file}")
    else:
        logger.error(f"The coefficients file {apps_file} not found")
        return pd.DataFrame()

    if model_name not in ["basic", "basic_perc", "spr_imc_perc"]:
        logger.error(f"Unsupported model: '{model_name}")
        return pd.DataFrame()
    else:
        if verbose:
            logger.info(f"The model {model_name} will be used")

    # Get all projections
    df_apps = compute_projections(apps_file, coeffs_file, model_name)

    df_mon = df_apps.query("POLICY in ['monitoring', 'optimize']")
    df_mon = simulate_cpu_policy(df_mon, thresholds)

    df_pol = df_apps.query("POLICY == 'min_energy'")
    df_pol = evaluate_cpu_policy(df_pol)
    df = pd.concat([df_mon, df_pol], ignore_index=True)

    # Show a summury of the potential and/or estimated savings
    if summary:
        logger.info("")
        savings_summarize(df_mon, df_pol)

    # Store the results in a csv file
    if save_results:
        data_filename = Path(apps_file)
        output_filename = data_filename.stem + ".SAVE"
        df.to_csv(output_filename+ ".csv", header=True, index=False, sep=';')
        if verbose:
            logger.info(f"The output file {output_filename}.csv containg the "
                "savings results is saved\n")

    return df
