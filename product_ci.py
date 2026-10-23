"""Ex-post labeled-pool carbon propagation and sold-product CI.

calculate_product_ci() reads saved hourly schedules and emission signals and
writes only an hourly ledger and an annual summary. There is no optimization
or ANN dependency. The original allocation and opening-stock conventions apply.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PRODUCTS     = ["H2", "NH3", "MeOH"]


M_H2_TO_NH3  = 0.18


M_H2_TO_MEOH = 0.22


K_H2_OUT     = 0.12024


K_NH3_REC    = 0.11848


K_H2_FUEL    = 0.035294


DE_NH3  = 0.00519   # kgCO2 / kg NH3


DE_MEOH = 0.08      # kgCO2 / kg MeOH


TOL = 1e-6


REQUIRED_MODEL_COLS = [
    "model_tag", "datetime_utc", "day_of_year", "hour_in_day",

    "P_elz_mw", "P_nh3_mw", "P_meoh_mw", "P_crk_mw",

    "elz_power_green_share",
    "nh3_power_green_share",
    "meoh_power_green_share",
    "crk_power_green_share",

    "grid_import_mw",
    "res_used_mw",
    "bat_ch_mw",
    "bat_dis_mw",

    "H2_prod_g_kgph", "H2_prod_ng_kgph",
    "NH3_prod_g_kgph", "NH3_prod_ng_kgph",
    "MeOH_prod_g_kgph", "MeOH_prod_ng_kgph",

    "crk_nh3_in_kgph", "crk_h2_out_kgph", "crk_h2_fuel_kgph",

    "H2_sell_g_kgph", "H2_sell_ng_kgph",
    "NH3_sell_g_kgph", "NH3_sell_ng_kgph",
    "MeOH_sell_g_kgph", "MeOH_sell_ng_kgph",

    "tank_H2_g_kg", "tank_H2_ng_kg",
    "tank_NH3_g_kg", "tank_NH3_ng_kg",
    "tank_MeOH_g_kg", "tank_MeOH_ng_kg",
]


def clip_small_negative(series, tol=TOL):
    x = series.copy()
    x[(x < 0) & (x > -tol)] = 0.0
    return x


def load_model_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MODEL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def load_signal_df(path: Path, ef_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing signal file: {path}")
    df = pd.read_csv(path)
    if "datetime_utc" not in df.columns or ef_col not in df.columns:
        raise KeyError(f"{path.name} must contain datetime_utc and {ef_col}")
    df = df[["datetime_utc", ef_col]].copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"])
    df = df.rename(columns={ef_col: "ef_kg_per_kwh"})
    return df


def reconstruct_unit_grid_battery_split(df: pd.DataFrame, tol=TOL):
    """Attribute battery discharge proportionally across non-green process loads."""
    d = df.copy()

    d["P_elz_total_mw"] = d["P_elz_mw"]
    d["P_nh3_total_mw"] = d["P_nh3_mw"]
    d["P_meoh_total_mw"] = d["P_meoh_mw"]
    d["P_crk_total_mw"] = d["P_crk_mw"]

    for col in ["elz_power_green_share", "nh3_power_green_share", "meoh_power_green_share", "crk_power_green_share"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").clip(lower=0.0, upper=1.0).fillna(0.0)

    d["P_elz_green_mw"] = d["P_elz_total_mw"] * d["elz_power_green_share"]
    d["P_nh3_green_mw"] = d["P_nh3_total_mw"] * d["nh3_power_green_share"]
    d["P_meoh_green_mw"] = d["P_meoh_total_mw"] * d["meoh_power_green_share"]
    d["P_crk_green_mw"] = d["P_crk_total_mw"] * d["crk_power_green_share"]

    d["P_elz_ng_mw"] = d["P_elz_total_mw"] - d["P_elz_green_mw"]
    d["P_nh3_ng_mw"] = d["P_nh3_total_mw"] - d["P_nh3_green_mw"]
    d["P_meoh_ng_mw"] = d["P_meoh_total_mw"] - d["P_meoh_green_mw"]
    d["P_crk_ng_mw"] = d["P_crk_total_mw"] - d["P_crk_green_mw"]

    for col in ["P_elz_ng_mw", "P_nh3_ng_mw", "P_meoh_ng_mw", "P_crk_ng_mw"]:
        d[col] = clip_small_negative(d[col], tol=tol)

    d["P_ng_sum_units_mw"] = (
        d["P_elz_ng_mw"] +
        d["P_nh3_ng_mw"] +
        d["P_meoh_ng_mw"] +
        d["P_crk_ng_mw"]
    )

    def alloc_bat(unit_ng, ng_sum, bat_dis):
        return np.where(
            (ng_sum > tol) & (bat_dis > tol),
            bat_dis * unit_ng / ng_sum,
            0.0
        )

    d["P_elz_bat_mw"] = alloc_bat(d["P_elz_ng_mw"], d["P_ng_sum_units_mw"], d["bat_dis_mw"])
    d["P_nh3_bat_mw"] = alloc_bat(d["P_nh3_ng_mw"], d["P_ng_sum_units_mw"], d["bat_dis_mw"])
    d["P_meoh_bat_mw"] = alloc_bat(d["P_meoh_ng_mw"], d["P_ng_sum_units_mw"], d["bat_dis_mw"])
    d["P_crk_bat_mw"] = alloc_bat(d["P_crk_ng_mw"], d["P_ng_sum_units_mw"], d["bat_dis_mw"])

    d["P_elz_grid_exact_mw"] = d["P_elz_ng_mw"] - d["P_elz_bat_mw"]
    d["P_nh3_grid_exact_mw"] = d["P_nh3_ng_mw"] - d["P_nh3_bat_mw"]
    d["P_meoh_grid_exact_mw"] = d["P_meoh_ng_mw"] - d["P_meoh_bat_mw"]
    d["P_crk_grid_exact_mw"] = d["P_crk_ng_mw"] - d["P_crk_bat_mw"]

    for col in ["P_elz_grid_exact_mw", "P_nh3_grid_exact_mw", "P_meoh_grid_exact_mw", "P_crk_grid_exact_mw"]:
        d[col] = clip_small_negative(d[col], tol=tol)

    d["grid_unit_sum_exact_mw"] = (
        d["P_elz_grid_exact_mw"] +
        d["P_nh3_grid_exact_mw"] +
        d["P_meoh_grid_exact_mw"] +
        d["P_crk_grid_exact_mw"]
    )
    d["bat_unit_sum_exact_mw"] = (
        d["P_elz_bat_mw"] +
        d["P_nh3_bat_mw"] +
        d["P_meoh_bat_mw"] +
        d["P_crk_bat_mw"]
    )

    d["grid_balance_gap_mw"] = d["grid_unit_sum_exact_mw"] - d["grid_import_mw"]
    d["bat_balance_gap_mw"] = d["bat_unit_sum_exact_mw"] - d["bat_dis_mw"]

    diag = pd.DataFrame({
        "metric": [
            "max_abs_grid_balance_gap_mw",
            "max_abs_battery_balance_gap_mw",
            "mean_abs_grid_balance_gap_mw",
            "mean_abs_battery_balance_gap_mw",
        ],
        "value": [
            float(d["grid_balance_gap_mw"].abs().max()),
            float(d["bat_balance_gap_mw"].abs().max()),
            float(d["grid_balance_gap_mw"].abs().mean()),
            float(d["bat_balance_gap_mw"].abs().mean()),
        ]
    })

    return d, diag


def reconstruct_intermediates(df: pd.DataFrame, flow_mode="auto"):
    """Use complete direct exports, otherwise preserve the archived reconstruction.

    Legacy final-row reconstruction is intentionally left invalid, as in the original.
    """
    d = df.copy()

    for p in ["H2", "NH3", "MeOH"]:
        for tag in ["g", "ng"]:
            d[f"tank_{p}_{tag}_next_kg"] = d[f"tank_{p}_{tag}_kg"].shift(-1)
            d[f"delta_tank_{p}_{tag}_kg"] = d[f"tank_{p}_{tag}_next_kg"] - d[f"tank_{p}_{tag}_kg"]

    d["has_next_state"] = d["tank_H2_g_next_kg"].notna()

    d["H_to_NH3_g_kgph"]   = M_H2_TO_NH3  * d["NH3_prod_g_kgph"]
    d["H_to_NH3_ng_kgph"]  = M_H2_TO_NH3  * d["NH3_prod_ng_kgph"]
    d["H_to_MeOH_g_kgph"]  = M_H2_TO_MEOH * d["MeOH_prod_g_kgph"]
    d["H_to_MeOH_ng_kgph"] = M_H2_TO_MEOH * d["MeOH_prod_ng_kgph"]

    denom = 1.0 - K_NH3_REC
    d["Q_crk_in_g_recon_kgph"] = (
        d["NH3_prod_g_kgph"] - d["NH3_sell_g_kgph"] - d["delta_tank_NH3_g_kg"]
    ) / denom
    d["Q_crk_in_ng_recon_kgph"] = (
        d["NH3_prod_ng_kgph"] - d["NH3_sell_ng_kgph"] - d["delta_tank_NH3_ng_kg"]
    ) / denom

    d["Q_crk_in_g_recon_kgph"] = clip_small_negative(d["Q_crk_in_g_recon_kgph"])
    d["Q_crk_in_ng_recon_kgph"] = clip_small_negative(d["Q_crk_in_ng_recon_kgph"])

    d["Q_crk_in_recon_sum_kgph"] = d["Q_crk_in_g_recon_kgph"] + d["Q_crk_in_ng_recon_kgph"]
    d["Q_crk_total_gap_kgph"] = d["Q_crk_in_recon_sum_kgph"] - d["crk_nh3_in_kgph"]

    d["Q_crk_split_valid"] = (
        d["has_next_state"] &
        d["Q_crk_in_g_recon_kgph"].notna() &
        d["Q_crk_in_ng_recon_kgph"].notna() &
        (d["Q_crk_total_gap_kgph"].abs() <= 5e-4)
    )

    valid = d["Q_crk_split_valid"] & (d["Q_crk_in_recon_sum_kgph"] > TOL)
    scale = np.where(valid, d["crk_nh3_in_kgph"] / d["Q_crk_in_recon_sum_kgph"], 1.0)
    d.loc[valid, "Q_crk_in_g_recon_kgph"] *= scale[valid]
    d.loc[valid, "Q_crk_in_ng_recon_kgph"] *= scale[valid]

    d["NH3_rec_g_recon_kgph"] = K_NH3_REC * d["Q_crk_in_g_recon_kgph"]
    d["NH3_rec_ng_recon_kgph"] = K_NH3_REC * d["Q_crk_in_ng_recon_kgph"]

    d["H_crk_out_g_recon_kgph"] = K_H2_OUT * d["Q_crk_in_g_recon_kgph"]
    d["H_crk_out_ng_recon_kgph"] = K_H2_OUT * d["Q_crk_in_ng_recon_kgph"]

    d["H_crk_fuel_g_recon_kgph"] = K_H2_FUEL * d["Q_crk_in_g_recon_kgph"]
    d["H_crk_fuel_ng_recon_kgph"] = K_H2_FUEL * d["Q_crk_in_ng_recon_kgph"]

    d["H_crk_out_total_gap_kgph"] = (
        d["H_crk_out_g_recon_kgph"] + d["H_crk_out_ng_recon_kgph"] - d["crk_h2_out_kgph"]
    )
    d["H_crk_fuel_total_gap_kgph"] = (
        d["H_crk_fuel_g_recon_kgph"] + d["H_crk_fuel_ng_recon_kgph"] - d["crk_h2_fuel_kgph"]
    )

    diag = pd.DataFrame({
        "metric": [
            "n_rows_total",
            "n_rows_with_next_state",
            "n_rows_valid_cracker_split",
            "n_rows_dropped_for_split_reconstruction",
            "max_abs_cracker_nh3in_gap_kgph",
            "max_abs_cracker_h2out_gap_kgph",
            "max_abs_cracker_h2fuel_gap_kgph",
        ],
        "value": [
            int(len(d)),
            int(d["has_next_state"].sum()),
            int(d["Q_crk_split_valid"].sum()),
            int((~d["Q_crk_split_valid"]).sum()),
            float(d["Q_crk_total_gap_kgph"].abs().max()),
            float(d["H_crk_out_total_gap_kgph"].abs().max()),
            float(d["H_crk_fuel_total_gap_kgph"].abs().max()),
        ]
    })

    d, source_name = use_direct_flows(d, flow_mode, original=df)
    d["cracker_flow_source"] = source_name
    return d, diag

def derive_zero_initial_cis(recon_df: pd.DataFrame, signal_df: pd.DataFrame, signal_name: str):
    """
    Simplified assumption:
    all initial tank inventories at the start of the horizon are assigned zero carbon intensity.
    """
    return {
        "initial_inventory_rule_note": (
            "All initial tank inventories at the start of the study horizon are assigned zero carbon intensity."
        ),
        "initial_inventory_basis_signal": signal_name,
        "initial_inventory_basis_datetime_utc": "assumed_zero_initial_inventory",
        "initial_inventory_basis_day_of_year": 1,
        "initial_inventory_basis_hour_in_day": 0,
        "initial_inventory_basis_ef_kg_per_kwh": 0.0,
        "CI_init_H2_g_kgco2_perkg": 0.0,
        "CI_init_H2_ng_kgco2_perkg": 0.0,
        "CI_init_NH3_g_kgco2_perkg": 0.0,
        "CI_init_NH3_ng_kgco2_perkg": 0.0,
        "CI_init_MeOH_g_kgco2_perkg": 0.0,
        "CI_init_MeOH_ng_kgco2_perkg": 0.0,
    }


def run_carbon_ledger(recon_df: pd.DataFrame, signal_name: str, signal_df: pd.DataFrame, init_ci_dict: dict):
    d = recon_df.merge(signal_df, on="datetime_utc", how="left").copy()
    if d["ef_kg_per_kwh"].isna().any():
        raise ValueError(f"Missing EF values after merge for {signal_name}")

    d["elec_carbon_elz_kgph"]  = 1000.0 * d["P_elz_grid_exact_mw"]  * d["ef_kg_per_kwh"]
    d["elec_carbon_nh3_kgph"]  = 1000.0 * d["P_nh3_grid_exact_mw"]  * d["ef_kg_per_kwh"]
    d["elec_carbon_meoh_kgph"] = 1000.0 * d["P_meoh_grid_exact_mw"] * d["ef_kg_per_kwh"]
    d["elec_carbon_crk_kgph"]  = 1000.0 * d["P_crk_grid_exact_mw"]  * d["ef_kg_per_kwh"]


    d["C_new_H2_g_kgco2"] = 0.0
    d["C_new_H2_ng_kgco2"] = d["elec_carbon_elz_kgph"]

    d["CI_new_H2_g_batch"] = np.where(d["H2_prod_g_kgph"] > TOL, d["C_new_H2_g_kgco2"] / d["H2_prod_g_kgph"], np.nan)
    d["CI_new_H2_ng_batch"] = np.where(d["H2_prod_ng_kgph"] > TOL, d["C_new_H2_ng_kgco2"] / d["H2_prod_ng_kgph"], np.nan)

    CH2g = float(d.loc[0, "tank_H2_g_kg"]   * init_ci_dict["CI_init_H2_g_kgco2_perkg"])
    CH2ng = float(d.loc[0, "tank_H2_ng_kg"] * init_ci_dict["CI_init_H2_ng_kgco2_perkg"])
    CNH3g = float(d.loc[0, "tank_NH3_g_kg"]   * init_ci_dict["CI_init_NH3_g_kgco2_perkg"])
    CNH3ng = float(d.loc[0, "tank_NH3_ng_kg"] * init_ci_dict["CI_init_NH3_ng_kgco2_perkg"])
    CMeOHg = float(d.loc[0, "tank_MeOH_g_kg"]   * init_ci_dict["CI_init_MeOH_g_kgco2_perkg"])
    CMeOHng = float(d.loc[0, "tank_MeOH_ng_kg"] * init_ci_dict["CI_init_MeOH_ng_kgco2_perkg"])

    rows = []

    for _, r in d.iterrows():
        stock_before = np.array([CH2g, CH2ng, CNH3g, CNH3ng, CMeOHg, CMeOHng])
        cracker_clip = 0.0
        ci_H2_g = CH2g / r["tank_H2_g_kg"] if r["tank_H2_g_kg"] > TOL else 0.0
        ci_H2_ng = CH2ng / r["tank_H2_ng_kg"] if r["tank_H2_ng_kg"] > TOL else 0.0

        ci_NH3_g = CNH3g / r["tank_NH3_g_kg"] if r["tank_NH3_g_kg"] > TOL else 0.0
        ci_NH3_ng = CNH3ng / r["tank_NH3_ng_kg"] if r["tank_NH3_ng_kg"] > TOL else 0.0

        ci_MeOH_g = CMeOHg / r["tank_MeOH_g_kg"] if r["tank_MeOH_g_kg"] > TOL else 0.0
        ci_MeOH_ng = CMeOHng / r["tank_MeOH_ng_kg"] if r["tank_MeOH_ng_kg"] > TOL else 0.0

        C_H2_to_NH3_g = ci_H2_g * r["H_to_NH3_g_kgph"]
        C_H2_to_NH3_ng = ci_H2_ng * r["H_to_NH3_ng_kgph"]
        C_H2_to_MeOH_g = ci_H2_g * r["H_to_MeOH_g_kgph"]
        C_H2_to_MeOH_ng = ci_H2_ng * r["H_to_MeOH_ng_kgph"]

        C_direct_NH3_g = DE_NH3 * r["NH3_prod_g_kgph"]
        C_direct_NH3_ng = DE_NH3 * r["NH3_prod_ng_kgph"]
        C_direct_MeOH_g = DE_MEOH * r["MeOH_prod_g_kgph"]
        C_direct_MeOH_ng = DE_MEOH * r["MeOH_prod_ng_kgph"]

        # Green downstream products inherit carbon from green H2 and direct process emissions only.
        # Grid-electricity carbon is assigned only to the non-green production stream.
        # Battery-fed non-green production remains non-green for attribution but receives zero grid CO2,
        # because elec_carbon_* is based only on reconstructed grid-fed unit electricity.

        C_new_NH3_g = (
            C_H2_to_NH3_g
            + C_direct_NH3_g
        )

        C_new_NH3_ng = (
            C_H2_to_NH3_ng
            + r["elec_carbon_nh3_kgph"]
            + C_direct_NH3_ng
        )

        C_new_MeOH_g = (
            C_H2_to_MeOH_g
            + C_direct_MeOH_g
        )

        C_new_MeOH_ng = (
            C_H2_to_MeOH_ng
            + r["elec_carbon_meoh_kgph"]
            + C_direct_MeOH_ng
        )

        if bool(r["Q_crk_split_valid"]):
            Qg = r["Q_crk_in_g_recon_kgph"]
            Qng = r["Q_crk_in_ng_recon_kgph"]

            NH3_rec_g = r["NH3_rec_g_recon_kgph"]
            NH3_rec_ng = r["NH3_rec_ng_recon_kgph"]

            H_out_g = r["H_crk_out_g_recon_kgph"]
            H_out_ng = r["H_crk_out_ng_recon_kgph"]

            H_fuel_g = r["H_crk_fuel_g_recon_kgph"]
            H_fuel_ng = r["H_crk_fuel_ng_recon_kgph"]

            # Cracker carbon propagation:
            # Incoming NH3 carries the current CI of the corresponding labeled NH3 tank.
            # Recovered NH3 returns with inherited NH3 carbon.
            # The remaining inherited carbon is assigned to recovered H2.
            # Cracker grid-electricity carbon is assigned only to the non-green recovered H2 stream.

            C_in_crk_NH3_g = ci_NH3_g * Qg
            C_in_crk_NH3_ng = ci_NH3_ng * Qng

            C_rec_NH3_g = ci_NH3_g * NH3_rec_g
            C_rec_NH3_ng = ci_NH3_ng * NH3_rec_ng

            C_H2_from_crk_g = C_in_crk_NH3_g - C_rec_NH3_g
            C_H2_from_crk_ng = C_in_crk_NH3_ng - C_rec_NH3_ng + r["elec_carbon_crk_kgph"]

            # Numerical safety for reconstructed cracker carbon flows.
            # Small negative values can occur from floating-point noise and reconstructed cracker splits.
            # Values below -1 kgCO2/h should be inspected because they may indicate a real reconstruction issue.

            CRK_CARBON_TOL = 1.0  # kgCO2/h

            if C_H2_from_crk_g < -CRK_CARBON_TOL:
                raise ValueError(
                    f"Large negative green cracker H2 carbon at {r['datetime_utc']}: {C_H2_from_crk_g}"
                )

            if C_H2_from_crk_ng < -CRK_CARBON_TOL:
                raise ValueError(
                    f"Large negative non-green cracker H2 carbon at {r['datetime_utc']}: {C_H2_from_crk_ng}"
                )

            cracker_clip = max(-C_H2_from_crk_g, 0.0) + max(-C_H2_from_crk_ng, 0.0)
            C_H2_from_crk_g = max(C_H2_from_crk_g, 0.0)
            C_H2_from_crk_ng = max(C_H2_from_crk_ng, 0.0)
        else:
            Qg = Qng = 0.0
            H_fuel_g = H_fuel_ng = 0.0
            C_rec_NH3_g = C_rec_NH3_ng = 0.0
            C_H2_from_crk_g = C_H2_from_crk_ng = 0.0

        C_sell_H2_g = ci_H2_g * r["H2_sell_g_kgph"]
        C_sell_H2_ng = ci_H2_ng * r["H2_sell_ng_kgph"]

        C_sell_NH3_g = ci_NH3_g * r["NH3_sell_g_kgph"]
        C_sell_NH3_ng = ci_NH3_ng * r["NH3_sell_ng_kgph"]

        C_sell_MeOH_g = ci_MeOH_g * r["MeOH_sell_g_kgph"]
        C_sell_MeOH_ng = ci_MeOH_ng * r["MeOH_sell_ng_kgph"]

        def checked_nonnegative(x, name, timestamp, tol=1.0):
            """
            Clip small negative carbon stocks caused by numerical noise or reconstructed flows.
            Raise an error only for large negative values.
            tol is in kgCO2.
            """
            if x < -tol:
                raise ValueError(f"Large negative carbon stock in {name} at {timestamp}: {x}")
            return max(x, 0.0)

        CH2g_new = (
            CH2g
            - C_sell_H2_g
            - C_H2_to_NH3_g
            - C_H2_to_MeOH_g
            + r["C_new_H2_g_kgco2"]
            + C_H2_from_crk_g
            - ci_H2_g * H_fuel_g
        )

        CH2ng_new = (
            CH2ng
            - C_sell_H2_ng
            - C_H2_to_NH3_ng
            - C_H2_to_MeOH_ng
            + r["C_new_H2_ng_kgco2"]
            + C_H2_from_crk_ng
            - ci_H2_ng * H_fuel_ng
        )

        CNH3g_new = (
            CNH3g
            - C_sell_NH3_g
            - ci_NH3_g * Qg
            + C_new_NH3_g
            + C_rec_NH3_g
        )

        CNH3ng_new = (
            CNH3ng
            - C_sell_NH3_ng
            - ci_NH3_ng * Qng
            + C_new_NH3_ng
            + C_rec_NH3_ng
        )

        CMeOHg_new = (
            CMeOHg
            - C_sell_MeOH_g
            + C_new_MeOH_g
        )

        CMeOHng_new = (
            CMeOHng
            - C_sell_MeOH_ng
            + C_new_MeOH_ng
        )

        stock_raw = np.array([CH2g_new, CH2ng_new, CNH3g_new, CNH3ng_new, CMeOHg_new, CMeOHng_new])
        CH2g = checked_nonnegative(CH2g_new, "H2_g", r["datetime_utc"])
        CH2ng = checked_nonnegative(CH2ng_new, "H2_ng", r["datetime_utc"])
        CNH3g = checked_nonnegative(CNH3g_new, "NH3_g", r["datetime_utc"])
        CNH3ng = checked_nonnegative(CNH3ng_new, "NH3_ng", r["datetime_utc"])
        CMeOHg = checked_nonnegative(CMeOHg_new, "MeOH_g", r["datetime_utc"])
        CMeOHng = checked_nonnegative(CMeOHng_new, "MeOH_ng", r["datetime_utc"])

        stock_after = np.array([CH2g, CH2ng, CNH3g, CNH3ng, CMeOHg, CMeOHng])
        stock_clip = float(np.sum(stock_after - stock_raw))
        fuel_sink = ci_H2_g * H_fuel_g + ci_H2_ng * H_fuel_ng
        electricity_in = sum(float(r[f"elec_carbon_{u}_kgph"]) for u in ["elz", "nh3", "meoh", "crk"])
        direct_in = C_direct_NH3_g + C_direct_NH3_ng + C_direct_MeOH_g + C_direct_MeOH_ng
        sold_total = C_sell_H2_g + C_sell_H2_ng + C_sell_NH3_g + C_sell_NH3_ng + C_sell_MeOH_g + C_sell_MeOH_ng
        unassigned_cracker = float(r["elec_carbon_crk_kgph"]) if not bool(r["Q_crk_split_valid"]) else 0.0
        plant_residual = float(stock_after.sum() - stock_before.sum() - (
            electricity_in + direct_in - sold_total - fuel_sink + stock_clip + cracker_clip - unassigned_cracker))
        stock_fields = {}
        for k, pool in enumerate(["H2_g", "H2_ng", "NH3_g", "NH3_ng", "MeOH_g", "MeOH_ng"]):
            stock_fields[f"carbon_{pool}_start_kgco2"] = stock_before[k]
            stock_fields[f"carbon_{pool}_end_kgco2"] = stock_after[k]
        rows.append({
            **stock_fields,
            "new_electricity_burden_kgco2": electricity_in,
            "new_direct_burden_kgco2": direct_in,
            "internal_H2_fuel_burden_kgco2": fuel_sink,
            "stock_clipping_addition_kgco2": stock_clip,
            "cracker_clipping_addition_kgco2": cracker_clip,
            "unassigned_cracker_electricity_kgco2": unassigned_cracker,
            "plant_carbon_balance_residual_kgco2": plant_residual,
            "datetime_utc": r["datetime_utc"],
            "day_of_year": int(r["day_of_year"]),
            "hour_in_day": int(r["hour_in_day"]),
            "CI_tank_H2_g": ci_H2_g,
            "CI_tank_H2_ng": ci_H2_ng,
            "CI_tank_NH3_g": ci_NH3_g,
            "CI_tank_NH3_ng": ci_NH3_ng,
            "CI_tank_MeOH_g": ci_MeOH_g,
            "CI_tank_MeOH_ng": ci_MeOH_ng,
            "C_sell_H2_g_kgco2": C_sell_H2_g,
            "C_sell_H2_ng_kgco2": C_sell_H2_ng,
            "C_sell_NH3_g_kgco2": C_sell_NH3_g,
            "C_sell_NH3_ng_kgco2": C_sell_NH3_ng,
            "C_sell_MeOH_g_kgco2": C_sell_MeOH_g,
            "C_sell_MeOH_ng_kgco2": C_sell_MeOH_ng,
            "H2_sold_total_kgph": float(r["H2_sell_g_kgph"] + r["H2_sell_ng_kgph"]),
            "NH3_sold_total_kgph": float(r["NH3_sell_g_kgph"] + r["NH3_sell_ng_kgph"]),
            "MeOH_sold_total_kgph": float(r["MeOH_sell_g_kgph"] + r["MeOH_sell_ng_kgph"]),
            "H2_sold_CI_kgco2_perkg_hourly": float((C_sell_H2_g + C_sell_H2_ng) / max(r["H2_sell_g_kgph"] + r["H2_sell_ng_kgph"], TOL)) if (r["H2_sell_g_kgph"] + r["H2_sell_ng_kgph"]) > TOL else np.nan,
            "NH3_sold_CI_kgco2_perkg_hourly": float((C_sell_NH3_g + C_sell_NH3_ng) / max(r["NH3_sell_g_kgph"] + r["NH3_sell_ng_kgph"], TOL)) if (r["NH3_sell_g_kgph"] + r["NH3_sell_ng_kgph"]) > TOL else np.nan,
            "MeOH_sold_CI_kgco2_perkg_hourly": float((C_sell_MeOH_g + C_sell_MeOH_ng) / max(r["MeOH_sell_g_kgph"] + r["MeOH_sell_ng_kgph"], TOL)) if (r["MeOH_sell_g_kgph"] + r["MeOH_sell_ng_kgph"]) > TOL else np.nan,
            "elec_carbon_elz_kgph": r["elec_carbon_elz_kgph"],
            "elec_carbon_nh3_kgph": r["elec_carbon_nh3_kgph"],
            "elec_carbon_meoh_kgph": r["elec_carbon_meoh_kgph"],
            "elec_carbon_crk_kgph": r["elec_carbon_crk_kgph"],
            "Q_crk_split_valid": bool(r["Q_crk_split_valid"]),
        })

    hourly = pd.DataFrame(rows)


    if len(hourly) >= 2:
        last_idx = hourly.index[-1]
        prev_idx = hourly.index[-2]
        fill_cols = [
            "CI_tank_H2_g", "CI_tank_H2_ng",
            "CI_tank_NH3_g", "CI_tank_NH3_ng",
            "CI_tank_MeOH_g", "CI_tank_MeOH_ng",
            "C_sell_H2_g_kgco2", "C_sell_H2_ng_kgco2",
            "C_sell_NH3_g_kgco2", "C_sell_NH3_ng_kgco2",
            "C_sell_MeOH_g_kgco2", "C_sell_MeOH_ng_kgco2",
        ]
        hourly.loc[last_idx, fill_cols] = hourly.loc[last_idx, fill_cols].where(hourly.loc[last_idx, fill_cols].notna(), hourly.loc[prev_idx, fill_cols])

    hourly["date"] = pd.to_datetime(hourly["datetime_utc"]).dt.date

    d2 = d.merge(
        hourly[[
            "datetime_utc",
            "CI_tank_H2_g", "CI_tank_H2_ng",
            "CI_tank_NH3_g", "CI_tank_NH3_ng",
            "CI_tank_MeOH_g", "CI_tank_MeOH_ng",
            "C_sell_H2_g_kgco2", "C_sell_H2_ng_kgco2",
            "C_sell_NH3_g_kgco2", "C_sell_NH3_ng_kgco2",
            "C_sell_MeOH_g_kgco2", "C_sell_MeOH_ng_kgco2"
        ]],
        on="datetime_utc",
        how="left"
    )

    daily = d2.groupby("day_of_year", as_index=False).agg(
        date=("datetime_utc", lambda x: str(pd.to_datetime(x.iloc[0]).date())),

        H2_sold_g_kg=("H2_sell_g_kgph", "sum"),
        H2_sold_ng_kg=("H2_sell_ng_kgph", "sum"),
        H2_carbon_g_kg=("C_sell_H2_g_kgco2", "sum"),
        H2_carbon_ng_kg=("C_sell_H2_ng_kgco2", "sum"),

        NH3_sold_g_kg=("NH3_sell_g_kgph", "sum"),
        NH3_sold_ng_kg=("NH3_sell_ng_kgph", "sum"),
        NH3_carbon_g_kg=("C_sell_NH3_g_kgco2", "sum"),
        NH3_carbon_ng_kg=("C_sell_NH3_ng_kgco2", "sum"),

        MeOH_sold_g_kg=("MeOH_sell_g_kgph", "sum"),
        MeOH_sold_ng_kg=("MeOH_sell_ng_kgph", "sum"),
        MeOH_carbon_g_kg=("C_sell_MeOH_g_kgco2", "sum"),
        MeOH_carbon_ng_kg=("C_sell_MeOH_ng_kgco2", "sum"),
    )

    for p in PRODUCTS:
        daily[f"{p}_sold_total_kg"] = daily[f"{p}_sold_g_kg"] + daily[f"{p}_sold_ng_kg"]
        daily[f"{p}_sold_green_share_pct"] = np.where(
            daily[f"{p}_sold_total_kg"] > TOL,
            100.0 * daily[f"{p}_sold_g_kg"] / daily[f"{p}_sold_total_kg"],
            np.nan
        )
        daily[f"{p}_sold_CI_kgco2_perkg"] = np.where(
            daily[f"{p}_sold_total_kg"] > TOL,
            (daily[f"{p}_carbon_g_kg"] + daily[f"{p}_carbon_ng_kg"]) / daily[f"{p}_sold_total_kg"],
            np.nan
        )

    summary = pd.DataFrame([{
        "initial_inventory_method_note": init_ci_dict["initial_inventory_rule_note"],
        "initial_inventory_basis_signal": init_ci_dict["initial_inventory_basis_signal"],
        "initial_inventory_basis_ef_kg_per_kwh": init_ci_dict["initial_inventory_basis_ef_kg_per_kwh"],
        "initial_inventory_basis_datetime_utc": init_ci_dict["initial_inventory_basis_datetime_utc"],
        "initial_inventory_basis_day_of_year": init_ci_dict["initial_inventory_basis_day_of_year"],
        "initial_inventory_basis_hour_in_day": init_ci_dict["initial_inventory_basis_hour_in_day"],

        "CI_init_H2_g_kgco2_perkg": init_ci_dict["CI_init_H2_g_kgco2_perkg"],
        "CI_init_H2_ng_kgco2_perkg": init_ci_dict["CI_init_H2_ng_kgco2_perkg"],
        "CI_init_NH3_g_kgco2_perkg": init_ci_dict["CI_init_NH3_g_kgco2_perkg"],
        "CI_init_NH3_ng_kgco2_perkg": init_ci_dict["CI_init_NH3_ng_kgco2_perkg"],
        "CI_init_MeOH_g_kgco2_perkg": init_ci_dict["CI_init_MeOH_g_kgco2_perkg"],
        "CI_init_MeOH_ng_kgco2_perkg": init_ci_dict["CI_init_MeOH_ng_kgco2_perkg"],

        "H2_sold_green_share_pct": float(100.0 * daily["H2_sold_g_kg"].sum() / max(daily["H2_sold_total_kg"].sum(), TOL)),
        "NH3_sold_green_share_pct": float(100.0 * daily["NH3_sold_g_kg"].sum() / max(daily["NH3_sold_total_kg"].sum(), TOL)),
        "MeOH_sold_green_share_pct": float(100.0 * daily["MeOH_sold_g_kg"].sum() / max(daily["MeOH_sold_total_kg"].sum(), TOL)),

        "H2_sold_CI_kgco2_perkg": float((daily["H2_carbon_g_kg"].sum() + daily["H2_carbon_ng_kg"].sum()) / max(daily["H2_sold_total_kg"].sum(), TOL)),
        "NH3_sold_CI_kgco2_perkg": float((daily["NH3_carbon_g_kg"].sum() + daily["NH3_carbon_ng_kg"].sum()) / max(daily["NH3_sold_total_kg"].sum(), TOL)),
        "MeOH_sold_CI_kgco2_perkg": float((daily["MeOH_carbon_g_kg"].sum() + daily["MeOH_carbon_ng_kg"].sum()) / max(daily["MeOH_sold_total_kg"].sum(), TOL)),

        "hours_excluded_from_propagation_due_to_missing_next_state": int((~d["has_next_state"]).sum()),
        "hours_excluded_from_cracker_split_due_to_invalid_reconstruction": int((~d["Q_crk_split_valid"]).sum()),
        "electricity_split_method_note": (
            "Unit grid power is attributed from total unit power, saved green share, "
            "and proportional allocation of battery discharge across non-green unit demand. "
            "Battery-fed power carries zero electricity carbon."
        ),
    }])

    return d2, hourly, daily, summary


def use_direct_flows(frame, mode="auto", original=None):
    """Prefer complete direct cracking exports; retain legacy fallback unchanged.

    `legacy` explicitly ignores direct columns, supporting a historical replay.
    Both direct feed labels are required together. Other directly exported
    streams are accepted if present; their fixed-yield definitions supply any
    missing derived stream. Aggregate and material-balance checks are separate.
    """
    if mode not in {"auto", "legacy", "direct"}:
        raise ValueError("flow mode must be auto, legacy, or direct")
    direct = [f"crk_nh3_in_{label}_kgph" for label in ("g", "ng")]
    present = [c in frame for c in direct]
    if mode != "legacy" and any(present) and not all(present):
        raise ValueError("Partial direct cracking feed: both g and ng columns are required")
    if mode == "direct" and not all(present):
        raise ValueError("Direct mode requires the two labeled cracking feed exports")
    if mode == "legacy" or not all(present):
        return frame, "legacy_reconstruction"
    d = frame.copy()
    factors = {"NH3_rec": ("crk_nh3_return", 0.11848),
               "H_crk_out": ("crk_h2_out", 0.12024),
               "H_crk_fuel": ("crk_h2_fuel", 0.035294)}
    for label, col in zip(("g", "ng"), direct):
        feed = d[col].to_numpy(dtype=float)
        if not np.isfinite(feed).all() or np.any(feed < -1e-6):
            raise ValueError(f"Invalid direct feed in {col}")
        d[f"Q_crk_in_{label}_recon_kgph"] = feed
        for target, (source, factor) in factors.items():
            source_col = f"{source}_{label}_kgph"
            values = d[source_col].to_numpy(dtype=float) if source_col in d else factor * feed
            if not np.isfinite(values).all() or np.any(values < -1e-6):
                raise ValueError(f"Invalid direct/derived stream {source_col}")
            if not np.allclose(values, factor * feed, rtol=0, atol=5e-4):
                raise ValueError(f"Direct stream violates the fixed yield: {source_col}")
            d[f"{target}_{label}_recon_kgph"] = values
    total = d[direct].sum(axis=1)
    if not np.allclose(total, d["crk_nh3_in_kgph"], rtol=0, atol=5e-4):
        raise ValueError("Direct labeled cracking feeds do not sum to the saved total")
    d["Q_crk_split_valid"] = True
    original = original if original is not None else frame
    for stem in ("H_to_NH3", "H_to_MeOH"):
        for label in ("g", "ng"):
            col = f"{stem}_{label}_kgph"
            # Same name is used by the archived stoichiometric reconstruction.
            # The caller restores any directly exported value after that step.
            if col in original:
                if not np.allclose(original[col], d[col], rtol=0, atol=5e-4):
                    raise ValueError(f"Direct material flow disagrees with stoichiometry: {col}")
                d[col] = original[col]
    return d, "direct_exports"


SIGNALS = {
    "aef": ("AEF", "grid_aef_hourly_2024.csv", "grid_aef_predicted_kg_per_kwh"),
    "mef_soft": ("Soft MEF", "grid_mef_soft_hourly_2024.csv", "grid_mef_soft_kg_per_kwh"),
    "mef_upramp": ("UpRamp MEF", "grid_mef_upramp_hourly_2024.csv", "grid_mef_upramp_pred_kg_per_kwh"),
}


def _check_hourly_times(frame, full_year):
    times = pd.DatetimeIndex(pd.to_datetime(frame.datetime_utc, utc=True))
    if times.empty or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Hourly timestamps must be nonempty, unique and increasing")
    expected = pd.date_range("2024-01-01", periods=8784 if full_year else len(times), freq="h", tz="UTC")
    if not times.equals(expected):
        raise ValueError("Use a contiguous sequence from 1 January 2024 (8784 hours for the full year)")


def calculate_product_ci(model_files, model_ready, output, signals=("aef", "mef_soft", "mef_upramp"),
                         full_year=True):
    """Account one or more saved schedules; return (hourly_ledger, annual_summary).

    model_files maps readable model names to hourly_timeseries_all_days.csv.
    A partial check must start at 1 January with the study's initial inventories;
    it is marked partial rather than reported as an annual result.
    """
    output = Path(output).resolve()
    if "Csv files of input and output" in output.parts:
        raise ValueError("Write new CI outputs outside the supplied data folders")
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    if not model_files or not signals:
        raise ValueError("Choose at least one model and one emission signal")
    signal_data = {}
    for key in signals:
        if key not in SIGNALS:
            raise ValueError(f"Unknown emission signal: {key}")
        name, file, col = SIGNALS[key]
        raw = pd.read_csv(Path(model_ready) / file)
        raw.datetime_utc = pd.to_datetime(raw.datetime_utc, utc=True)
        _check_hourly_times(raw.sort_values("datetime_utc"), True)
        signal_data[name] = load_signal_df(Path(model_ready) / file, col)
    ledgers, summaries = [], []
    for model, file in model_files.items():
        d = load_model_df(Path(file))
        _check_hourly_times(d, full_year)
        d, _ = reconstruct_unit_grid_battery_split(d)
        d, _ = reconstruct_intermediates(d)
        for name, ef in signal_data.items():
            init = derive_zero_initial_cis(d, ef, name)
            _, hourly, daily, summary = run_carbon_ledger(d, name, ef, init)
            hourly.insert(0, "Signal", name); hourly.insert(0, "Model", model)
            hourly["cracker_flow_source"] = d.cracker_flow_source.to_numpy()
            annual = {"Model": model, "Signal": name,
                      "period_start_utc": str(hourly.datetime_utc.iloc[0]),
                      "period_end_utc": str(hourly.datetime_utc.iloc[-1]),
                      "full_year_2024": bool(full_year)}
            for p in PRODUCTS:
                annual[f"{p}_sold_mass_kg"] = float(daily[f"{p}_sold_total_kg"].sum())
                annual[f"{p}_sold_burden_kgco2"] = float(daily[f"{p}_carbon_g_kg"].sum() + daily[f"{p}_carbon_ng_kg"].sum())
                annual[f"{p}_sold_CI_kgco2_perkg"] = float(summary[f"{p}_sold_CI_kgco2_perkg"].iloc[0])
                annual[f"{p}_sold_green_share_pct"] = float(summary[f"{p}_sold_green_share_pct"].iloc[0])
            ledgers.append(hourly); summaries.append(annual)
            print(f"Calculated {model} / {name}")
    hourly = pd.concat(ledgers, ignore_index=True)
    annual = pd.DataFrame(summaries)
    for p in PRODUCTS:
        baseline = annual.loc[annual.Model.eq("M0")].set_index("Signal")[f"{p}_sold_CI_kgco2_perkg"].to_dict()
        annual[f"{p}_CI_reduction_vs_M0_pct"] = [
            100 * (1 - row[f"{p}_sold_CI_kgco2_perkg"] / baseline[row.Signal])
            if row.Signal in baseline and baseline[row.Signal] != 0 else np.nan
            for _, row in annual.iterrows()]
    output.mkdir(parents=True)
    hourly.to_csv(output / "hourly_carbon_ledger.csv", index=False)
    annual.to_csv(output / "annual_product_ci.csv", index=False)
    return hourly, annual


def hourly_statistics(hourly):
    """Empirical unweighted hourly statistics; quantiles linear, SD ddof=1."""
    records = []
    for (model, signal), group in hourly.groupby(["Model", "Signal"], sort=False):
        for p in PRODUCTS:
            s = group[f"{p}_sold_CI_kgco2_perkg_hourly"]
            valid = s.dropna()
            # Require actual adjacent hours, rather than joining across missing withdrawals.
            records.append({"Model": model, "Signal": signal, "Product": p,
                "n_withdrawal_hours": len(valid), "mean": valid.mean(), "median": valid.median(),
                "min": valid.min(), "max": valid.max(), "SD_ddof1": valid.std(ddof=1),
                "P05": valid.quantile(.05), "P25": valid.quantile(.25),
                "P75": valid.quantile(.75), "P95": valid.quantile(.95),
                "IQR": valid.quantile(.75)-valid.quantile(.25),
                "mean_abs_successive_hour_change": s.diff().abs().mean()})
    return pd.DataFrame(records)


def paired_hourly_statistics(hourly, model_signals=None):
    """Paired CI differences on common positive-sale hours, with M2 delivery weights."""
    records = []
    equality_tolerance = 10 ** (-12)
    matched = model_signals if model_signals is not None else {
        "M2-AEF0.5": "AEF", "M2-Soft MEF0.5": "Soft MEF",
        "M2-UpRamp MEF0.5": "UpRamp MEF",
    }
    for model, signal in matched.items():
        if model not in set(hourly.Model):
            continue
        base = hourly[(hourly.Model == "M0") & (hourly.Signal == signal)]
        case = hourly[(hourly.Model == model) & (hourly.Signal == signal)]
        if base.empty or case.empty:
            continue
        for p in PRODUCTS:
            ci = f"{p}_sold_CI_kgco2_perkg_hourly"
            mass = f"{p}_sold_total_kgph"
            pair = base[["datetime_utc", ci, mass]].merge(case[["datetime_utc", ci, mass]],
                on="datetime_utc", suffixes=("_M0", "_M2"), validate="one_to_one")
            pair = pair[(pair[mass+"_M0"] > TOL) & (pair[mass+"_M2"] > TOL)].dropna()
            delta = pair[ci+"_M2"] - pair[ci+"_M0"]
            positive = delta > equality_tolerance
            lower = delta < -equality_tolerance
            equal = ~(positive | lower)
            weights = pair[mass+"_M2"]
            pos = delta[positive]
            all_case_mass = case[mass].sum()
            records.append({"Model": model, "Signal": signal, "Product": p,
                "equality_tolerance_kgco2_perkg": equality_tolerance,
                "n_paired": len(pair), "n_higher": int(positive.sum()),
                "lower_hours_pct": 100*lower.mean(), "equal_hours_pct": 100*equal.mean(),
                "higher_hours_pct": 100*positive.mean(),
                "mean_delta": delta.mean(), "median_delta": delta.median(),
                "mass_weighted_paired_mean_delta": np.average(delta, weights=weights),
                "P05_delta": delta.quantile(.05), "P95_delta": delta.quantile(.95),
                "SD_delta_ddof1": delta.std(ddof=1), "min_delta": delta.min(), "max_delta": delta.max(),
                "mean_positive_delta": pos.mean(), "median_positive_delta": pos.median(),
                "P95_positive_delta": pos.quantile(.95),
                "higher_hour_M2_mass_kg": weights[positive].sum(),
                "higher_hour_M2_mass_pct_annual": 100*weights[positive].sum()/all_case_mass,
                "unpaired_M2_mass_kg": all_case_mass-weights.sum()})
    return pd.DataFrame(records)

