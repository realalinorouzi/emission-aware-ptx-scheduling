"""Daily rolling-horizon PtX scheduling: M0, M1 and M2.

Use run_optimization() from the principal notebook. Physical parameters remain
in each model's scalar block; only paths, horizon, signal and alpha are selected
by the calling notebook. Model definitions are retained separately for clarity.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

idx_2024 = pd.date_range("2024-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="h", tz="UTC")

def read_hourly_csv(path: Path, time_col="datetime_utc") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"{path.name}: missing '{time_col}'. Found: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).set_index(time_col)
    df = df.tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")
    return df


def assert_full_hourly(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.index.duplicated().any():
        dup = df.index[df.index.duplicated()][0]
        raise ValueError(f"{name}: duplicated timestamp example: {dup}")
    missing = idx_2024.difference(df.index)
    extra   = df.index.difference(idx_2024)
    if len(missing) > 0:
        raise ValueError(f"{name}: missing {len(missing)} hours (first 5: {missing[:5].tolist()})")
    if len(extra) > 0:
        raise ValueError(f"{name}: extra {len(extra)} hours outside 2024 (first 5: {extra[:5].tolist()})")

    df2 = df.reindex(idx_2024)
    df2.index.name = "datetime_utc"

    if df2.isna().any().any():
        cols = df2.columns[df2.isna().any()].tolist()
        where = df2[df2[cols].isna().any(axis=1)].index[:5].tolist()
        raise ValueError(f"{name}: NaNs after reindex in cols {cols}; example {where}")
    return df2


def day_to_time_slice(day_of_year: int):
    start_ts = idx_2024[0] + pd.Timedelta(days=day_of_year-1)
    end_ts = start_ts + pd.Timedelta(hours=24)
    return pd.date_range(start_ts, end_ts - pd.Timedelta(hours=1), freq="h", tz="UTC")


def safe_get_attr(model, attr, default=np.nan):
    try:
        return getattr(model, attr)
    except Exception:
        return default


def status_name(code: int) -> str:
    from gurobipy import GRB
    mp = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return mp.get(int(code), f"STATUS_{int(code)}")


def get_objpass_metrics(model, obj_idx: int):
    out = {
        "obj_idx": int(obj_idx),
        "pass_status": np.nan,
        "pass_status_name": "NA",
        "pass_runtime": np.nan,
        "pass_objval": np.nan,
        "pass_objbound": np.nan,
        "pass_mipgap": np.nan,
    }
    try:
        prev_objnum  = getattr(model.Params, "ObjNumber", None)
        prev_passnum = getattr(model.Params, "ObjPassNumber", None)

        model.Params.ObjNumber = int(obj_idx)
        model.Params.ObjPassNumber = -1

        st = safe_get_attr(model, "ObjPassNStatus", np.nan)
        out["pass_status"] = float(st) if not np.isnan(st) else np.nan
        out["pass_status_name"] = status_name(int(st)) if not np.isnan(st) else "NA"
        out["pass_runtime"]  = float(safe_get_attr(model, "ObjPassNRuntime", np.nan))
        out["pass_objval"]   = float(safe_get_attr(model, "ObjPassNObjVal", np.nan))
        out["pass_objbound"] = float(safe_get_attr(model, "ObjPassNObjBound", np.nan))
        out["pass_mipgap"]   = float(safe_get_attr(model, "ObjPassNMipGap", np.nan))

        if prev_objnum is not None:
            model.Params.ObjNumber = prev_objnum
        if prev_passnum is not None:
            model.Params.ObjPassNumber = prev_passnum
    except Exception:
        pass
    return out
def existing_solution_exports(t, namespace):
    """Extract the incumbent's existing flows/states, in kg/h, MW, and kg.

    Called only after optimize() and the original incumbent checks. `val` is
    the original extractor, so its handling of Gurobi solution attributes stays
    consistent with the rest of that model's solution extraction.
    """
    val = namespace["val"]
    out = {}
    fields = {
        "crk_nh3_in": "Q_crk_in",
        "crk_h2_out": "H_crk_out",
        "crk_nh3_return": "NH3_rec",
        "crk_h2_fuel": "H_crk_fuel",
        "H_to_NH3": "H_to_NH3",
        "H_to_MeOH": "H_to_MeOH",
    }
    for field, variable in fields.items():
        for label in ("g", "ng"):
            out[f"{field}_{label}_kgph"] = val(namespace[f"{variable}_{label}"][t])
    for product, variable in [("H2", "S_H2"), ("NH3", "S_NH3"), ("MeOH", "S_MEOH")]:
        for label in ("g", "ng"):
            out[f"tank_{product}_{label}_next_kg"] = val(namespace[f"{variable}_{label}"][t + 1])
    for unit, sink in [("elz", "elz"), ("nh3", "NH3"), ("meoh", "MeOH"), ("crk", "crk")]:
        for source in ("grid", "bat"):
            out[f"{source}_to_{unit}_mw"] = val(namespace["P"][source, sink, t])
    return out




# M0 formulation
def _run_m0(config):
    # ===================== M0 =====================

    MODEL_TAG = "M0"
    FORMULATION_TAG = "M0"
    SCENARIO_TAG = "S1"# we only used one scenario and demand in the work
    DEMAND_TAG = "D1"
    GREEN_TAG = "S1_green_split_with_rfnbo_hourly"

    import gurobipy as gp
    from gurobipy import GRB

    DATA_DIR = Path(config["model_ready"]).resolve().parent
    MR = Path(config["model_ready"]).resolve()

    # Required model-ready files (from utility cell)
    RES_CSV         = MR / "res_hourly_2024.csv"
    DEM_CSV         = MR / "demand_hourly_2024.csv"
    PRICE_CSV       = MR / "grid_price_hourly_2024.csv"
    AEF_CSV         = MR / "grid_aef_hourly_2024.csv"
    MEF_UPRAMP_CSV  = MR / "grid_mef_upramp_hourly_2024.csv"
    MEF_SOFT_CSV    = MR / "grid_mef_soft_hourly_2024.csv"
    PARAMS_JSON     = MR / "params_scenario_all.json"

    # column names from utility cell
    AEF_COL        = "grid_aef_predicted_kg_per_kwh"
    MEF_UPRAMP_COL = "grid_mef_upramp_pred_kg_per_kwh"
    MEF_SOFT_COL   = "grid_mef_soft_kg_per_kwh"

    # Run controls
    N_DAYS_TO_RUN = int(config.get("days", 1))
    START_DAY_OF_YEAR = 1  # Resume requires explicit carryover, not supported by this entry point.
    MIPGAP            = 0.01
    TIME_LIMIT_S = float(config.get("time_limit", 350))
    OUTPUTFLAG = int(config.get("solver_output", 1))


    def build_run_id():
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        return f"{MODEL_TAG}__{SCENARIO_TAG}__{DEMAND_TAG}__start{START_DAY_OF_YEAR:03d}__ndays{N_DAYS_TO_RUN:03d}__{stamp}"

    RUN_ID = build_run_id()

    OUT_BASE = Path(config["output"]).resolve()
    OUT_RUN   = OUT_BASE / RUN_ID
    OUT_POST  = OUT_RUN / "post"
    OUT_RUN.mkdir(parents=True, exist_ok=False)
    OUT_POST.mkdir(parents=True, exist_ok=True)

    res        = read_hourly_csv(RES_CSV)[["pv_mw","wind_mw","res_total_mw"]]
    dem        = read_hourly_csv(DEM_CSV)[["h2_demand_kgph","nh3_demand_kgph","meoh_demand_kgph"]]
    price      = read_hourly_csv(PRICE_CSV)[["da_elec_price_EURperMWh"]]
    aef        = read_hourly_csv(AEF_CSV)[[AEF_COL]]
    mef_upramp = read_hourly_csv(MEF_UPRAMP_CSV)[[MEF_UPRAMP_COL]]
    mef_soft   = read_hourly_csv(MEF_SOFT_CSV)[[MEF_SOFT_COL]]

    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    idx_2024 = pd.date_range("2024-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="h", tz="UTC")


    res        = assert_full_hourly(res, "res")
    dem        = assert_full_hourly(dem, "dem")
    price      = assert_full_hourly(price, "price")
    aef        = assert_full_hourly(aef, "aef")
    mef_upramp = assert_full_hourly(mef_upramp, "mef_upramp")
    mef_soft   = assert_full_hourly(mef_soft, "mef_soft")

    # ---------------- Scalars ----------------
    P_GRID_MAX_MW = 295.58

    P_BAT_MAX_MW = 50.0
    E_BAT_MAX_MWH = 100.32
    ETA_CH = 0.964
    ETA_DIS = 0.964
    SOC_MIN = 0.15
    SOC_MAX = 0.95
    SOC_INIT = 0.5
    #elz
    P_ELZ_MIN = 8 #mw
    P_ELZ_MAX = 52.25
    H_ELZ_MAX = 914.39161 #kg/h
    A_elz = -0.075045052
    B_elz = 21.73040904
    C_elz = -16.14458074
    N_stack = 4 #four modules
    omega_H2O = 0.015     # m3/kgH2
    c_H2O = 2.24          # EUR/m3
    C_su_elz = 1332  # Original runs mistakenly used EUR 1,332/module startup; intended/manuscript value: EUR 1,392.
    P_ELZ_MAX_TOT = N_stack * P_ELZ_MAX
    H_ELZ_MAX_TOT = N_stack * H_ELZ_MAX
    # Electrolyzer linear under-estimator
    H_min = A_elz*(P_ELZ_MIN**2) + B_elz*P_ELZ_MIN + C_elz
    H_max = A_elz*(P_ELZ_MAX**2) + B_elz*P_ELZ_MAX + C_elz
    Q1 = (H_max - H_min) / (P_ELZ_MAX - P_ELZ_MIN)
    Q0 = H_min - Q1 * P_ELZ_MIN
    #nh3
    Q_NH3_CAP = 13472.04
    f_NH3_min = 0.61
    RR_NH3 = 0.2
    mH2_to_NH3 = 0.18
    mN2_to_NH3 = 0.87
    SEC_HB = 0.36
    SEC_ASU = 0.068
    C_su_NH3 = 2397.46 # EUR/start
    de_NH3 = 0.00519
    #meoh
    Q_MEOH_CAP = 5818.856
    f_MEOH_min = 0.1
    R_MEOH = 0.5 * Q_MEOH_CAP
    mH2_to_MEOH = 0.22
    SEC_MEOH = 161.75/1000.0
    mCO2_to_MEOH = 1.59
    c_CO2 = 0.2
    C_su_MEOH = 1159.37  # EUR/start
    de_MEOH = 0.08
    #crk
    Q_CRK_CAP = 6062.416
    f_CRK_min = 0.05
    R_CRK = 0.5 * Q_CRK_CAP
    kH2_out = 0.12024
    kNH3_rec = 0.11848
    kH2_fuel = 0.035294
    SEC_CRK_E = 0.524667
    kSteam = 0.941176
    eSteam = 0.75
    omega_steam = 0.001

    C_su_CRK  = 1000.0      # EUR/start
    #tanks+EOD
    H_TANK = 12.0
    ALPHA_TANK_FILL = 0.15

    S_H2_CAP   = H_TANK * H_ELZ_MAX_TOT #kg
    S_NH3_CAP  = H_TANK * Q_NH3_CAP
    S_MEOH_CAP = H_TANK * Q_MEOH_CAP

    S_H2_INIT   = ALPHA_TANK_FILL * S_H2_CAP
    S_NH3_INIT  = ALPHA_TANK_FILL * S_NH3_CAP
    S_MEOH_INIT = ALPHA_TANK_FILL * S_MEOH_CAP

    carry = {
        "soc": SOC_INIT,
        "S_H2_g": 0.5*S_H2_INIT,
        "S_H2_ng": 0.5*S_H2_INIT,
        "S_NH3_g": 0.5*S_NH3_INIT,
        "S_NH3_ng": 0.5*S_NH3_INIT,
        "S_MEOH_g": 0.5*S_MEOH_INIT,
        "S_MEOH_ng": 0.5*S_MEOH_INIT,
        "y_elz_prev_mod": [0]*N_stack,
        "y_nh3_prev": 0,
        "y_meoh_prev": 0,
        "y_crk_prev": 0,
        "Q_nh3_prev": 0.0,
        "Q_meoh_prev": 0.0,
        "Q_crk_prev": 0.0,
    }

    daily_rows = []
    solver_rows = []
    state_rows = []
    all_hourly_frames = []
    # run loop
    t0_global = time.time()

    for d in range(START_DAY_OF_YEAR, START_DAY_OF_YEAR + N_DAYS_TO_RUN):
        day_hours = day_to_time_slice(d)

        if day_hours[0] < idx_2024[0] or day_hours[-1] > idx_2024[-1]:
            break


        pv_av    = res.loc[day_hours, "pv_mw"].values.astype(float)
        wd_av    = res.loc[day_hours, "wind_mw"].values.astype(float)
        pr_eur   = price.loc[day_hours, "da_elec_price_EURperMWh"].values.astype(float)
        dem_h2   = dem.loc[day_hours, "h2_demand_kgph"].values.astype(float)
        dem_nh3  = dem.loc[day_hours, "nh3_demand_kgph"].values.astype(float)
        dem_meoh = dem.loc[day_hours, "meoh_demand_kgph"].values.astype(float)

        aef_k        = aef.loc[day_hours, AEF_COL].values.astype(float)
        mef_upramp_k = mef_upramp.loc[day_hours, MEF_UPRAMP_COL].values.astype(float)
        mef_soft_k   = mef_soft.loc[day_hours, MEF_SOFT_COL].values.astype(float)

        m = gp.Model(f"{MODEL_TAG}_day{d:03d}")
        m.Params.MIPGap = MIPGAP
        m.Params.TimeLimit = TIME_LIMIT_S
        m.Params.OutputFlag = OUTPUTFLAG
        m.Params.NumericFocus = 1
        m.Params.MultiObjMethod = 2
        m.Params.MultiObjPre = 1

        T = range(24) #flows
        T2 = range(25) #states

        S_sources = ["pv", "wd", "grid", "bat"]
        K_sinks = ["elz", "NH3", "MeOH", "crk", "bch"]
        S_res = ["pv", "wd"] # S1: battery discharge does not counted green (accounting only)
        # Decision variables
        P = m.addVars(S_sources, K_sinks, T, lb=0.0, name="P_s2k")

        P_pv_cur = m.addVars(T, lb=0.0, name="P_pv_cur")
        P_wd_cur = m.addVars(T, lb=0.0, name="P_wd_cur")

        P_grid = m.addVars(T, lb=0.0, ub=P_GRID_MAX_MW, name="P_grid")

        P_ch  = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_ch")
        P_dis = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_dis")
        SOC   = m.addVars(T2, lb=SOC_MIN, ub=SOC_MAX, name="SOC")

        Pk_g  = m.addVars(K_sinks, T, lb=0.0, name="P_sink_g")
        Pk_ng = m.addVars(K_sinks, T, lb=0.0, name="P_sink_ng")

        I_elz = range(N_stack)

        y_elz_m  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="y_elz_m")
        su_elz_m = m.addVars(I_elz, T, vtype=GRB.BINARY, name="su_elz_m")
        # per-module "color choice": 1 => green-powered, 0 => non-green-powered (grid/bat depending on scenario definition)
        b_elz_g  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="b_elz_g")

        P_elz_m_g   = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_g")
        P_elz_m_ng  = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_ng")
        P_elz_m_tot = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_tot")

        H_elz_m    = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m")
        H_elz_m_g  = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_g")
        H_elz_m_ng = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_ng")

        y_elz    = m.addVars(T, vtype=GRB.BINARY, name="y_elz")
        n_elz_on = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="n_elz_on")
        su_elz   = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="su_elz")

        P_elz_g   = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_g")
        P_elz_ng  = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_ng")
        P_elz_tot = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_tot")
        H_prod    = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod")
        H_prod_g  = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_g")
        H_prod_ng = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_ng")
        W_elz     = m.addVars(T, lb=0.0, name="W_elz_m3ph")

        y_nh3   = m.addVars(T, vtype=GRB.BINARY, name="y_nh3")
        su_nh3  = m.addVars(T, vtype=GRB.BINARY, name="su_nh3")
        Q_nh3_g = m.addVars(T, lb=0.0, name="Q_nh3_g")
        Q_nh3_ng = m.addVars(T, lb=0.0, name="Q_nh3_ng")
        Q_nh3   = m.addVars(T, lb=0.0, name="Q_nh3")
        P_nh3_g = m.addVars(T, lb=0.0, name="P_nh3_g")
        P_nh3_ng = m.addVars(T, lb=0.0, name="P_nh3_ng")

        y_meoh   = m.addVars(T, vtype=GRB.BINARY, name="y_meoh")
        su_meoh  = m.addVars(T, vtype=GRB.BINARY, name="su_meoh")
        Q_meoh_g = m.addVars(T, lb=0.0, name="Q_meoh_g")
        Q_meoh_ng = m.addVars(T, lb=0.0, name="Q_meoh_ng")
        Q_meoh   = m.addVars(T, lb=0.0, name="Q_meoh")
        P_meoh_g = m.addVars(T, lb=0.0, name="P_meoh_g")
        P_meoh_ng = m.addVars(T, lb=0.0, name="P_meoh_ng")
        CO2_buy  = m.addVars(T, lb=0.0, name="CO2_buy_kgph")

        y_crk    = m.addVars(T, vtype=GRB.BINARY, name="y_crk")
        su_crk   = m.addVars(T, vtype=GRB.BINARY, name="su_crk")
        Q_crk_in_g = m.addVars(T, lb=0.0, name="Q_crk_in_g")
        Q_crk_in_ng = m.addVars(T, lb=0.0, name="Q_crk_in_ng")
        Q_crk_in = m.addVars(T, lb=0.0, name="Q_crk_in")

        H_crk_out_g  = m.addVars(T, lb=0.0, name="H_crk_out_g")
        H_crk_out_ng = m.addVars(T, lb=0.0, name="H_crk_out_ng")
        NH3_rec_g    = m.addVars(T, lb=0.0, name="NH3_rec_g")
        NH3_rec_ng   = m.addVars(T, lb=0.0, name="NH3_rec_ng")
        H_crk_fuel_g  = m.addVars(T, lb=0.0, name="H_crk_fuel_g")
        H_crk_fuel_ng = m.addVars(T, lb=0.0, name="H_crk_fuel_ng")
        Steam_g      = m.addVars(T, lb=0.0, name="Steam_g_kgph")
        Steam_ng     = m.addVars(T, lb=0.0, name="Steam_ng_kgph")
        P_crk_g      = m.addVars(T, lb=0.0, name="P_crk_g")
        P_crk_ng     = m.addVars(T, lb=0.0, name="P_crk_ng")
        W_steam      = m.addVars(T, lb=0.0, name="W_steam_m3ph")
        # Tanks (states, t=0..24)
        # one physical tank per product.
        # We track green/non-green "labels" as bookkeeping, but only the TOTAL is capacity constrained.
        S_H2_g   = m.addVars(T2, lb=0.0, name="S_H2_g")
        S_H2_ng  = m.addVars(T2, lb=0.0, name="S_H2_ng")
        S_NH3_g  = m.addVars(T2, lb=0.0, name="S_NH3_g")
        S_NH3_ng = m.addVars(T2, lb=0.0, name="S_NH3_ng")
        S_MEOH_g = m.addVars(T2, lb=0.0, name="S_MEOH_g")
        S_MEOH_ng = m.addVars(T2, lb=0.0, name="S_MEOH_ng")

        H_sell_g = m.addVars(T, lb=0.0, name="H_sell_g")
        H_sell_ng = m.addVars(T, lb=0.0, name="H_sell_ng")
        NH3_sell_g = m.addVars(T, lb=0.0, name="NH3_sell_g")
        NH3_sell_ng = m.addVars(T, lb=0.0, name="NH3_sell_ng")
        MEOH_sell_g = m.addVars(T, lb=0.0, name="MEOH_sell_g")
        MEOH_sell_ng = m.addVars(T, lb=0.0, name="MEOH_sell_ng")

        H_to_NH3_g = m.addVars(T, lb=0.0, name="H_to_NH3_g")
        H_to_NH3_ng = m.addVars(T, lb=0.0, name="H_to_NH3_ng")
        H_to_MEOH_g = m.addVars(T, lb=0.0, name="H_to_MEOH_g")
        H_to_MEOH_ng = m.addVars(T, lb=0.0, name="H_to_MEOH_ng")

        u_H2   = m.addVars(T, lb=0.0, name="u_H2")
        u_NH3  = m.addVars(T, lb=0.0, name="u_NH3")
        u_MEOH = m.addVars(T, lb=0.0, name="u_MEOH")
        #bounds
        for t in T:
            m.addConstr(P_pv_cur[t] <= pv_av[t], name=f"ub_pv_cur[{t}]")
            m.addConstr(P_wd_cur[t] <= wd_av[t], name=f"ub_wd_cur[{t}]")

        for t in T:
            for k in K_sinks:
                m.addConstr(P["pv", k, t]   <= pv_av[t], name=f"ub_P_pv_{k}[{t}]")
                m.addConstr(P["wd", k, t]   <= wd_av[t], name=f"ub_P_wd_{k}[{t}]")
                m.addConstr(P["grid", k, t] <= P_GRID_MAX_MW, name=f"ub_P_grid_{k}[{t}]")
                m.addConstr(P["bat", k, t]  <= P_BAT_MAX_MW, name=f"ub_P_bat_{k}[{t}]")

        for t in T:
            m.addConstr(gp.quicksum(P["pv",k,t] for k in K_sinks) + P_pv_cur[t] == pv_av[t], name=f"pv_balance[{t}]")
            m.addConstr(gp.quicksum(P["wd",k,t] for k in K_sinks) + P_wd_cur[t] == wd_av[t], name=f"wd_balance[{t}]")

            m.addConstr(gp.quicksum(P["grid",k,t] for k in K_sinks) == P_grid[t], name=f"grid_alloc[{t}]")
            m.addConstr(gp.quicksum(P["bat",k,t] for k in K_sinks) == P_dis[t], name=f"bat_alloc[{t}]")
            m.addConstr(gp.quicksum(P[s,"bch",t] for s in S_sources) == P_ch[t], name=f"bch_sink[{t}]")
            # Baseline rule: grid cannot charge battery; battery cannot charge battery
            m.addConstr(P["grid","bch",t] == 0.0, name=f"no_grid_to_bch[{t}]")
            m.addConstr(P["bat","bch",t] == 0.0, name=f"no_bat_to_bch[{t}]")
            m.addConstr(P["pv","bch",t] + P["wd","bch",t] == P_ch[t], name=f"bch_from_res[{t}]")

            m.addSOS(GRB.SOS_TYPE1, [P_ch[t], P_dis[t]])
            m.addConstr(P_dis[t] <= ETA_DIS * (SOC[t] - SOC_MIN) * E_BAT_MAX_MWH, name=f"ub_Pdis_soc[{t}]")
            m.addConstr(P_ch[t]  <= (SOC_MAX - SOC[t]) * E_BAT_MAX_MWH / ETA_CH, name=f"ub_Pch_soc[{t}]")

            for k in K_sinks:
                m.addConstr(Pk_ng[k,t] == P["grid",k,t] + P["bat",k,t], name=f"Pk_ng_def[{k},{t}]")
                m.addConstr(Pk_g[k,t] == gp.quicksum(P[s,k,t] for s in S_res), name=f"Pk_g_def[{k},{t}]")

        m.addConstr(SOC[0] == float(carry["soc"]), name="SOC_init_carry")
        for t in T:
            m.addConstr(
                SOC[t+1] == SOC[t] + (ETA_CH*P_ch[t] - (1.0/ETA_DIS)*P_dis[t]) / E_BAT_MAX_MWH,
                name=f"SOC_dyn[{t}]"
            )

        m.addConstrs((y_elz_m[i,t] >= y_elz_m[i+1,t] for i in range(N_stack-1) for t in T), name="elz_module_order")

        # ---------------- ELZ ----------------
        for t in T:
            m.addConstr(P_elz_g[t]  == Pk_g["elz",t],  name=f"P_elz_g_link[{t}]")
            m.addConstr(P_elz_ng[t] == Pk_ng["elz",t], name=f"P_elz_ng_link[{t}]")
            m.addConstr(P_elz_g[t]  == gp.quicksum(P_elz_m_g[i,t] for i in I_elz), name=f"P_elz_g_sum_mod[{t}]")
            m.addConstr(P_elz_ng[t] == gp.quicksum(P_elz_m_ng[i,t] for i in I_elz), name=f"P_elz_ng_sum_mod[{t}]")
            m.addConstr(P_elz_tot[t] == P_elz_g[t] + P_elz_ng[t], name=f"P_elz_tot_sum[{t}]")

            m.addConstr(n_elz_on[t] == gp.quicksum(y_elz_m[i,t] for i in I_elz), name=f"n_elz_on_def[{t}]")
            m.addConstr(n_elz_on[t] >= y_elz[t], name=f"y_elz_lb[{t}]")
            m.addConstr(n_elz_on[t] <= N_stack * y_elz[t], name=f"y_elz_ub[{t}]")
            m.addConstr(su_elz[t] == gp.quicksum(su_elz_m[i,t] for i in I_elz), name=f"su_elz_sum[{t}]")

            m.addConstr(H_prod[t]    == gp.quicksum(H_elz_m[i,t] for i in I_elz), name=f"H_prod_sum_mod[{t}]")
            m.addConstr(H_prod_g[t]  == gp.quicksum(H_elz_m_g[i,t] for i in I_elz), name=f"H_prod_g_sum_mod[{t}]")
            m.addConstr(H_prod_ng[t] == gp.quicksum(H_elz_m_ng[i,t] for i in I_elz), name=f"H_prod_ng_sum_mod[{t}]")
            m.addConstr(H_prod[t] == H_prod_g[t] + H_prod_ng[t], name=f"H_prod_split[{t}]")
            m.addConstr(W_elz[t] == omega_H2O * H_prod[t], name=f"W_elz[{t}]")

            for i in range(N_stack - 1):
                m.addConstr(b_elz_g[i,t] >= b_elz_g[i+1,t], name=f"elz_bg_order[{i},{t}]")

            for i in I_elz:
                m.addConstr(P_elz_m_tot[i,t] == P_elz_m_g[i,t] + P_elz_m_ng[i,t], name=f"P_elz_m_tot_def[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] <= P_ELZ_MAX * y_elz_m[i,t], name=f"elz_m_max[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] >= P_ELZ_MIN * y_elz_m[i,t], name=f"elz_m_min[{i},{t}]")
                m.addConstr(b_elz_g[i,t] <= y_elz_m[i,t], name=f"b_elz_g_gate[{i},{t}]")
                m.addConstr(P_elz_m_g[i,t]  <= P_ELZ_MAX * b_elz_g[i,t], name=f"P_elz_m_g_cap[{i},{t}]")
                m.addConstr(P_elz_m_ng[i,t] <= P_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"P_elz_m_ng_cap[{i},{t}]")
                m.addConstr(H_elz_m[i,t] == H_elz_m_g[i,t] + H_elz_m_ng[i,t], name=f"H_elz_m_split[{i},{t}]")

                m.addQConstr(
                    H_elz_m[i,t] <= A_elz*(P_elz_m_tot[i,t]*P_elz_m_tot[i,t]) + B_elz*P_elz_m_tot[i,t] + C_elz*y_elz_m[i,t],
                    name=f"elz_m_quad[{i},{t}]"
                )
                m.addConstr(H_elz_m[i,t] >= Q1*P_elz_m_tot[i,t] + Q0*y_elz_m[i,t], name=f"elz_m_lin[{i},{t}]")

                m.addConstr(H_elz_m_g[i,t] <= H_ELZ_MAX * b_elz_g[i,t], name=f"H_elz_m_g_ub1[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] <= H_elz_m[i,t], name=f"H_elz_m_g_ub2[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*(1 - b_elz_g[i,t]), name=f"H_elz_m_g_lb[{i},{t}]")

                m.addConstr(H_elz_m_ng[i,t] <= H_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"H_elz_m_ng_ub1[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] <= H_elz_m[i,t], name=f"H_elz_m_ng_ub2[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*b_elz_g[i,t], name=f"H_elz_m_ng_lb[{i},{t}]")
                # startup linkage per module
                if t >= 1:
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_elz_m[i,t-1], name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_elz_m[i,t-1], name=f"su_elz_m3[{i},{t}]")
                else:
                    y_prev_i = int(carry["y_elz_prev_mod"][i])
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_prev_i, name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_prev_i, name=f"su_elz_m3[{i},{t}]")

        # ---------------- NH3 ----------------
        for t in T:
            m.addConstr(Q_nh3[t] == Q_nh3_g[t] + Q_nh3_ng[t], name=f"Q_nh3_sum[{t}]")
            m.addConstr(Q_nh3[t] <= Q_NH3_CAP * y_nh3[t], name=f"nh3_max[{t}]")
            m.addConstr(Q_nh3[t] >= f_NH3_min * Q_NH3_CAP * y_nh3[t], name=f"nh3_min[{t}]")
            # Startup linkage
            if t >= 1:
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_nh3[t-1])
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_nh3[t-1])
            else:
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_prev)
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_prev)

            m.addConstr(
                Q_nh3[t] <= f_NH3_min * Q_NH3_CAP + (Q_NH3_CAP - f_NH3_min * Q_NH3_CAP) * (1 - su_nh3[t]),
                name=f"nh3_start_minload[{t}]"
            )

            ramp_nh3 = RR_NH3 * Q_NH3_CAP
            if t == 0:
                Q_prev = float(carry["Q_nh3_prev"])
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(Q_nh3[t] - Q_prev <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rup0[{t}]")
                m.addConstr(Q_prev - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rdn0[{t}]")
            else:
                m.addConstr(Q_nh3[t] - Q_nh3[t-1] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rup[{t}]")
                m.addConstr(Q_nh3[t-1] - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rdn[{t}]")

            m.addConstr(P_nh3_g[t]  == (SEC_HB*Q_nh3_g[t]  + SEC_ASU*(mN2_to_NH3*Q_nh3_g[t]))  / 1000.0, name=f"P_nh3_g_map[{t}]")
            m.addConstr(P_nh3_ng[t] == (SEC_HB*Q_nh3_ng[t] + SEC_ASU*(mN2_to_NH3*Q_nh3_ng[t])) / 1000.0, name=f"P_nh3_ng_map[{t}]")

            m.addConstr(P_nh3_g[t]  == Pk_g["NH3",t],  name=f"P_nh3_g_link[{t}]")
            m.addConstr(P_nh3_ng[t] == Pk_ng["NH3",t], name=f"P_nh3_ng_link[{t}]")

            m.addConstr(H_to_NH3_g[t]  == mH2_to_NH3 * Q_nh3_g[t],  name=f"H_to_NH3_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] == mH2_to_NH3 * Q_nh3_ng[t], name=f"H_to_NH3_ng[{t}]")

        # ---------------- MeOH ----------------
        for t in T:
            m.addConstr(Q_meoh[t] == Q_meoh_g[t] + Q_meoh_ng[t], name=f"Q_meoh_sum[{t}]")
            m.addConstr(Q_meoh[t] <= Q_MEOH_CAP * y_meoh[t], name=f"meoh_max[{t}]")
            m.addConstr(Q_meoh[t] >= f_MEOH_min * Q_MEOH_CAP * y_meoh[t], name=f"meoh_min[{t}]")

            if t >= 1:
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_meoh[t-1])
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_meoh[t-1])
            else:
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_prev)
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_prev)

            m.addConstr(
                Q_meoh[t] <= f_MEOH_min * Q_MEOH_CAP + (Q_MEOH_CAP - f_MEOH_min * Q_MEOH_CAP) * (1 - su_meoh[t]),
                name=f"meoh_start_minload[{t}]"
            )

            ramp_meoh = R_MEOH
            if t == 0:
                Q_prev = float(carry["Q_meoh_prev"])
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(Q_meoh[t] - Q_prev <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rup0[{t}]")
                m.addConstr(Q_prev - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rdn0[{t}]")
            else:
                m.addConstr(Q_meoh[t] - Q_meoh[t-1] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rup[{t}]")
                m.addConstr(Q_meoh[t-1] - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rdn[{t}]")

            m.addConstr(P_meoh_g[t]  == (SEC_MEOH * Q_meoh_g[t])  / 1000.0, name=f"P_meoh_g_map[{t}]")
            m.addConstr(P_meoh_ng[t] == (SEC_MEOH * Q_meoh_ng[t]) / 1000.0, name=f"P_meoh_ng_map[{t}]")

            m.addConstr(P_meoh_g[t]  == Pk_g["MeOH",t],  name=f"P_meoh_g_link[{t}]")
            m.addConstr(P_meoh_ng[t] == Pk_ng["MeOH",t], name=f"P_meoh_ng_link[{t}]")

            m.addConstr(CO2_buy[t] == mCO2_to_MEOH * Q_meoh[t], name=f"CO2_buy[{t}]")
            m.addConstr(H_to_MEOH_g[t]  == mH2_to_MEOH * Q_meoh_g[t],  name=f"H_to_MEOH_g[{t}]")
            m.addConstr(H_to_MEOH_ng[t] == mH2_to_MEOH * Q_meoh_ng[t], name=f"H_to_MEOH_ng[{t}]")

        # ---------------- Cracker ----------------
        for t in T:
            m.addConstr(Q_crk_in[t] == Q_crk_in_g[t] + Q_crk_in_ng[t], name=f"Q_crk_sum[{t}]")
            m.addConstr(Q_crk_in[t] <= Q_CRK_CAP * y_crk[t], name=f"crk_max[{t}]")
            m.addConstr(Q_crk_in[t] >= f_CRK_min * Q_CRK_CAP * y_crk[t], name=f"crk_min[{t}]")

            if t >= 1:
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_crk[t-1])
                m.addConstr(su_crk[t] >= y_crk[t] - y_crk[t-1])
            else:
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_prev)
                m.addConstr(su_crk[t] >= y_crk[t] - y_prev)

            m.addConstr(
                Q_crk_in[t] <= f_CRK_min * Q_CRK_CAP + (Q_CRK_CAP - f_CRK_min * Q_CRK_CAP) * (1 - su_crk[t]),
                name=f"crk_start_minload[{t}]"
            )

            ramp_crk = R_CRK
            if t == 0:
                Q_prev = float(carry["Q_crk_prev"])
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(Q_crk_in[t] - Q_prev <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rup0[{t}]")
                m.addConstr(Q_prev - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rdn0[{t}]")
            else:
                m.addConstr(Q_crk_in[t] - Q_crk_in[t-1] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rup[{t}]")
                m.addConstr(Q_crk_in[t-1] - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rdn[{t}]")

            m.addConstr(H_crk_out_g[t]  == kH2_out * Q_crk_in_g[t],  name=f"Hcrk_out_g[{t}]")
            m.addConstr(H_crk_out_ng[t] == kH2_out * Q_crk_in_ng[t], name=f"Hcrk_out_ng[{t}]")
            m.addConstr(NH3_rec_g[t]    == kNH3_rec * Q_crk_in_g[t],  name=f"NH3rec_g[{t}]")
            m.addConstr(NH3_rec_ng[t]   == kNH3_rec * Q_crk_in_ng[t], name=f"NH3rec_ng[{t}]")
            m.addConstr(H_crk_fuel_g[t]  == kH2_fuel * Q_crk_in_g[t],  name=f"Hcrk_fuel_g[{t}]")
            m.addConstr(H_crk_fuel_ng[t] == kH2_fuel * Q_crk_in_ng[t], name=f"Hcrk_fuel_ng[{t}]")
            m.addConstr(Steam_g[t]  == kSteam * Q_crk_in_g[t],  name=f"Steam_g[{t}]")
            m.addConstr(Steam_ng[t] == kSteam * Q_crk_in_ng[t], name=f"Steam_ng[{t}]")
            m.addConstr(W_steam[t]  == omega_steam * (Steam_g[t] + Steam_ng[t]), name=f"W_steam[{t}]")
            m.addConstr(P_crk_g[t]  == (SEC_CRK_E*Q_crk_in_g[t]  + eSteam*Steam_g[t])  / 1000.0, name=f"P_crk_g_map[{t}]")
            m.addConstr(P_crk_ng[t] == (SEC_CRK_E*Q_crk_in_ng[t] + eSteam*Steam_ng[t]) / 1000.0, name=f"P_crk_ng_map[{t}]")
            m.addConstr(P_crk_g[t]  == Pk_g["crk",t],  name=f"P_crk_g_link[{t}]")
            m.addConstr(P_crk_ng[t] == Pk_ng["crk",t], name=f"P_crk_ng_link[{t}]")

        # ---------------- Tanks and demand ----------------
        m.addConstr(S_H2_g[0]   == float(carry["S_H2_g"]), name="S_H2_g_init")
        m.addConstr(S_H2_ng[0]  == float(carry["S_H2_ng"]), name="S_H2_ng_init")
        m.addConstr(S_NH3_g[0]  == float(carry["S_NH3_g"]), name="S_NH3_g_init")
        m.addConstr(S_NH3_ng[0] == float(carry["S_NH3_ng"]), name="S_NH3_ng_init")
        m.addConstr(S_MEOH_g[0] == float(carry["S_MEOH_g"]), name="S_MEOH_g_init")
        m.addConstr(S_MEOH_ng[0] == float(carry["S_MEOH_ng"]), name="S_MEOH_ng_init")

        for tt in T2:
            m.addConstr(S_H2_g[tt] + S_H2_ng[tt] <= S_H2_CAP, name=f"S_H2_cap[{tt}]")
            m.addConstr(S_NH3_g[tt] + S_NH3_ng[tt] <= S_NH3_CAP, name=f"S_NH3_cap[{tt}]")
            m.addConstr(S_MEOH_g[tt] + S_MEOH_ng[tt] <= S_MEOH_CAP, name=f"S_MEOH_cap[{tt}]")
        # ---------------- Hard EOD minimum inventory (fraction of tank capacity) ----------------
        m.addConstr(S_H2_g[24] + S_H2_ng[24] >= ALPHA_TANK_FILL * S_H2_CAP, name="eod_hard_H2_frac")
        m.addConstr(S_NH3_g[24] + S_NH3_ng[24] >= ALPHA_TANK_FILL * S_NH3_CAP, name="eod_hard_NH3_frac")
        m.addConstr(S_MEOH_g[24] + S_MEOH_ng[24] >= ALPHA_TANK_FILL * S_MEOH_CAP, name="eod_hard_MeOH_frac")

        for t in T:
            m.addConstr(H_sell_g[t] + H_sell_ng[t] + u_H2[t] == dem_h2[t], name=f"dem_H2[{t}]")
            m.addConstr(NH3_sell_g[t] + NH3_sell_ng[t] + u_NH3[t] == dem_nh3[t], name=f"dem_NH3[{t}]")
            m.addConstr(MEOH_sell_g[t] + MEOH_sell_ng[t] + u_MEOH[t] == dem_meoh[t], name=f"dem_MEOH[{t}]")

            m.addConstr(S_H2_g[t+1] == S_H2_g[t] + H_prod_g[t] + H_crk_out_g[t] - H_to_NH3_g[t] - H_to_MEOH_g[t] - H_sell_g[t] - H_crk_fuel_g[t], name=f"S_H2_g_dyn[{t}]")
            m.addConstr(S_H2_ng[t+1] == S_H2_ng[t] + H_prod_ng[t] + H_crk_out_ng[t] - H_to_NH3_ng[t] - H_to_MEOH_ng[t] - H_sell_ng[t] - H_crk_fuel_ng[t], name=f"S_H2_ng_dyn[{t}]")

            m.addConstr(S_NH3_g[t+1] == S_NH3_g[t] + Q_nh3_g[t] + NH3_rec_g[t] - NH3_sell_g[t] - Q_crk_in_g[t], name=f"S_NH3_g_dyn[{t}]")
            m.addConstr(S_NH3_ng[t+1] == S_NH3_ng[t] + Q_nh3_ng[t] + NH3_rec_ng[t] - NH3_sell_ng[t] - Q_crk_in_ng[t], name=f"S_NH3_ng_dyn[{t}]")

            m.addConstr(S_MEOH_g[t+1] == S_MEOH_g[t] + Q_meoh_g[t] - MEOH_sell_g[t], name=f"S_MEOH_g_dyn[{t}]")
            m.addConstr(S_MEOH_ng[t+1] == S_MEOH_ng[t] + Q_meoh_ng[t] - MEOH_sell_ng[t], name=f"S_MEOH_ng_dyn[{t}]")

            m.addConstr(H_to_NH3_g[t] + H_to_MEOH_g[t] + H_sell_g[t] + H_crk_fuel_g[t] <= S_H2_g[t], name=f"dir_H2_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] + H_to_MEOH_ng[t] + H_sell_ng[t] + H_crk_fuel_ng[t] <= S_H2_ng[t], name=f"dir_H2_ng[{t}]")
            m.addConstr(NH3_sell_g[t] + Q_crk_in_g[t] <= S_NH3_g[t], name=f"dir_NH3_g[{t}]")
            m.addConstr(NH3_sell_ng[t] + Q_crk_in_ng[t] <= S_NH3_ng[t], name=f"dir_NH3_ng[{t}]")
            m.addConstr(MEOH_sell_g[t] <= S_MEOH_g[t], name=f"dir_MEOH_g[{t}]")
            m.addConstr(MEOH_sell_ng[t] <= S_MEOH_ng[t], name=f"dir_MEOH_ng[{t}]")
        # Priority 1: minimize total unmet demand (kg over the day)
        unmet_total = gp.quicksum(u_H2[t] + u_NH3[t] + u_MEOH[t] for t in T)
        # Priority 2: minimize operating cost (EUR) among solutions with minimal unmet
        cost_total = gp.LinExpr()
        for t in T:
            cost_total += pr_eur[t] * P_grid[t]
            cost_total += c_H2O * (W_elz[t] + W_steam[t])
            cost_total += c_CO2 * CO2_buy[t]
            cost_total += C_su_elz * gp.quicksum(su_elz_m[i,t] for i in I_elz)
            cost_total += C_su_NH3 * su_nh3[t]
            cost_total += C_su_MEOH * su_meoh[t]
            cost_total += C_su_CRK * su_crk[t]

        m.setObjectiveN(unmet_total, index=0, priority=2, reltol=1e-4, abstol=1e-6, name="OBJ1_min_unmet")
        m.setObjectiveN(cost_total, index=1, priority=1, reltol=1e-4, abstol=1e-6, name="OBJ2_min_cost")

        m.optimize()

        status = int(safe_get_attr(m, "Status", -1))
        runtime = float(safe_get_attr(m, "Runtime", np.nan))
        nodecnt = float(safe_get_attr(m, "NodeCount", np.nan))
        solcnt  = int(safe_get_attr(m, "SolCount", -1))
        # index=0 is unmet, index=1 is cost (for M0)
        pass0 = get_objpass_metrics(m, 0)
        pass1 = get_objpass_metrics(m, 1)

        mip_gap = pass1["pass_mipgap"]
        obj_val = pass1["pass_objval"]
        obj_bnd = pass1["pass_objbound"]

        itercnt = safe_get_attr(m, "IterCount", np.nan)
        bariter = safe_get_attr(m, "BarIterCount", np.nan)
        work = safe_get_attr(m, "Work", np.nan)

        obj1_val = np.nan
        obj2_val = np.nan
        try:
            obj1_val = float(unmet_total.getValue())
        except Exception:
            pass
        try:
            obj2_val = float(cost_total.getValue())
        except Exception:
            pass

        is_optimal = (status == GRB.OPTIMAL)
        hit_time_limit = (status == GRB.TIME_LIMIT)
        has_incumbent = (solcnt is not None) and (solcnt >= 1)
        is_within_target_gap = bool(has_incumbent) and (mip_gap is not None) and (not np.isnan(mip_gap)) and (float(mip_gap) <= float(MIPGAP) + 1e-12)
        is_acceptable = bool(is_optimal or is_within_target_gap)

        solver_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "solver_status": status,
            "solver_status_name": status_name(status),
            "runtime_seconds": runtime,
            "time_limit_s": TIME_LIMIT_S,
            "mipgap_target": float(MIPGAP),
            "mip_gap": pass1["pass_mipgap"],
            "obj0_pass_status": pass0["pass_status"],
            "obj0_pass_status_name": pass0["pass_status_name"],
            "obj0_pass_runtime_s": pass0["pass_runtime"],
            "obj0_pass_objval": pass0["pass_objval"],
            "obj0_pass_objbound": pass0["pass_objbound"],
            "obj0_pass_mipgap": pass0["pass_mipgap"],
            "obj1_pass_status": pass1["pass_status"],
            "obj1_pass_status_name": pass1["pass_status_name"],
            "obj1_pass_runtime_s": pass1["pass_runtime"],
            "obj1_pass_objval": pass1["pass_objval"],
            "obj1_pass_objbound": pass1["pass_objbound"],
            "obj1_pass_mipgap": pass1["pass_mipgap"],
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),
            "obj_val": float(obj_val) if not np.isnan(obj_val) else np.nan,
            "obj_bound": float(obj_bnd) if not np.isnan(obj_bnd) else np.nan,
            "node_count": nodecnt,
            "sol_count": solcnt,
            "iter_count": float(itercnt) if not np.isnan(itercnt) else np.nan,
            "bar_iter_count": float(bariter) if not np.isnan(bariter) else np.nan,
            "work": float(work) if not np.isnan(work) else np.nan,
            "obj1_unmet_value": float(obj1_val) if not np.isnan(obj1_val) else np.nan,
            "obj2_cost_value": float(obj2_val) if not np.isnan(obj2_val) else np.nan,
        })

        if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            try:
                m.computeIIS()
                iis_path = day_dir / f"{MODEL_TAG}_day{d:03d}_iis.ilp"
                m.write(str(iis_path))
            except Exception:
                pass
            raise RuntimeError(f"Day {d}: solver status {status} (not optimal/time_limit).")

        def val(v):
            return float(v.X)

        rows = []
        unmet_rows = []

        for t in T:
            ts = day_hours[t]
            pv_used = pv_av[t] - val(P_pv_cur[t])
            wd_used = wd_av[t] - val(P_wd_cur[t])
            grid_mw = val(P_grid[t])
            res_used_mw = pv_used + wd_used
            res_curt_mw = val(P_pv_cur[t]) + val(P_wd_cur[t])

            P_elz = val(P_elz_g[t]) + val(P_elz_ng[t])
            P_nh3 = val(P_nh3_g[t]) + val(P_nh3_ng[t])
            P_meo = val(P_meoh_g[t]) + val(P_meoh_ng[t])
            P_crk = val(P_crk_g[t]) + val(P_crk_ng[t])

            H_total = val(H_prod[t])
            H_g = val(H_prod_g[t])
            H_ng = val(H_prod_ng[t])

            nh3_total = val(Q_nh3[t])
            nh3_g = val(Q_nh3_g[t])
            nh3_ng = val(Q_nh3_ng[t])

            me_total = val(Q_meoh[t])
            me_g = val(Q_meoh_g[t])
            me_ng = val(Q_meoh_ng[t])

            h2_sell_g = val(H_sell_g[t])
            h2_sell_ng = val(H_sell_ng[t])
            nh3_sell_g = val(NH3_sell_g[t])
            nh3_sell_ng = val(NH3_sell_ng[t])
            meoh_sell_g = val(MEOH_sell_g[t])
            meoh_sell_ng = val(MEOH_sell_ng[t])

            u1 = val(u_H2[t])
            u2 = val(u_NH3[t])
            u3 = val(u_MEOH[t])

            def share(g, ng):
                tot = g + ng
                return (g / tot) if tot > 1e-9 else np.nan

            H2_min_to_NH3  = mH2_to_NH3 * f_NH3_min * Q_NH3_CAP * int(round(val(y_nh3[t])))
            H2_min_to_MeOH = mH2_to_MEOH * f_MEOH_min * Q_MEOH_CAP * int(round(val(y_meoh[t])))
            H2_min_to_CRK  = kH2_fuel * f_CRK_min * Q_CRK_CAP * int(round(val(y_crk[t])))
            H2_min_committed = H2_min_to_NH3 + H2_min_to_MeOH + H2_min_to_CRK
            H2_tank_start = val(S_H2_g[t]) + val(S_H2_ng[t])
            H2_free_margin = H2_tank_start - H2_min_committed

            elz_mod = {}
            for i in I_elz:
                elz_mod[f"elz_m{i}_y"]     = int(round(val(y_elz_m[i,t])))
                elz_mod[f"elz_m{i}_su"]    = int(round(val(su_elz_m[i,t])))
                elz_mod[f"elz_m{i}_b_g"]   = int(round(val(b_elz_g[i,t])))
                elz_mod[f"elz_m{i}_P_tot"] = float(val(P_elz_m_tot[i,t]))
                elz_mod[f"elz_m{i}_P_g"]   = float(val(P_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_P_ng"]  = float(val(P_elz_m_ng[i,t]))
                elz_mod[f"elz_m{i}_H_tot"] = float(val(H_elz_m[i,t]))
                elz_mod[f"elz_m{i}_H_g"]   = float(val(H_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_H_ng"]  = float(val(H_elz_m_ng[i,t]))

            rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "H2_tank_start_kg": H2_tank_start,
                "H2_min_committed_kgph": H2_min_committed,
                "H2_free_margin_kg": H2_free_margin,
                "pv_avail_mw": pv_av[t],
                "wind_avail_mw": wd_av[t],
                "pv_curt_mw": val(P_pv_cur[t]),
                "wind_curt_mw": val(P_wd_cur[t]),
                "res_used_mw": res_used_mw,
                "res_curtailed_mw": res_curt_mw,
                "grid_import_mw": grid_mw,
                "da_price_eur_per_mwh": pr_eur[t],
                "bat_ch_mw": val(P_ch[t]),
                "bat_dis_mw": val(P_dis[t]),
                "battery_soc": val(SOC[t]),
                "battery_soc_next": val(SOC[t+1]),
                "elz_grid_mw": val(P_elz_ng[t]),
                "nh3_grid_mw": val(P_nh3_ng[t]),
                "meoh_grid_mw": val(P_meoh_ng[t]),
                "crk_grid_mw": val(P_crk_ng[t]),
                "P_elz_mw": P_elz,
                "P_nh3_mw": P_nh3,
                "P_meoh_mw": P_meo,
                "P_crk_mw": P_crk,
                "elz_power_green_share": share(val(P_elz_g[t]), val(P_elz_ng[t])),
                "nh3_power_green_share": share(val(P_nh3_g[t]), val(P_nh3_ng[t])),
                "meoh_power_green_share": share(val(P_meoh_g[t]), val(P_meoh_ng[t])),
                "crk_power_green_share": share(val(P_crk_g[t]), val(P_crk_ng[t])),
                "H2_prod_kgph": H_total,
                "H2_prod_g_kgph": H_g,
                "H2_prod_ng_kgph": H_ng,
                "NH3_prod_kgph": nh3_total,
                "NH3_prod_g_kgph": nh3_g,
                "NH3_prod_ng_kgph": nh3_ng,
                "MeOH_prod_kgph": me_total,
                "MeOH_prod_g_kgph": me_g,
                "MeOH_prod_ng_kgph": me_ng,
                "crk_nh3_in_kgph": val(Q_crk_in[t]),
                "crk_h2_out_kgph": val(H_crk_out_g[t]) + val(H_crk_out_ng[t]),
                "crk_h2_fuel_kgph": val(H_crk_fuel_g[t]) + val(H_crk_fuel_ng[t]),
                "steam_kgph": val(Steam_g[t]) + val(Steam_ng[t]),
                "water_elz_m3ph": val(W_elz[t]),
                "water_steam_m3ph": val(W_steam[t]),
                "co2_buy_kgph": val(CO2_buy[t]),

                "H2_sell_g_kgph": h2_sell_g,
                "H2_sell_ng_kgph": h2_sell_ng,
                "H2_sell_kgph": h2_sell_g + h2_sell_ng,

                "NH3_sell_g_kgph": nh3_sell_g,
                "NH3_sell_ng_kgph": nh3_sell_ng,
                "NH3_sell_kgph": nh3_sell_g + nh3_sell_ng,

                "MeOH_sell_g_kgph": meoh_sell_g,
                "MeOH_sell_ng_kgph": meoh_sell_ng,
                "MeOH_sell_kgph": meoh_sell_g + meoh_sell_ng,

                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "tank_H2_g_kg": val(S_H2_g[t]),
                "tank_H2_ng_kg": val(S_H2_ng[t]),
                "tank_NH3_g_kg": val(S_NH3_g[t]),
                "tank_NH3_ng_kg": val(S_NH3_ng[t]),
                "tank_MeOH_g_kg": val(S_MEOH_g[t]),
                "tank_MeOH_ng_kg": val(S_MEOH_ng[t]),
                "b_ch": int(val(P_ch[t]) > 1e-6),
                "n_elz_on": int(round(val(n_elz_on[t]))),
                "y_elz": int(round(val(y_elz[t]))),
                "su_elz": int(round(val(su_elz[t]))),
                "y_nh3": int(round(val(y_nh3[t]))),
                "su_nh3": int(round(val(su_nh3[t]))),
                "y_meoh": int(round(val(y_meoh[t]))),
                "su_meoh": int(round(val(su_meoh[t]))),
                "y_crk": int(round(val(y_crk[t]))),
                "su_crk": int(round(val(su_crk[t]))),
                **existing_solution_exports(t, locals()),
                **elz_mod,
            })

            unmet_rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "da_price_eur_per_mwh": pr_eur[t],
                "grid_import_mw": grid_mw,
                "res_curtailed_mw": res_curt_mw,
            })

        day_hourly = pd.DataFrame(rows)


        day_hourly_all = day_hourly.copy()
        day_hourly_all["day_id"] = int(d)
        all_hourly_frames.append(day_hourly_all)

        # ---------------- Daily post-processing ----------------
        grid_cost_eur   = float(np.sum([pr_eur[t] * val(P_grid[t]) for t in T]))
        water_cost_eur  = float(c_H2O * np.sum([val(W_elz[t]) + val(W_steam[t]) for t in T]))
        co2_cost_eur    = float(c_CO2 * np.sum([val(CO2_buy[t]) for t in T]))
        startup_elz_eur = float(C_su_elz * np.sum([val(su_elz[t]) for t in T]))
        startup_nh3_eur = float(C_su_NH3 * np.sum([val(su_nh3[t]) for t in T]))
        startup_meoh_eur = float(C_su_MEOH * np.sum([val(su_meoh[t]) for t in T]))
        startup_crk_eur = float(C_su_CRK * np.sum([val(su_crk[t]) for t in T]))

        total_unmet_kg = float(np.sum(day_hourly["unmet_h2_kgph"].values + day_hourly["unmet_nh3_kgph"].values + day_hourly["unmet_meoh_kgph"].values))
        total_cost = float(obj2_val) if not np.isnan(obj2_val) else float(grid_cost_eur + water_cost_eur + co2_cost_eur + startup_elz_eur + startup_nh3_eur + startup_meoh_eur + startup_crk_eur)

        grid_emis_aef = float(np.sum([val(P_grid[t]) * 1000.0 * aef_k[t] for t in T]))
        grid_emis_mef_upramp = float(np.sum([val(P_grid[t]) * 1000.0 * mef_upramp_k[t] for t in T]))
        grid_emis_mef_soft = float(np.sum([val(P_grid[t]) * 1000.0 * mef_soft_k[t] for t in T]))

        direct_emis_nh3 = float(np.sum([de_NH3 * val(Q_nh3[t]) for t in T]))
        direct_emis_meoh = float(np.sum([de_MEOH * val(Q_meoh[t]) for t in T]))
        direct_emis_total = direct_emis_nh3 + direct_emis_meoh

        demand_H2_kg_day   = float(np.sum(dem_h2))
        demand_NH3_kg_day  = float(np.sum(dem_nh3))
        demand_MeOH_kg_day = float(np.sum(dem_meoh))

        elz_kwh_day  = float(np.sum(day_hourly["P_elz_mw"].values * 1000.0))
        nh3_kwh_day  = float(np.sum(day_hourly["P_nh3_mw"].values * 1000.0))
        meoh_kwh_day = float(np.sum(day_hourly["P_meoh_mw"].values * 1000.0))
        crk_kwh_day  = float(np.sum(day_hourly["P_crk_mw"].values * 1000.0))
        units_kwh_day = elz_kwh_day + nh3_kwh_day + meoh_kwh_day + crk_kwh_day

        res_gen_kwh = float(np.sum((pv_av + wd_av) * 1000.0))
        res_used_kwh = float(np.sum((pv_av - day_hourly["pv_curt_mw"].values + wd_av - day_hourly["wind_curt_mw"].values) * 1000.0))
        res_curt_kwh = float(np.sum((day_hourly["pv_curt_mw"].values + day_hourly["wind_curt_mw"].values) * 1000.0))
        grid_import_kwh = float(np.sum(day_hourly["grid_import_mw"].values * 1000.0))
        grid_share_percent = 100.0 * grid_import_kwh / max(grid_import_kwh + res_used_kwh, 1e-9)

        H2_g_sum = float(np.sum(day_hourly["H2_prod_g_kgph"].values))
        H2_ng_sum = float(np.sum(day_hourly["H2_prod_ng_kgph"].values))
        NH3_g_sum = float(np.sum(day_hourly["NH3_prod_g_kgph"].values))
        NH3_ng_sum = float(np.sum(day_hourly["NH3_prod_ng_kgph"].values))
        Me_g_sum = float(np.sum(day_hourly["MeOH_prod_g_kgph"].values))
        Me_ng_sum = float(np.sum(day_hourly["MeOH_prod_ng_kgph"].values))

        def green_share(sum_g, sum_ng):
            tot = sum_g + sum_ng
            return (sum_g / tot) if tot > 1e-9 else np.nan

        daily_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "mip_gap": float(mip_gap) if mip_gap is not None else np.nan,
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),

            "total_cost_eur": total_cost,
            "obj1_total_unmet_kg": total_unmet_kg,
            "obj2_operating_cost_eur": total_cost,

            "res_generation_kWh": res_gen_kwh,
            "res_used_kWh": res_used_kwh,
            "res_curtailed_kWh": res_curt_kwh,
            "grid_import_kWh": grid_import_kwh,
            "grid_share_percent": grid_share_percent,

            "battery_soc_avg": float(np.mean(day_hourly["battery_soc"].values)),
            "tank_H2_level_avg_kg": float(np.mean(day_hourly["tank_H2_g_kg"].values + day_hourly["tank_H2_ng_kg"].values)),
            "tank_NH3_level_avg_kg": float(np.mean(day_hourly["tank_NH3_g_kg"].values + day_hourly["tank_NH3_ng_kg"].values)),
            "tank_MeOH_level_avg_kg": float(np.mean(day_hourly["tank_MeOH_g_kg"].values + day_hourly["tank_MeOH_ng_kg"].values)),

            "H2_prod_green_share": green_share(H2_g_sum, H2_ng_sum),
            "NH3_prod_green_share": green_share(NH3_g_sum, NH3_ng_sum),
            "MeOH_prod_green_share": green_share(Me_g_sum, Me_ng_sum),

            # grid emissions only, by signal
            "emissions_grid_aef_kg": grid_emis_aef,
            "emissions_grid_mef_upramp_kg": grid_emis_mef_upramp,
            "emissions_grid_mef_soft_kg": grid_emis_mef_soft,

            # direct emissions separate
            "emissions_direct_nh3_kg": direct_emis_nh3,
            "emissions_direct_meoh_kg": direct_emis_meoh,
            "emissions_direct_total_kg": direct_emis_total,

            "unmet_H2_kg": float(np.sum(day_hourly["unmet_h2_kgph"].values)),
            "unmet_NH3_kg": float(np.sum(day_hourly["unmet_nh3_kgph"].values)),
            "unmet_MeOH_kg": float(np.sum(day_hourly["unmet_meoh_kgph"].values)),

            "demand_H2_kg_day": demand_H2_kg_day,
            "demand_NH3_kg_day": demand_NH3_kg_day,
            "demand_MeOH_kg_day": demand_MeOH_kg_day,

            "elz_consumption_kWh_day": elz_kwh_day,
            "nh3_consumption_kWh_day": nh3_kwh_day,
            "meoh_consumption_kWh_day": meoh_kwh_day,
            "crk_consumption_kWh_day": crk_kwh_day,
            "units_total_consumption_kWh_day": units_kwh_day,

            # sales accounting
            "H2_sell_g_kg_day": float(np.sum(day_hourly["H2_sell_g_kgph"].values)),
            "H2_sell_ng_kg_day": float(np.sum(day_hourly["H2_sell_ng_kgph"].values)),
            "H2_sell_total_kg_day": float(np.sum(day_hourly["H2_sell_kgph"].values)),

            "NH3_sell_g_kg_day": float(np.sum(day_hourly["NH3_sell_g_kgph"].values)),
            "NH3_sell_ng_kg_day": float(np.sum(day_hourly["NH3_sell_ng_kgph"].values)),
            "NH3_sell_total_kg_day": float(np.sum(day_hourly["NH3_sell_kgph"].values)),

            "MeOH_sell_g_kg_day": float(np.sum(day_hourly["MeOH_sell_g_kgph"].values)),
            "MeOH_sell_ng_kg_day": float(np.sum(day_hourly["MeOH_sell_ng_kgph"].values)),
            "MeOH_sell_total_kg_day": float(np.sum(day_hourly["MeOH_sell_kgph"].values)),

            "eod_H2_total_kg": float(val(S_H2_g[24]) + val(S_H2_ng[24])),
            "eod_NH3_total_kg": float(val(S_NH3_g[24]) + val(S_NH3_ng[24])),
            "eod_MeOH_total_kg": float(val(S_MEOH_g[24]) + val(S_MEOH_ng[24])),
            "eod_req_H2_kg": float(ALPHA_TANK_FILL * S_H2_CAP),
            "eod_req_NH3_kg": float(ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_req_MeOH_kg": float(ALPHA_TANK_FILL * S_MEOH_CAP),
            "eod_margin_H2_kg": float((val(S_H2_g[24]) + val(S_H2_ng[24])) - ALPHA_TANK_FILL * S_H2_CAP),
            "eod_margin_NH3_kg": float((val(S_NH3_g[24]) + val(S_NH3_ng[24])) - ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_margin_MeOH_kg": float((val(S_MEOH_g[24]) + val(S_MEOH_ng[24])) - ALPHA_TANK_FILL * S_MEOH_CAP),

            "cost_grid_eur": grid_cost_eur,
            "cost_water_eur": water_cost_eur,
            "cost_co2_eur": co2_cost_eur,
            "cost_startup_elz_eur": startup_elz_eur,
            "cost_startup_nh3_eur": startup_nh3_eur,
            "cost_startup_meoh_eur": startup_meoh_eur,
            "cost_startup_crk_eur": startup_crk_eur,
        })

        carry["soc"] = val(SOC[24])
        carry["S_H2_g"] = val(S_H2_g[24]); carry["S_H2_ng"] = val(S_H2_ng[24])
        carry["S_NH3_g"] = val(S_NH3_g[24]); carry["S_NH3_ng"] = val(S_NH3_ng[24])
        carry["S_MEOH_g"] = val(S_MEOH_g[24]); carry["S_MEOH_ng"] = val(S_MEOH_ng[24])
        carry["y_elz_prev_mod"] = [int(round(val(y_elz_m[i,23]))) for i in I_elz]
        carry["y_nh3_prev"]  = int(round(val(y_nh3[23])))
        carry["y_meoh_prev"] = int(round(val(y_meoh[23])))
        carry["y_crk_prev"]  = int(round(val(y_crk[23])))
        carry["Q_nh3_prev"]  = float(val(Q_nh3[23]))
        carry["Q_meoh_prev"] = float(val(Q_meoh[23]))
        carry["Q_crk_prev"]  = float(val(Q_crk_in[23]))

        state_rows.append({
            "run_id": RUN_ID,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "SOC_end": carry["soc"],
            "S_H2_g_end": carry["S_H2_g"],
            "S_H2_ng_end": carry["S_H2_ng"],
            "S_NH3_g_end": carry["S_NH3_g"],
            "S_NH3_ng_end": carry["S_NH3_ng"],
            "S_MEOH_g_end": carry["S_MEOH_g"],
            "S_MEOH_ng_end": carry["S_MEOH_ng"],
            "y_elz_prev_mod_end": carry["y_elz_prev_mod"],
            "y_nh3_end": carry["y_nh3_prev"],
            "y_meoh_end": carry["y_meoh_prev"],
            "y_crk_end": carry["y_crk_prev"],
            "Q_nh3_end": carry["Q_nh3_prev"],
            "Q_meoh_end": carry["Q_meoh_prev"],
            "Q_crk_end": carry["Q_crk_prev"],
        })

        eps = 1e-6
        sim_cd = float(np.sum((day_hourly["bat_ch_mw"].values > eps) & (day_hourly["bat_dis_mw"].values > eps)))
        soc_viol = float(np.sum((day_hourly["battery_soc"].values < SOC_MIN - 1e-6) | (day_hourly["battery_soc"].values > SOC_MAX + 1e-6)))
        unmet_any = float(np.sum((day_hourly["unmet_h2_kgph"].values > eps) | (day_hourly["unmet_nh3_kgph"].values > eps) | (day_hourly["unmet_meoh_kgph"].values > eps)))


    # ---------------- End of run aggregation ----------------
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        raise RuntimeError("No days were solved (daily_df is empty).")

    solver_df = pd.DataFrame(solver_rows)
    state_df  = pd.DataFrame(state_rows)

    if len(all_hourly_frames) > 0:
        hourly_all_df = pd.concat(all_hourly_frames, ignore_index=True)
        hourly_all_df.to_csv(OUT_POST / "hourly_timeseries_all_days.csv", index=False)
    else:
        hourly_all_df = pd.DataFrame()

    daily_df.to_csv(OUT_POST / "daily_summary.csv", index=False)
    solver_df.to_csv(OUT_POST / "solver_log_daily.csv", index=False)
    state_df.to_csv(OUT_POST / "state_trace.csv", index=False)

    annual_summary = {
        "run_id": RUN_ID,
        "model_tag": MODEL_TAG,
        "scenario_tag": SCENARIO_TAG,
        "demand_tag": DEMAND_TAG,
        "start_day_of_year": START_DAY_OF_YEAR,
        "n_days_run": int(len(daily_df)),

        "total_cost_eur": float(daily_df["total_cost_eur"].sum()),

        "total_grid_emissions_aef_kg": float(daily_df["emissions_grid_aef_kg"].sum()),
        "total_grid_emissions_mef_upramp_kg": float(daily_df["emissions_grid_mef_upramp_kg"].sum()),
        "total_grid_emissions_mef_soft_kg": float(daily_df["emissions_grid_mef_soft_kg"].sum()),

        "total_direct_emissions_nh3_kg": float(daily_df["emissions_direct_nh3_kg"].sum()),
        "total_direct_emissions_meoh_kg": float(daily_df["emissions_direct_meoh_kg"].sum()),
        "total_direct_emissions_kg": float(daily_df["emissions_direct_total_kg"].sum()),

        "total_grid_import_kWh": float(daily_df["grid_import_kWh"].sum()),
        "total_res_used_kWh": float(daily_df["res_used_kWh"].sum()),
        "total_res_curtailed_kWh": float(daily_df["res_curtailed_kWh"].sum()),
        "avg_grid_share_percent": float(daily_df["grid_share_percent"].mean()),

        "total_unmet_H2_kg": float(daily_df["unmet_H2_kg"].sum()),
        "total_unmet_NH3_kg": float(daily_df["unmet_NH3_kg"].sum()),
        "total_unmet_MeOH_kg": float(daily_df["unmet_MeOH_kg"].sum()),

        "total_H2_sell_g_kg": float(daily_df["H2_sell_g_kg_day"].sum()),
        "total_H2_sell_ng_kg": float(daily_df["H2_sell_ng_kg_day"].sum()),
        "total_H2_sell_kg": float(daily_df["H2_sell_total_kg_day"].sum()),

        "total_NH3_sell_g_kg": float(daily_df["NH3_sell_g_kg_day"].sum()),
        "total_NH3_sell_ng_kg": float(daily_df["NH3_sell_ng_kg_day"].sum()),
        "total_NH3_sell_kg": float(daily_df["NH3_sell_total_kg_day"].sum()),

        "total_MeOH_sell_g_kg": float(daily_df["MeOH_sell_g_kg_day"].sum()),
        "total_MeOH_sell_ng_kg": float(daily_df["MeOH_sell_ng_kg_day"].sum()),
        "total_MeOH_sell_kg": float(daily_df["MeOH_sell_total_kg_day"].sum()),
    }
    pd.DataFrame([annual_summary]).to_csv(OUT_POST / "annual_summary.csv", index=False)


    return OUT_RUN


# M1 formulation
def _run_m1(config):
    # ===================== M1 ====================

    # min unmet >> min emissions, for lower bound of emission for M2

    # ---------------- Header / metadata ----------------
    MODEL_TAG = "M1"
    FORMULATION_TAG = "M1_final_v2"
    SCENARIO_TAG = "S0"
    DEMAND_TAG = "D1"
    GREEN_TAG = "S0_green_split_with_rfnbo_hourly"

    # Choose which EF to use for the emissions objective:
    # "aef" or "mef_upramp" or "mef_soft"
    EMIS_EF_MODE = config["signal"]

    # ---------------- Imports ----------------
    import gurobipy as gp
    from gurobipy import GRB

    # ---------------- Global configuration ----------------
    DATA_DIR = Path(config["model_ready"]).resolve().parent
    MR = Path(config["model_ready"]).resolve()

    # Required model-ready files (from utility cell)
    RES_CSV          = MR / "res_hourly_2024.csv"
    DEM_CSV          = MR / "demand_hourly_2024.csv"
    PRICE_CSV        = MR / "grid_price_hourly_2024.csv"
    AEF_CSV          = MR / "grid_aef_hourly_2024.csv"
    MEF_UPRAMP_CSV   = MR / "grid_mef_upramp_hourly_2024.csv"
    MEF_SOFT_CSV     = MR / "grid_mef_soft_hourly_2024.csv"
    PARAMS_JSON      = MR / "params_scenario_all.json"

    AEF_COL          = "grid_aef_predicted_kg_per_kwh"
    MEF_UPRAMP_COL   = "grid_mef_upramp_pred_kg_per_kwh"
    MEF_SOFT_COL     = "grid_mef_soft_kg_per_kwh"

    # ---------------- Run controls ----------------
    N_DAYS_TO_RUN = int(config.get("days", 1))
    START_DAY_OF_YEAR = 1  # Resume requires explicit carryover, not supported by this entry point.
    MIPGAP            = 0.01
    TIME_LIMIT_S = float(config.get("time_limit", 350))
    OUTPUTFLAG = int(config.get("solver_output", 1))

    # ---------------- License handling ----------------

    # ---------------- Run ID builder ----------------
    def build_run_id():
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        return f"{MODEL_TAG}__{SCENARIO_TAG}__{DEMAND_TAG}__start{START_DAY_OF_YEAR:03d}__ndays{N_DAYS_TO_RUN:03d}__{stamp}"

    RUN_ID = build_run_id()

    # ---------------- Paths for outputs ----------------
    OUT_BASE = Path(config["output"]).resolve()
    OUT_RUN   = OUT_BASE / RUN_ID
    OUT_POST  = OUT_RUN / "post"
    OUT_RUN.mkdir(parents=True, exist_ok=False)
    OUT_POST.mkdir(parents=True, exist_ok=True)


    # ---------------- Input loading ----------------

    res        = read_hourly_csv(RES_CSV)[["pv_mw","wind_mw","res_total_mw"]]
    dem        = read_hourly_csv(DEM_CSV)[["h2_demand_kgph","nh3_demand_kgph","meoh_demand_kgph"]]
    price      = read_hourly_csv(PRICE_CSV)[["da_elec_price_EURperMWh"]]
    aef        = read_hourly_csv(AEF_CSV)[[AEF_COL]]
    mef_upramp = read_hourly_csv(MEF_UPRAMP_CSV)[[MEF_UPRAMP_COL]]
    mef_soft   = read_hourly_csv(MEF_SOFT_CSV)[[MEF_SOFT_COL]]

    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    # ---------------- Input validation & alignment ----------------
    idx_2024 = pd.date_range("2024-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="h", tz="UTC")


    res        = assert_full_hourly(res, "res")
    dem        = assert_full_hourly(dem, "dem")
    price      = assert_full_hourly(price, "price")
    aef        = assert_full_hourly(aef, "aef")
    mef_upramp = assert_full_hourly(mef_upramp, "mef_upramp")
    mef_soft   = assert_full_hourly(mef_soft, "mef_soft")

    # ---------------- Scalar parameters (aligned with M0-V2) ----------------
    P_GRID_MAX_MW = 295.58

    # Battery
    P_BAT_MAX_MW  = 50.0
    E_BAT_MAX_MWH = 100.32
    ETA_CH        = 0.964
    ETA_DIS       = 0.964
    SOC_MIN       = 0.15
    SOC_MAX       = 0.95
    SOC_INIT      = 0.5

    # Electrolyzer
    P_ELZ_MIN = 8
    P_ELZ_MAX = 52.25
    H_ELZ_MAX = 914.39161
    A_elz = -0.075045052
    B_elz = 21.73040904
    C_elz = -16.14458074
    N_stack = 4
    omega_H2O = 0.015
    c_H2O = 2.24
    C_su_elz = 1392

    P_ELZ_MAX_TOT = N_stack * P_ELZ_MAX
    H_ELZ_MAX_TOT = N_stack * H_ELZ_MAX

    H_min = A_elz*(P_ELZ_MIN**2) + B_elz*P_ELZ_MIN + C_elz
    H_max = A_elz*(P_ELZ_MAX**2) + B_elz*P_ELZ_MAX + C_elz
    Q1 = (H_max - H_min) / (P_ELZ_MAX - P_ELZ_MIN)
    Q0 = H_min - Q1 * P_ELZ_MIN

    # NH3
    Q_NH3_CAP = 13472.04
    f_NH3_min = 0.61
    RR_NH3 = 0.2
    mH2_to_NH3 = 0.18
    mN2_to_NH3 = 0.87
    SEC_HB = 0.36
    SEC_ASU = 0.068
    C_su_NH3 = 2397.46
    de_NH3 = 0.00519

    # MeOH
    Q_MEOH_CAP = 5818.856
    f_MEOH_min = 0.1
    R_MEOH = 0.5 * Q_MEOH_CAP
    mH2_to_MEOH = 0.22
    SEC_MEOH = 161.75/1000.0
    mCO2_to_MEOH = 1.59
    c_CO2 = 0.2
    C_su_MEOH = 1159.37
    de_MEOH = 0.08

    # Cracker
    Q_CRK_CAP = 6062.416
    f_CRK_min = 0.05
    R_CRK = 0.5 * Q_CRK_CAP
    kH2_out = 0.12024
    kNH3_rec = 0.11848
    kH2_fuel = 0.035294
    SEC_CRK_E = 0.524667
    kSteam = 0.941176
    eSteam = 0.75
    omega_steam = 0.001
    C_su_CRK = 1000.0

    # Tanks
    H_TANK = 12.0
    ALPHA_TANK_FILL = 0.15

    S_H2_CAP   = H_TANK * H_ELZ_MAX_TOT
    S_NH3_CAP  = H_TANK * Q_NH3_CAP
    S_MEOH_CAP = H_TANK * Q_MEOH_CAP

    S_H2_INIT   = ALPHA_TANK_FILL * S_H2_CAP
    S_NH3_INIT  = ALPHA_TANK_FILL * S_NH3_CAP
    S_MEOH_INIT = ALPHA_TANK_FILL * S_MEOH_CAP

    # ---------------- Helpers ----------------

    # ---------------- Carry-over initialization ----------------
    carry = {
        "soc": SOC_INIT,
        "S_H2_g": 0.5*S_H2_INIT,
        "S_H2_ng": 0.5*S_H2_INIT,
        "S_NH3_g": 0.5*S_NH3_INIT,
        "S_NH3_ng": 0.5*S_NH3_INIT,
        "S_MEOH_g": 0.5*S_MEOH_INIT,
        "S_MEOH_ng": 0.5*S_MEOH_INIT,
        "y_elz_prev_mod": [0]*N_stack,
        "y_nh3_prev": 0,
        "y_meoh_prev": 0,
        "y_crk_prev": 0,
        "Q_nh3_prev": 0.0,
        "Q_meoh_prev": 0.0,
        "Q_crk_prev": 0.0,
    }

    daily_rows = []
    solver_rows = []
    state_rows = []
    all_hourly_frames = []

    # ---------------- Run loop ----------------
    t0_global = time.time()

    for d in range(START_DAY_OF_YEAR, START_DAY_OF_YEAR + N_DAYS_TO_RUN):
        day_hours = day_to_time_slice(d)

        if day_hours[0] < idx_2024[0] or day_hours[-1] > idx_2024[-1]:
            break


        pv_av    = res.loc[day_hours, "pv_mw"].values.astype(float)
        wd_av    = res.loc[day_hours, "wind_mw"].values.astype(float)
        pr_eur   = price.loc[day_hours, "da_elec_price_EURperMWh"].values.astype(float)
        dem_h2   = dem.loc[day_hours, "h2_demand_kgph"].values.astype(float)
        dem_nh3  = dem.loc[day_hours, "nh3_demand_kgph"].values.astype(float)
        dem_meoh = dem.loc[day_hours, "meoh_demand_kgph"].values.astype(float)

        aef_k        = aef.loc[day_hours, AEF_COL].values.astype(float)
        mef_upramp_k = mef_upramp.loc[day_hours, MEF_UPRAMP_COL].values.astype(float)
        mef_soft_k   = mef_soft.loc[day_hours, MEF_SOFT_COL].values.astype(float)

        if EMIS_EF_MODE.lower() == "aef":
            ef_obj_k = aef_k
        elif EMIS_EF_MODE.lower() == "mef_upramp":
            ef_obj_k = mef_upramp_k
        elif EMIS_EF_MODE.lower() == "mef_soft":
            ef_obj_k = mef_soft_k
        else:
            raise ValueError("EMIS_EF_MODE must be one of: 'aef', 'mef_upramp', 'mef_soft'")

        # ---------------- Model ----------------
        m = gp.Model(f"{MODEL_TAG}_day{d:03d}")
        m.Params.MIPGap = MIPGAP
        m.Params.TimeLimit = TIME_LIMIT_S
        m.Params.OutputFlag = OUTPUTFLAG
        m.Params.NumericFocus = 1
        m.Params.MultiObjMethod = 2
        m.Params.MultiObjPre = 1

        T = range(24)
        T2 = range(25)

        S_sources = ["pv", "wd", "grid", "bat"]
        K_sinks = ["elz", "NH3", "MeOH", "crk", "bch"]
        S_res = ["pv", "wd"]

        # ---------------- Variables ----------------
        P = m.addVars(S_sources, K_sinks, T, lb=0.0, name="P_s2k")
        P_pv_cur = m.addVars(T, lb=0.0, name="P_pv_cur")
        P_wd_cur = m.addVars(T, lb=0.0, name="P_wd_cur")
        P_grid = m.addVars(T, lb=0.0, ub=P_GRID_MAX_MW, name="P_grid")

        P_ch  = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_ch")
        P_dis = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_dis")
        SOC   = m.addVars(T2, lb=SOC_MIN, ub=SOC_MAX, name="SOC")

        Pk_g  = m.addVars(K_sinks, T, lb=0.0, name="P_sink_g")
        Pk_ng = m.addVars(K_sinks, T, lb=0.0, name="P_sink_ng")

        # Electrolyzer
        I_elz = range(N_stack)
        y_elz_m  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="y_elz_m")
        su_elz_m = m.addVars(I_elz, T, vtype=GRB.BINARY, name="su_elz_m")
        b_elz_g  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="b_elz_g")

        P_elz_m_g   = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_g")
        P_elz_m_ng  = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_ng")
        P_elz_m_tot = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_tot")

        H_elz_m    = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m")
        H_elz_m_g  = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_g")
        H_elz_m_ng = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_ng")

        y_elz    = m.addVars(T, vtype=GRB.BINARY, name="y_elz")
        n_elz_on = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="n_elz_on")
        su_elz   = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="su_elz")

        P_elz_g   = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_g")
        P_elz_ng  = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_ng")
        P_elz_tot = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_tot")
        H_prod    = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod")
        H_prod_g  = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_g")
        H_prod_ng = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_ng")
        W_elz     = m.addVars(T, lb=0.0, name="W_elz_m3ph")

        # NH3
        y_nh3    = m.addVars(T, vtype=GRB.BINARY, name="y_nh3")
        su_nh3   = m.addVars(T, vtype=GRB.BINARY, name="su_nh3")
        Q_nh3_g  = m.addVars(T, lb=0.0, name="Q_nh3_g")
        Q_nh3_ng = m.addVars(T, lb=0.0, name="Q_nh3_ng")
        Q_nh3    = m.addVars(T, lb=0.0, name="Q_nh3")
        P_nh3_g  = m.addVars(T, lb=0.0, name="P_nh3_g")
        P_nh3_ng = m.addVars(T, lb=0.0, name="P_nh3_ng")

        # MeOH
        y_meoh    = m.addVars(T, vtype=GRB.BINARY, name="y_meoh")
        su_meoh   = m.addVars(T, vtype=GRB.BINARY, name="su_meoh")
        Q_meoh_g  = m.addVars(T, lb=0.0, name="Q_meoh_g")
        Q_meoh_ng = m.addVars(T, lb=0.0, name="Q_meoh_ng")
        Q_meoh    = m.addVars(T, lb=0.0, name="Q_meoh")
        P_meoh_g  = m.addVars(T, lb=0.0, name="P_meoh_g")
        P_meoh_ng = m.addVars(T, lb=0.0, name="P_meoh_ng")
        CO2_buy   = m.addVars(T, lb=0.0, name="CO2_buy_kgph")

        # Cracker
        y_crk      = m.addVars(T, vtype=GRB.BINARY, name="y_crk")
        su_crk     = m.addVars(T, vtype=GRB.BINARY, name="su_crk")
        Q_crk_in_g = m.addVars(T, lb=0.0, name="Q_crk_in_g")
        Q_crk_in_ng= m.addVars(T, lb=0.0, name="Q_crk_in_ng")
        Q_crk_in   = m.addVars(T, lb=0.0, name="Q_crk_in")

        H_crk_out_g   = m.addVars(T, lb=0.0, name="H_crk_out_g")
        H_crk_out_ng  = m.addVars(T, lb=0.0, name="H_crk_out_ng")
        NH3_rec_g     = m.addVars(T, lb=0.0, name="NH3_rec_g")
        NH3_rec_ng    = m.addVars(T, lb=0.0, name="NH3_rec_ng")
        H_crk_fuel_g  = m.addVars(T, lb=0.0, name="H_crk_fuel_g")
        H_crk_fuel_ng = m.addVars(T, lb=0.0, name="H_crk_fuel_ng")
        Steam_g       = m.addVars(T, lb=0.0, name="Steam_g_kgph")
        Steam_ng      = m.addVars(T, lb=0.0, name="Steam_ng_kgph")
        P_crk_g       = m.addVars(T, lb=0.0, name="P_crk_g")
        P_crk_ng      = m.addVars(T, lb=0.0, name="P_crk_ng")
        W_steam       = m.addVars(T, lb=0.0, name="W_steam_m3ph")

        # Tanks
        S_H2_g    = m.addVars(T2, lb=0.0, name="S_H2_g")
        S_H2_ng   = m.addVars(T2, lb=0.0, name="S_H2_ng")
        S_NH3_g   = m.addVars(T2, lb=0.0, name="S_NH3_g")
        S_NH3_ng  = m.addVars(T2, lb=0.0, name="S_NH3_ng")
        S_MEOH_g  = m.addVars(T2, lb=0.0, name="S_MEOH_g")
        S_MEOH_ng = m.addVars(T2, lb=0.0, name="S_MEOH_ng")

        # Sales
        H_sell_g     = m.addVars(T, lb=0.0, name="H_sell_g")
        H_sell_ng    = m.addVars(T, lb=0.0, name="H_sell_ng")
        NH3_sell_g   = m.addVars(T, lb=0.0, name="NH3_sell_g")
        NH3_sell_ng  = m.addVars(T, lb=0.0, name="NH3_sell_ng")
        MEOH_sell_g  = m.addVars(T, lb=0.0, name="MEOH_sell_g")
        MEOH_sell_ng = m.addVars(T, lb=0.0, name="MEOH_sell_ng")

        # H2 to units
        H_to_NH3_g   = m.addVars(T, lb=0.0, name="H_to_NH3_g")
        H_to_NH3_ng  = m.addVars(T, lb=0.0, name="H_to_NH3_ng")
        H_to_MEOH_g  = m.addVars(T, lb=0.0, name="H_to_MEOH_g")
        H_to_MEOH_ng = m.addVars(T, lb=0.0, name="H_to_MEOH_ng")

        # Unmet
        u_H2   = m.addVars(T, lb=0.0, name="u_H2")
        u_NH3  = m.addVars(T, lb=0.0, name="u_NH3")
        u_MEOH = m.addVars(T, lb=0.0, name="u_MEOH")

        # ---------------- Tight bounds ----------------
        for t in T:
            m.addConstr(P_pv_cur[t] <= pv_av[t], name=f"ub_pv_cur[{t}]")
            m.addConstr(P_wd_cur[t] <= wd_av[t], name=f"ub_wd_cur[{t}]")

        for t in T:
            for k in K_sinks:
                m.addConstr(P["pv",k,t]   <= pv_av[t],      name=f"ub_P_pv_{k}[{t}]")
                m.addConstr(P["wd",k,t]   <= wd_av[t],      name=f"ub_P_wd_{k}[{t}]")
                m.addConstr(P["grid",k,t] <= P_GRID_MAX_MW, name=f"ub_P_grid_{k}[{t}]")
                m.addConstr(P["bat",k,t]  <= P_BAT_MAX_MW,  name=f"ub_P_bat_{k}[{t}]")

        # ---------------- General physics ----------------
        for t in T:
            m.addConstr(gp.quicksum(P["pv",k,t] for k in K_sinks) + P_pv_cur[t] == pv_av[t], name=f"pv_balance[{t}]")
            m.addConstr(gp.quicksum(P["wd",k,t] for k in K_sinks) + P_wd_cur[t] == wd_av[t], name=f"wd_balance[{t}]")

            m.addConstr(gp.quicksum(P["grid",k,t] for k in K_sinks) == P_grid[t], name=f"grid_alloc[{t}]")
            m.addConstr(gp.quicksum(P["bat",k,t] for k in K_sinks)  == P_dis[t],  name=f"bat_alloc[{t}]")

            m.addConstr(gp.quicksum(P[s,"bch",t] for s in S_sources) == P_ch[t], name=f"bch_sink[{t}]")

            m.addConstr(P["grid","bch",t] == 0.0, name=f"no_grid_to_bch[{t}]")
            m.addConstr(P["bat","bch",t]  == 0.0, name=f"no_bat_to_bch[{t}]")
            m.addConstr(P["pv","bch",t] + P["wd","bch",t] == P_ch[t], name=f"bch_from_res[{t}]")

            m.addSOS(GRB.SOS_TYPE1, [P_ch[t], P_dis[t]])
            m.addConstr(P_dis[t] <= ETA_DIS * (SOC[t] - SOC_MIN) * E_BAT_MAX_MWH, name=f"ub_Pdis_soc[{t}]")
            m.addConstr(P_ch[t]  <= (SOC_MAX - SOC[t]) * E_BAT_MAX_MWH / ETA_CH,  name=f"ub_Pch_soc[{t}]")

            for k in K_sinks:
                m.addConstr(Pk_ng[k,t] == P["grid",k,t] + P["bat",k,t], name=f"Pk_ng_def[{k},{t}]")
                m.addConstr(Pk_g[k,t]  == gp.quicksum(P[s,k,t] for s in S_res), name=f"Pk_g_def[{k},{t}]")

        m.addConstr(SOC[0] == float(carry["soc"]), name="SOC_init_carry")
        for t in T:
            m.addConstr(
                SOC[t+1] == SOC[t] + (ETA_CH*P_ch[t] - (1.0/ETA_DIS)*P_dis[t]) / E_BAT_MAX_MWH,
                name=f"SOC_dyn[{t}]"
            )

        m.addConstrs((y_elz_m[i,t] >= y_elz_m[i+1,t] for i in range(N_stack-1) for t in T), name="elz_module_order")

        # ---------------- Electrolyzer ----------------
        for t in T:
            m.addConstr(P_elz_g[t]  == Pk_g["elz",t],  name=f"P_elz_g_link[{t}]")
            m.addConstr(P_elz_ng[t] == Pk_ng["elz",t], name=f"P_elz_ng_link[{t}]")

            m.addConstr(P_elz_g[t]  == gp.quicksum(P_elz_m_g[i,t] for i in I_elz), name=f"P_elz_g_sum_mod[{t}]")
            m.addConstr(P_elz_ng[t] == gp.quicksum(P_elz_m_ng[i,t] for i in I_elz), name=f"P_elz_ng_sum_mod[{t}]")
            m.addConstr(P_elz_tot[t] == P_elz_g[t] + P_elz_ng[t], name=f"P_elz_tot_sum[{t}]")

            m.addConstr(n_elz_on[t] == gp.quicksum(y_elz_m[i,t] for i in I_elz), name=f"n_elz_on_def[{t}]")
            m.addConstr(n_elz_on[t] >= y_elz[t], name=f"y_elz_lb[{t}]")
            m.addConstr(n_elz_on[t] <= N_stack * y_elz[t], name=f"y_elz_ub[{t}]")
            m.addConstr(su_elz[t] == gp.quicksum(su_elz_m[i,t] for i in I_elz), name=f"su_elz_sum[{t}]")

            m.addConstr(H_prod[t]    == gp.quicksum(H_elz_m[i,t] for i in I_elz), name=f"H_prod_sum_mod[{t}]")
            m.addConstr(H_prod_g[t]  == gp.quicksum(H_elz_m_g[i,t] for i in I_elz), name=f"H_prod_g_sum_mod[{t}]")
            m.addConstr(H_prod_ng[t] == gp.quicksum(H_elz_m_ng[i,t] for i in I_elz), name=f"H_prod_ng_sum_mod[{t}]")
            m.addConstr(H_prod[t] == H_prod_g[t] + H_prod_ng[t], name=f"H_prod_split[{t}]")
            m.addConstr(W_elz[t] == omega_H2O * H_prod[t], name=f"W_elz[{t}]")

            for i in range(N_stack - 1):
                m.addConstr(b_elz_g[i,t] >= b_elz_g[i+1,t], name=f"elz_bg_order[{i},{t}]")

            for i in I_elz:
                m.addConstr(P_elz_m_tot[i,t] == P_elz_m_g[i,t] + P_elz_m_ng[i,t], name=f"P_elz_m_tot_def[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] <= P_ELZ_MAX * y_elz_m[i,t], name=f"elz_m_max[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] >= P_ELZ_MIN * y_elz_m[i,t], name=f"elz_m_min[{i},{t}]")
                m.addConstr(b_elz_g[i,t] <= y_elz_m[i,t], name=f"b_elz_g_gate[{i},{t}]")

                m.addConstr(P_elz_m_g[i,t]  <= P_ELZ_MAX * b_elz_g[i,t], name=f"P_elz_m_g_cap[{i},{t}]")
                m.addConstr(P_elz_m_ng[i,t] <= P_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"P_elz_m_ng_cap[{i},{t}]")

                m.addConstr(H_elz_m[i,t] == H_elz_m_g[i,t] + H_elz_m_ng[i,t], name=f"H_elz_m_split[{i},{t}]")

                m.addQConstr(
                    H_elz_m[i,t] <= A_elz*(P_elz_m_tot[i,t]*P_elz_m_tot[i,t]) + B_elz*P_elz_m_tot[i,t] + C_elz*y_elz_m[i,t],
                    name=f"elz_m_quad[{i},{t}]"
                )
                m.addConstr(H_elz_m[i,t] >= Q1*P_elz_m_tot[i,t] + Q0*y_elz_m[i,t], name=f"elz_m_lin[{i},{t}]")

                m.addConstr(H_elz_m_g[i,t] <= H_ELZ_MAX * b_elz_g[i,t], name=f"H_elz_m_g_ub1[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] <= H_elz_m[i,t], name=f"H_elz_m_g_ub2[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*(1 - b_elz_g[i,t]), name=f"H_elz_m_g_lb[{i},{t}]")

                m.addConstr(H_elz_m_ng[i,t] <= H_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"H_elz_m_ng_ub1[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] <= H_elz_m[i,t], name=f"H_elz_m_ng_ub2[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*b_elz_g[i,t], name=f"H_elz_m_ng_lb[{i},{t}]")

                if t >= 1:
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_elz_m[i,t-1], name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_elz_m[i,t-1], name=f"su_elz_m3[{i},{t}]")
                else:
                    y_prev_i = int(carry["y_elz_prev_mod"][i])
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_prev_i, name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_prev_i, name=f"su_elz_m3[{i},{t}]")

        # ---------------- NH3 ----------------
        for t in T:
            m.addConstr(Q_nh3[t] == Q_nh3_g[t] + Q_nh3_ng[t], name=f"Q_nh3_sum[{t}]")
            m.addConstr(Q_nh3[t] <= Q_NH3_CAP * y_nh3[t], name=f"nh3_max[{t}]")
            m.addConstr(Q_nh3[t] >= f_NH3_min * Q_NH3_CAP * y_nh3[t], name=f"nh3_min[{t}]")

            if t >= 1:
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_nh3[t-1])
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_nh3[t-1])
            else:
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_prev)
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_prev)

            m.addConstr(
                Q_nh3[t] <= f_NH3_min * Q_NH3_CAP + (Q_NH3_CAP - f_NH3_min * Q_NH3_CAP) * (1 - su_nh3[t]),
                name=f"nh3_start_minload[{t}]"
            )

            ramp_nh3 = RR_NH3 * Q_NH3_CAP

            if t == 0:
                Q_prev = float(carry["Q_nh3_prev"])
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(Q_nh3[t] - Q_prev <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rup0[{t}]")
                m.addConstr(Q_prev - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rdn0[{t}]")
            else:
                m.addConstr(Q_nh3[t] - Q_nh3[t-1] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rup[{t}]")
                m.addConstr(Q_nh3[t-1] - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rdn[{t}]")

            m.addConstr(P_nh3_g[t]  == (SEC_HB*Q_nh3_g[t]  + SEC_ASU*(mN2_to_NH3*Q_nh3_g[t]))  / 1000.0, name=f"P_nh3_g_map[{t}]")
            m.addConstr(P_nh3_ng[t] == (SEC_HB*Q_nh3_ng[t] + SEC_ASU*(mN2_to_NH3*Q_nh3_ng[t])) / 1000.0, name=f"P_nh3_ng_map[{t}]")

            m.addConstr(P_nh3_g[t]  == Pk_g["NH3",t],  name=f"P_nh3_g_link[{t}]")
            m.addConstr(P_nh3_ng[t] == Pk_ng["NH3",t], name=f"P_nh3_ng_link[{t}]")

            m.addConstr(H_to_NH3_g[t]  == mH2_to_NH3 * Q_nh3_g[t],  name=f"H_to_NH3_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] == mH2_to_NH3 * Q_nh3_ng[t], name=f"H_to_NH3_ng[{t}]")

        # ---------------- MeOH ----------------
        for t in T:
            m.addConstr(Q_meoh[t] == Q_meoh_g[t] + Q_meoh_ng[t], name=f"Q_meoh_sum[{t}]")
            m.addConstr(Q_meoh[t] <= Q_MEOH_CAP * y_meoh[t], name=f"meoh_max[{t}]")
            m.addConstr(Q_meoh[t] >= f_MEOH_min * Q_MEOH_CAP * y_meoh[t], name=f"meoh_min[{t}]")

            if t >= 1:
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_meoh[t-1])
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_meoh[t-1])
            else:
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_prev)
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_prev)

            m.addConstr(
                Q_meoh[t] <= f_MEOH_min * Q_MEOH_CAP + (Q_MEOH_CAP - f_MEOH_min * Q_MEOH_CAP) * (1 - su_meoh[t]),
                name=f"meoh_start_minload[{t}]"
            )

            ramp_meoh = R_MEOH

            if t == 0:
                Q_prev = float(carry["Q_meoh_prev"])
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(Q_meoh[t] - Q_prev <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rup0[{t}]")
                m.addConstr(Q_prev - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rdn0[{t}]")
            else:
                m.addConstr(Q_meoh[t] - Q_meoh[t-1] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rup[{t}]")
                m.addConstr(Q_meoh[t-1] - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rdn[{t}]")

            m.addConstr(P_meoh_g[t]  == (SEC_MEOH * Q_meoh_g[t])  / 1000.0, name=f"P_meoh_g_map[{t}]")
            m.addConstr(P_meoh_ng[t] == (SEC_MEOH * Q_meoh_ng[t]) / 1000.0, name=f"P_meoh_ng_map[{t}]")

            m.addConstr(P_meoh_g[t]  == Pk_g["MeOH",t],  name=f"P_meoh_g_link[{t}]")
            m.addConstr(P_meoh_ng[t] == Pk_ng["MeOH",t], name=f"P_meoh_ng_link[{t}]")

            m.addConstr(CO2_buy[t] == mCO2_to_MEOH * Q_meoh[t], name=f"CO2_buy[{t}]")

            m.addConstr(H_to_MEOH_g[t]  == mH2_to_MEOH * Q_meoh_g[t],  name=f"H_to_MEOH_g[{t}]")
            m.addConstr(H_to_MEOH_ng[t] == mH2_to_MEOH * Q_meoh_ng[t], name=f"H_to_MEOH_ng[{t}]")

        # ---------------- Cracker ----------------
        for t in T:
            m.addConstr(Q_crk_in[t] == Q_crk_in_g[t] + Q_crk_in_ng[t], name=f"Q_crk_sum[{t}]")
            m.addConstr(Q_crk_in[t] <= Q_CRK_CAP * y_crk[t], name=f"crk_max[{t}]")
            m.addConstr(Q_crk_in[t] >= f_CRK_min * Q_CRK_CAP * y_crk[t], name=f"crk_min[{t}]")

            if t >= 1:
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_crk[t-1])
                m.addConstr(su_crk[t] >= y_crk[t] - y_crk[t-1])
            else:
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_prev)
                m.addConstr(su_crk[t] >= y_crk[t] - y_prev)

            m.addConstr(
                Q_crk_in[t] <= f_CRK_min * Q_CRK_CAP + (Q_CRK_CAP - f_CRK_min * Q_CRK_CAP) * (1 - su_crk[t]),
                name=f"crk_start_minload[{t}]"
            )

            ramp_crk = R_CRK

            if t == 0:
                Q_prev = float(carry["Q_crk_prev"])
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(Q_crk_in[t] - Q_prev <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rup0[{t}]")
                m.addConstr(Q_prev - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rdn0[{t}]")
            else:
                m.addConstr(Q_crk_in[t] - Q_crk_in[t-1] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rup[{t}]")
                m.addConstr(Q_crk_in[t-1] - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rdn[{t}]")

            m.addConstr(H_crk_out_g[t]  == kH2_out * Q_crk_in_g[t],  name=f"Hcrk_out_g[{t}]")
            m.addConstr(H_crk_out_ng[t] == kH2_out * Q_crk_in_ng[t], name=f"Hcrk_out_ng[{t}]")
            m.addConstr(NH3_rec_g[t]    == kNH3_rec * Q_crk_in_g[t],  name=f"NH3rec_g[{t}]")
            m.addConstr(NH3_rec_ng[t]   == kNH3_rec * Q_crk_in_ng[t], name=f"NH3rec_ng[{t}]")
            m.addConstr(H_crk_fuel_g[t]  == kH2_fuel * Q_crk_in_g[t],  name=f"Hcrk_fuel_g[{t}]")
            m.addConstr(H_crk_fuel_ng[t] == kH2_fuel * Q_crk_in_ng[t], name=f"Hcrk_fuel_ng[{t}]")

            m.addConstr(Steam_g[t]  == kSteam * Q_crk_in_g[t],  name=f"Steam_g[{t}]")
            m.addConstr(Steam_ng[t] == kSteam * Q_crk_in_ng[t], name=f"Steam_ng[{t}]")
            m.addConstr(W_steam[t]  == omega_steam * (Steam_g[t] + Steam_ng[t]), name=f"W_steam[{t}]")

            m.addConstr(P_crk_g[t]  == (SEC_CRK_E*Q_crk_in_g[t]  + eSteam*Steam_g[t])  / 1000.0, name=f"P_crk_g_map[{t}]")
            m.addConstr(P_crk_ng[t] == (SEC_CRK_E*Q_crk_in_ng[t] + eSteam*Steam_ng[t]) / 1000.0, name=f"P_crk_ng_map[{t}]")

            m.addConstr(P_crk_g[t]  == Pk_g["crk",t],  name=f"P_crk_g_link[{t}]")
            m.addConstr(P_crk_ng[t] == Pk_ng["crk",t], name=f"P_crk_ng_link[{t}]")

        # ---------------- Tanks / demand ----------------
        m.addConstr(S_H2_g[0]    == float(carry["S_H2_g"]),    name="S_H2_g_init")
        m.addConstr(S_H2_ng[0]   == float(carry["S_H2_ng"]),   name="S_H2_ng_init")
        m.addConstr(S_NH3_g[0]   == float(carry["S_NH3_g"]),   name="S_NH3_g_init")
        m.addConstr(S_NH3_ng[0]  == float(carry["S_NH3_ng"]),  name="S_NH3_ng_init")
        m.addConstr(S_MEOH_g[0]  == float(carry["S_MEOH_g"]),  name="S_MEOH_g_init")
        m.addConstr(S_MEOH_ng[0] == float(carry["S_MEOH_ng"]), name="S_MEOH_ng_init")

        for tt in T2:
            m.addConstr(S_H2_g[tt] + S_H2_ng[tt] <= S_H2_CAP,   name=f"S_H2_cap[{tt}]")
            m.addConstr(S_NH3_g[tt] + S_NH3_ng[tt] <= S_NH3_CAP, name=f"S_NH3_cap[{tt}]")
            m.addConstr(S_MEOH_g[tt] + S_MEOH_ng[tt] <= S_MEOH_CAP, name=f"S_MEOH_cap[{tt}]")

        m.addConstr(S_H2_g[24]   + S_H2_ng[24]   >= ALPHA_TANK_FILL * S_H2_CAP,   name="eod_hard_H2_frac")
        m.addConstr(S_NH3_g[24]  + S_NH3_ng[24]  >= ALPHA_TANK_FILL * S_NH3_CAP,  name="eod_hard_NH3_frac")
        m.addConstr(S_MEOH_g[24] + S_MEOH_ng[24] >= ALPHA_TANK_FILL * S_MEOH_CAP, name="eod_hard_MeOH_frac")

        for t in T:
            m.addConstr(H_sell_g[t] + H_sell_ng[t] + u_H2[t] == dem_h2[t], name=f"dem_H2[{t}]")
            m.addConstr(NH3_sell_g[t] + NH3_sell_ng[t] + u_NH3[t] == dem_nh3[t], name=f"dem_NH3[{t}]")
            m.addConstr(MEOH_sell_g[t] + MEOH_sell_ng[t] + u_MEOH[t] == dem_meoh[t], name=f"dem_MEOH[{t}]")

            m.addConstr(
                S_H2_g[t+1] == S_H2_g[t] + H_prod_g[t] + H_crk_out_g[t] - H_to_NH3_g[t] - H_to_MEOH_g[t] - H_sell_g[t] - H_crk_fuel_g[t],
                name=f"S_H2_g_dyn[{t}]"
            )
            m.addConstr(
                S_H2_ng[t+1] == S_H2_ng[t] + H_prod_ng[t] + H_crk_out_ng[t] - H_to_NH3_ng[t] - H_to_MEOH_ng[t] - H_sell_ng[t] - H_crk_fuel_ng[t],
                name=f"S_H2_ng_dyn[{t}]"
            )

            m.addConstr(
                S_NH3_g[t+1] == S_NH3_g[t] + Q_nh3_g[t] + NH3_rec_g[t] - NH3_sell_g[t] - Q_crk_in_g[t],
                name=f"S_NH3_g_dyn[{t}]"
            )
            m.addConstr(
                S_NH3_ng[t+1] == S_NH3_ng[t] + Q_nh3_ng[t] + NH3_rec_ng[t] - NH3_sell_ng[t] - Q_crk_in_ng[t],
                name=f"S_NH3_ng_dyn[{t}]"
            )

            m.addConstr(
                S_MEOH_g[t+1] == S_MEOH_g[t] + Q_meoh_g[t] - MEOH_sell_g[t],
                name=f"S_MEOH_g_dyn[{t}]"
            )
            m.addConstr(
                S_MEOH_ng[t+1] == S_MEOH_ng[t] + Q_meoh_ng[t] - MEOH_sell_ng[t],
                name=f"S_MEOH_ng_dyn[{t}]"
            )

            m.addConstr(H_to_NH3_g[t] + H_to_MEOH_g[t] + H_sell_g[t] + H_crk_fuel_g[t] <= S_H2_g[t], name=f"dir_H2_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] + H_to_MEOH_ng[t] + H_sell_ng[t] + H_crk_fuel_ng[t] <= S_H2_ng[t], name=f"dir_H2_ng[{t}]")

            m.addConstr(NH3_sell_g[t] + Q_crk_in_g[t] <= S_NH3_g[t], name=f"dir_NH3_g[{t}]")
            m.addConstr(NH3_sell_ng[t] + Q_crk_in_ng[t] <= S_NH3_ng[t], name=f"dir_NH3_ng[{t}]")

            m.addConstr(MEOH_sell_g[t] <= S_MEOH_g[t], name=f"dir_MEOH_g[{t}]")
            m.addConstr(MEOH_sell_ng[t] <= S_MEOH_ng[t], name=f"dir_MEOH_ng[{t}]")

        # ---------------- Objectives ----------------
        unmet_total = gp.quicksum(u_H2[t] + u_NH3[t] + u_MEOH[t] for t in T)

        emis_total = gp.LinExpr()
        for t in T:
            emis_total += 1000.0 * ef_obj_k[t] * P_grid[t]
            emis_total += de_NH3 * Q_nh3[t]
            emis_total += de_MEOH * Q_meoh[t]

        cost_total = gp.LinExpr()
        for t in T:
            cost_total += pr_eur[t] * P_grid[t]
            cost_total += c_H2O * (W_elz[t] + W_steam[t])
            cost_total += c_CO2 * CO2_buy[t]
            cost_total += C_su_elz * gp.quicksum(su_elz_m[i,t] for i in I_elz)
            cost_total += C_su_NH3 * su_nh3[t]
            cost_total += C_su_MEOH * su_meoh[t]
            cost_total += C_su_CRK * su_crk[t]

        m.setObjectiveN(unmet_total, index=0, priority=2, reltol=1e-4, abstol=1e-6, name="OBJ1_min_unmet")
        m.setObjectiveN(emis_total,  index=1, priority=1, reltol=1e-4, abstol=1e-6, name=f"OBJ2_min_emis_{EMIS_EF_MODE.lower()}")

        # ---------------- Solve ----------------
        m.optimize()

        status = int(safe_get_attr(m, "Status", -1))
        runtime = float(safe_get_attr(m, "Runtime", np.nan))
        nodecnt = float(safe_get_attr(m, "NodeCount", np.nan))
        solcnt  = int(safe_get_attr(m, "SolCount", -1))

        pass0 = get_objpass_metrics(m, 0)
        pass1 = get_objpass_metrics(m, 1)

        mip_gap = pass1["pass_mipgap"]
        obj_val = pass1["pass_objval"]
        obj_bnd = pass1["pass_objbound"]

        itercnt = safe_get_attr(m, "IterCount", np.nan)
        bariter = safe_get_attr(m, "BarIterCount", np.nan)
        work = safe_get_attr(m, "Work", np.nan)

        obj1_val = np.nan
        obj2_val = np.nan
        obj3_val = np.nan
        try:
            obj1_val = float(unmet_total.getValue())
        except Exception:
            pass
        try:
            obj2_val = float(emis_total.getValue())
        except Exception:
            pass
        try:
            obj3_val = float(cost_total.getValue()) #there is no obj 3, its just arbitary
        except Exception:
            pass

        is_optimal = (status == GRB.OPTIMAL)
        hit_time_limit = (status == GRB.TIME_LIMIT)
        has_incumbent = (solcnt is not None) and (solcnt >= 1)
        is_within_target_gap = bool(has_incumbent) and (mip_gap is not None) and (not np.isnan(mip_gap)) and (float(mip_gap) <= float(MIPGAP) + 1e-12)
        is_acceptable = bool(is_optimal or is_within_target_gap)

        solver_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "solver_status": status,
            "solver_status_name": status_name(status),
            "runtime_seconds": runtime,
            "time_limit_s": TIME_LIMIT_S,
            "mipgap_target": float(MIPGAP),
            "mip_gap": pass1["pass_mipgap"],
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),
            "obj_val": float(obj_val) if not np.isnan(obj_val) else np.nan,
            "obj_bound": float(obj_bnd) if not np.isnan(obj_bnd) else np.nan,

            "obj0_pass_status": pass0["pass_status"],
            "obj0_pass_status_name": pass0["pass_status_name"],
            "obj0_pass_runtime_s": pass0["pass_runtime"],
            "obj0_pass_objval": pass0["pass_objval"],
            "obj0_pass_objbound": pass0["pass_objbound"],
            "obj0_pass_mipgap": pass0["pass_mipgap"],

            "obj1_pass_status": pass1["pass_status"],
            "obj1_pass_status_name": pass1["pass_status_name"],
            "obj1_pass_runtime_s": pass1["pass_runtime"],
            "obj1_pass_objval": pass1["pass_objval"],
            "obj1_pass_objbound": pass1["pass_objbound"],
            "obj1_pass_mipgap": pass1["pass_mipgap"],

            "node_count": nodecnt,
            "sol_count": solcnt,
            "iter_count": float(itercnt) if not np.isnan(itercnt) else np.nan,
            "bar_iter_count": float(bariter) if not np.isnan(bariter) else np.nan,
            "work": float(work) if not np.isnan(work) else np.nan,
            "obj1_unmet_value": float(obj1_val) if not np.isnan(obj1_val) else np.nan,
            "obj2_emis_value": float(obj2_val) if not np.isnan(obj2_val) else np.nan,
            "metric_cost_value": float(obj3_val) if not np.isnan(obj3_val) else np.nan,
            "emis_ef_mode": EMIS_EF_MODE.lower(),
        })

        if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            try:
                m.computeIIS()
                iis_path = day_dir / f"{MODEL_TAG}_day{d:03d}_iis.ilp"
                m.write(str(iis_path))
            except Exception:
                pass
            raise RuntimeError(f"Day {d}: solver status {status} (not optimal/time_limit).")

        # ---------------- Extract hourly results ----------------
        def val(v):
            return float(v.X)

        rows = []
        unmet_rows = []

        for t in T:
            ts = day_hours[t]
            pv_used = pv_av[t] - val(P_pv_cur[t])
            wd_used = wd_av[t] - val(P_wd_cur[t])
            grid_mw = val(P_grid[t])
            res_used_mw = pv_used + wd_used
            res_curt_mw = val(P_pv_cur[t]) + val(P_wd_cur[t])

            P_elz = val(P_elz_g[t]) + val(P_elz_ng[t])
            P_nh3 = val(P_nh3_g[t]) + val(P_nh3_ng[t])
            P_meo = val(P_meoh_g[t]) + val(P_meoh_ng[t])
            P_crk = val(P_crk_g[t]) + val(P_crk_ng[t])

            H_total = val(H_prod[t])
            H_g = val(H_prod_g[t])
            H_ng = val(H_prod_ng[t])

            nh3_total = val(Q_nh3[t])
            nh3_g = val(Q_nh3_g[t])
            nh3_ng = val(Q_nh3_ng[t])

            me_total = val(Q_meoh[t])
            me_g = val(Q_meoh_g[t])
            me_ng = val(Q_meoh_ng[t])

            h2_sell_g = val(H_sell_g[t])
            h2_sell_ng = val(H_sell_ng[t])
            nh3_sell_g = val(NH3_sell_g[t])
            nh3_sell_ng = val(NH3_sell_ng[t])
            meoh_sell_g = val(MEOH_sell_g[t])
            meoh_sell_ng = val(MEOH_sell_ng[t])

            u1 = val(u_H2[t]); u2 = val(u_NH3[t]); u3 = val(u_MEOH[t])

            def share(g, ng):
                tot = g + ng
                return (g / tot) if tot > 1e-9 else np.nan

            H2_min_to_NH3  = mH2_to_NH3 * f_NH3_min * Q_NH3_CAP * int(round(val(y_nh3[t])))
            H2_min_to_MeOH = mH2_to_MEOH * f_MEOH_min * Q_MEOH_CAP * int(round(val(y_meoh[t])))
            H2_min_to_CRK  = kH2_fuel * f_CRK_min * Q_CRK_CAP * int(round(val(y_crk[t])))
            H2_min_committed = H2_min_to_NH3 + H2_min_to_MeOH + H2_min_to_CRK
            H2_tank_start = val(S_H2_g[t]) + val(S_H2_ng[t])
            H2_free_margin = H2_tank_start - H2_min_committed

            elz_mod = {}
            for i in I_elz:
                elz_mod[f"elz_m{i}_y"]     = int(round(val(y_elz_m[i,t])))
                elz_mod[f"elz_m{i}_su"]    = int(round(val(su_elz_m[i,t])))
                elz_mod[f"elz_m{i}_b_g"]   = int(round(val(b_elz_g[i,t])))
                elz_mod[f"elz_m{i}_P_tot"] = float(val(P_elz_m_tot[i,t]))
                elz_mod[f"elz_m{i}_P_g"]   = float(val(P_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_P_ng"]  = float(val(P_elz_m_ng[i,t]))
                elz_mod[f"elz_m{i}_H_tot"] = float(val(H_elz_m[i,t]))
                elz_mod[f"elz_m{i}_H_g"]   = float(val(H_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_H_ng"]  = float(val(H_elz_m_ng[i,t]))

            rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "H2_tank_start_kg": H2_tank_start,
                "H2_min_committed_kgph": H2_min_committed,
                "H2_free_margin_kg": H2_free_margin,
                "pv_avail_mw": pv_av[t],
                "wind_avail_mw": wd_av[t],
                "pv_curt_mw": val(P_pv_cur[t]),
                "wind_curt_mw": val(P_wd_cur[t]),
                "res_used_mw": res_used_mw,
                "res_curtailed_mw": res_curt_mw,
                "grid_import_mw": grid_mw,
                "da_price_eur_per_mwh": pr_eur[t],
                "bat_ch_mw": val(P_ch[t]),
                "bat_dis_mw": val(P_dis[t]),
                "battery_soc": val(SOC[t]),
                "battery_soc_next": val(SOC[t+1]),
                "elz_grid_mw": val(P_elz_ng[t]),
                "nh3_grid_mw": val(P_nh3_ng[t]),
                "meoh_grid_mw": val(P_meoh_ng[t]),
                "crk_grid_mw": val(P_crk_ng[t]),
                "P_elz_mw": P_elz,
                "P_nh3_mw": P_nh3,
                "P_meoh_mw": P_meo,
                "P_crk_mw": P_crk,
                "elz_power_green_share": share(val(P_elz_g[t]), val(P_elz_ng[t])),
                "nh3_power_green_share": share(val(P_nh3_g[t]), val(P_nh3_ng[t])),
                "meoh_power_green_share": share(val(P_meoh_g[t]), val(P_meoh_ng[t])),
                "crk_power_green_share": share(val(P_crk_g[t]), val(P_crk_ng[t])),
                "H2_prod_kgph": H_total,
                "H2_prod_g_kgph": H_g,
                "H2_prod_ng_kgph": H_ng,
                "NH3_prod_kgph": nh3_total,
                "NH3_prod_g_kgph": nh3_g,
                "NH3_prod_ng_kgph": nh3_ng,
                "MeOH_prod_kgph": me_total,
                "MeOH_prod_g_kgph": me_g,
                "MeOH_prod_ng_kgph": me_ng,
                "crk_nh3_in_kgph": val(Q_crk_in[t]),
                "crk_h2_out_kgph": val(H_crk_out_g[t]) + val(H_crk_out_ng[t]),
                "crk_h2_fuel_kgph": val(H_crk_fuel_g[t]) + val(H_crk_fuel_ng[t]),
                "steam_kgph": val(Steam_g[t]) + val(Steam_ng[t]),
                "water_elz_m3ph": val(W_elz[t]),
                "water_steam_m3ph": val(W_steam[t]),
                "co2_buy_kgph": val(CO2_buy[t]),

                "H2_sell_g_kgph": h2_sell_g,
                "H2_sell_ng_kgph": h2_sell_ng,
                "H2_sell_kgph": h2_sell_g + h2_sell_ng,

                "NH3_sell_g_kgph": nh3_sell_g,
                "NH3_sell_ng_kgph": nh3_sell_ng,
                "NH3_sell_kgph": nh3_sell_g + nh3_sell_ng,

                "MeOH_sell_g_kgph": meoh_sell_g,
                "MeOH_sell_ng_kgph": meoh_sell_ng,
                "MeOH_sell_kgph": meoh_sell_g + meoh_sell_ng,

                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "tank_H2_g_kg": val(S_H2_g[t]),
                "tank_H2_ng_kg": val(S_H2_ng[t]),
                "tank_NH3_g_kg": val(S_NH3_g[t]),
                "tank_NH3_ng_kg": val(S_NH3_ng[t]),
                "tank_MeOH_g_kg": val(S_MEOH_g[t]),
                "tank_MeOH_ng_kg": val(S_MEOH_ng[t]),
                "b_ch": int(val(P_ch[t]) > 1e-6),

                "n_elz_on": int(round(val(n_elz_on[t]))),
                "y_elz": int(round(val(y_elz[t]))),
                "su_elz": int(round(val(su_elz[t]))),
                "y_nh3": int(round(val(y_nh3[t]))),
                "su_nh3": int(round(val(su_nh3[t]))),
                "y_meoh": int(round(val(y_meoh[t]))),
                "su_meoh": int(round(val(su_meoh[t]))),
                "y_crk": int(round(val(y_crk[t]))),
                "su_crk": int(round(val(su_crk[t]))),
                **existing_solution_exports(t, locals()),
                **elz_mod,
            })

            unmet_rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "da_price_eur_per_mwh": pr_eur[t],
                "grid_import_mw": grid_mw,
                "res_curtailed_mw": res_curt_mw,
            })

        day_hourly = pd.DataFrame(rows)


        day_hourly_all = day_hourly.copy()
        day_hourly_all["day_id"] = int(d)
        all_hourly_frames.append(day_hourly_all)

        # ---------------- Daily post-processing ----------------
        grid_cost_eur   = float(np.sum([pr_eur[t] * val(P_grid[t]) for t in T]))
        water_cost_eur  = float(c_H2O * np.sum([val(W_elz[t]) + val(W_steam[t]) for t in T]))
        co2_cost_eur    = float(c_CO2 * np.sum([val(CO2_buy[t]) for t in T]))
        startup_elz_eur = float(C_su_elz * np.sum([val(su_elz[t]) for t in T]))
        startup_nh3_eur = float(C_su_NH3 * np.sum([val(su_nh3[t]) for t in T]))
        startup_meoh_eur = float(C_su_MEOH * np.sum([val(su_meoh[t]) for t in T]))
        startup_crk_eur = float(C_su_CRK * np.sum([val(su_crk[t]) for t in T]))

        total_unmet_kg = float(np.sum(day_hourly["unmet_h2_kgph"].values + day_hourly["unmet_nh3_kgph"].values + day_hourly["unmet_meoh_kgph"].values))
        total_cost = float(obj3_val) if not np.isnan(obj3_val) else float(grid_cost_eur + water_cost_eur + co2_cost_eur + startup_elz_eur + startup_nh3_eur + startup_meoh_eur + startup_crk_eur)

        grid_emis_aef = float(np.sum([val(P_grid[t]) * 1000.0 * aef_k[t] for t in T]))
        grid_emis_mef_upramp = float(np.sum([val(P_grid[t]) * 1000.0 * mef_upramp_k[t] for t in T]))
        grid_emis_mef_soft = float(np.sum([val(P_grid[t]) * 1000.0 * mef_soft_k[t] for t in T]))

        direct_emis_nh3 = float(np.sum([de_NH3 * val(Q_nh3[t]) for t in T]))
        direct_emis_meoh = float(np.sum([de_MEOH * val(Q_meoh[t]) for t in T]))
        direct_emis_total = direct_emis_nh3 + direct_emis_meoh

        demand_H2_kg_day   = float(np.sum(dem_h2))
        demand_NH3_kg_day  = float(np.sum(dem_nh3))
        demand_MeOH_kg_day = float(np.sum(dem_meoh))

        elz_kwh_day  = float(np.sum(day_hourly["P_elz_mw"].values * 1000.0))
        nh3_kwh_day  = float(np.sum(day_hourly["P_nh3_mw"].values * 1000.0))
        meoh_kwh_day = float(np.sum(day_hourly["P_meoh_mw"].values * 1000.0))
        crk_kwh_day  = float(np.sum(day_hourly["P_crk_mw"].values * 1000.0))
        units_kwh_day = elz_kwh_day + nh3_kwh_day + meoh_kwh_day + crk_kwh_day

        res_gen_kwh = float(np.sum((pv_av + wd_av) * 1000.0))
        res_used_kwh = float(np.sum((pv_av - day_hourly["pv_curt_mw"].values + wd_av - day_hourly["wind_curt_mw"].values) * 1000.0))
        res_curt_kwh = float(np.sum((day_hourly["pv_curt_mw"].values + day_hourly["wind_curt_mw"].values) * 1000.0))
        grid_import_kwh = float(np.sum(day_hourly["grid_import_mw"].values * 1000.0))
        grid_share_percent = 100.0 * grid_import_kwh / max(grid_import_kwh + res_used_kwh, 1e-9)

        H2_g_sum = float(np.sum(day_hourly["H2_prod_g_kgph"].values))
        H2_ng_sum = float(np.sum(day_hourly["H2_prod_ng_kgph"].values))
        NH3_g_sum = float(np.sum(day_hourly["NH3_prod_g_kgph"].values))
        NH3_ng_sum = float(np.sum(day_hourly["NH3_prod_ng_kgph"].values))
        Me_g_sum = float(np.sum(day_hourly["MeOH_prod_g_kgph"].values))
        Me_ng_sum = float(np.sum(day_hourly["MeOH_prod_ng_kgph"].values))

        def green_share(sum_g, sum_ng):
            tot = sum_g + sum_ng
            return (sum_g / tot) if tot > 1e-9 else np.nan

        daily_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "mip_gap": float(mip_gap) if mip_gap is not None else np.nan,
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),

            "total_cost_eur": total_cost,
            "obj1_total_unmet_kg": total_unmet_kg,
            "obj2_total_emissions_kg": float(obj2_val) if not np.isnan(obj2_val) else np.nan,
            "metric_operating_cost_eur": total_cost,
            "emis_ef_mode": EMIS_EF_MODE.lower(),

            "res_generation_kWh": res_gen_kwh,
            "res_used_kWh": res_used_kwh,
            "res_curtailed_kWh": res_curt_kwh,
            "grid_import_kWh": grid_import_kwh,
            "grid_share_percent": grid_share_percent,

            "battery_soc_avg": float(np.mean(day_hourly["battery_soc"].values)),
            "tank_H2_level_avg_kg": float(np.mean(day_hourly["tank_H2_g_kg"].values + day_hourly["tank_H2_ng_kg"].values)),
            "tank_NH3_level_avg_kg": float(np.mean(day_hourly["tank_NH3_g_kg"].values + day_hourly["tank_NH3_ng_kg"].values)),
            "tank_MeOH_level_avg_kg": float(np.mean(day_hourly["tank_MeOH_g_kg"].values + day_hourly["tank_MeOH_ng_kg"].values)),

            "H2_prod_green_share": green_share(H2_g_sum, H2_ng_sum),
            "NH3_prod_green_share": green_share(NH3_g_sum, NH3_ng_sum),
            "MeOH_prod_green_share": green_share(Me_g_sum, Me_ng_sum),

            # grid emissions only
            "emissions_grid_aef_kg": grid_emis_aef,
            "emissions_grid_mef_upramp_kg": grid_emis_mef_upramp,
            "emissions_grid_mef_soft_kg": grid_emis_mef_soft,

            # direct emissions separate
            "emissions_direct_nh3_kg": direct_emis_nh3,
            "emissions_direct_meoh_kg": direct_emis_meoh,
            "emissions_direct_total_kg": direct_emis_total,

            "unmet_H2_kg": float(np.sum(day_hourly["unmet_h2_kgph"].values)),
            "unmet_NH3_kg": float(np.sum(day_hourly["unmet_nh3_kgph"].values)),
            "unmet_MeOH_kg": float(np.sum(day_hourly["unmet_meoh_kgph"].values)),

            "demand_H2_kg_day": demand_H2_kg_day,
            "demand_NH3_kg_day": demand_NH3_kg_day,
            "demand_MeOH_kg_day": demand_MeOH_kg_day,

            "elz_consumption_kWh_day": elz_kwh_day,
            "nh3_consumption_kWh_day": nh3_kwh_day,
            "meoh_consumption_kWh_day": meoh_kwh_day,
            "crk_consumption_kWh_day": crk_kwh_day,
            "units_total_consumption_kWh_day": units_kwh_day,

            # sales accounting
            "H2_sell_g_kg_day": float(np.sum(day_hourly["H2_sell_g_kgph"].values)),
            "H2_sell_ng_kg_day": float(np.sum(day_hourly["H2_sell_ng_kgph"].values)),
            "H2_sell_total_kg_day": float(np.sum(day_hourly["H2_sell_kgph"].values)),

            "NH3_sell_g_kg_day": float(np.sum(day_hourly["NH3_sell_g_kgph"].values)),
            "NH3_sell_ng_kg_day": float(np.sum(day_hourly["NH3_sell_ng_kgph"].values)),
            "NH3_sell_total_kg_day": float(np.sum(day_hourly["NH3_sell_kgph"].values)),

            "MeOH_sell_g_kg_day": float(np.sum(day_hourly["MeOH_sell_g_kgph"].values)),
            "MeOH_sell_ng_kg_day": float(np.sum(day_hourly["MeOH_sell_ng_kgph"].values)),
            "MeOH_sell_total_kg_day": float(np.sum(day_hourly["MeOH_sell_kgph"].values)),

            "eod_H2_total_kg": float(val(S_H2_g[24]) + val(S_H2_ng[24])),
            "eod_NH3_total_kg": float(val(S_NH3_g[24]) + val(S_NH3_ng[24])),
            "eod_MeOH_total_kg": float(val(S_MEOH_g[24]) + val(S_MEOH_ng[24])),
            "eod_req_H2_kg": float(ALPHA_TANK_FILL * S_H2_CAP),
            "eod_req_NH3_kg": float(ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_req_MeOH_kg": float(ALPHA_TANK_FILL * S_MEOH_CAP),
            "eod_margin_H2_kg": float((val(S_H2_g[24]) + val(S_H2_ng[24])) - ALPHA_TANK_FILL * S_H2_CAP),
            "eod_margin_NH3_kg": float((val(S_NH3_g[24]) + val(S_NH3_ng[24])) - ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_margin_MeOH_kg": float((val(S_MEOH_g[24]) + val(S_MEOH_ng[24])) - ALPHA_TANK_FILL * S_MEOH_CAP),

            "cost_grid_eur": grid_cost_eur,
            "cost_water_eur": water_cost_eur,
            "cost_co2_eur": co2_cost_eur,
            "cost_startup_elz_eur": startup_elz_eur,
            "cost_startup_nh3_eur": startup_nh3_eur,
            "cost_startup_meoh_eur": startup_meoh_eur,
            "cost_startup_crk_eur": startup_crk_eur,
        })

        # ---------------- Carry-over update ----------------
        carry["soc"] = val(SOC[24])
        carry["S_H2_g"] = val(S_H2_g[24]); carry["S_H2_ng"] = val(S_H2_ng[24])
        carry["S_NH3_g"] = val(S_NH3_g[24]); carry["S_NH3_ng"] = val(S_NH3_ng[24])
        carry["S_MEOH_g"] = val(S_MEOH_g[24]); carry["S_MEOH_ng"] = val(S_MEOH_ng[24])
        carry["y_elz_prev_mod"] = [int(round(val(y_elz_m[i,23]))) for i in I_elz]
        carry["y_nh3_prev"]  = int(round(val(y_nh3[23])))
        carry["y_meoh_prev"] = int(round(val(y_meoh[23])))
        carry["y_crk_prev"]  = int(round(val(y_crk[23])))
        carry["Q_nh3_prev"]  = float(val(Q_nh3[23]))
        carry["Q_meoh_prev"] = float(val(Q_meoh[23]))
        carry["Q_crk_prev"]  = float(val(Q_crk_in[23]))

        state_rows.append({
            "run_id": RUN_ID,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "SOC_end": carry["soc"],
            "S_H2_g_end": carry["S_H2_g"],
            "S_H2_ng_end": carry["S_H2_ng"],
            "S_NH3_g_end": carry["S_NH3_g"],
            "S_NH3_ng_end": carry["S_NH3_ng"],
            "S_MEOH_g_end": carry["S_MEOH_g"],
            "S_MEOH_ng_end": carry["S_MEOH_ng"],
            "y_elz_prev_mod_end": carry["y_elz_prev_mod"],
            "y_nh3_end": carry["y_nh3_prev"],
            "y_meoh_end": carry["y_meoh_prev"],
            "y_crk_end": carry["y_crk_prev"],
            "Q_nh3_end": carry["Q_nh3_prev"],
            "Q_meoh_end": carry["Q_meoh_prev"],
            "Q_crk_end": carry["Q_crk_prev"],
        })

        eps = 1e-6
        sim_cd = float(np.sum((day_hourly["bat_ch_mw"].values > eps) & (day_hourly["bat_dis_mw"].values > eps)))
        soc_viol = float(np.sum((day_hourly["battery_soc"].values < SOC_MIN - 1e-6) | (day_hourly["battery_soc"].values > SOC_MAX + 1e-6)))
        unmet_any = float(np.sum((day_hourly["unmet_h2_kgph"].values > eps) | (day_hourly["unmet_nh3_kgph"].values > eps) | (day_hourly["unmet_meoh_kgph"].values > eps)))


    # ---------------- End-of-run aggregation ----------------
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        raise RuntimeError("No days were solved (daily_df is empty).")

    solver_df = pd.DataFrame(solver_rows)
    state_df  = pd.DataFrame(state_rows)

    if len(all_hourly_frames) > 0:
        hourly_all_df = pd.concat(all_hourly_frames, ignore_index=True)
        hourly_all_df.to_csv(OUT_POST / "hourly_timeseries_all_days.csv", index=False)
    else:
        hourly_all_df = pd.DataFrame()

    daily_df.to_csv(OUT_POST / "daily_summary.csv", index=False)
    solver_df.to_csv(OUT_POST / "solver_log_daily.csv", index=False)
    state_df.to_csv(OUT_POST / "state_trace.csv", index=False)

    annual_summary = {
        "run_id": RUN_ID,
        "model_tag": MODEL_TAG,
        "scenario_tag": SCENARIO_TAG,
        "demand_tag": DEMAND_TAG,
        "start_day_of_year": START_DAY_OF_YEAR,
        "n_days_run": int(len(daily_df)),

        "total_cost_eur": float(daily_df["total_cost_eur"].sum()),

        "total_grid_emissions_aef_kg": float(daily_df["emissions_grid_aef_kg"].sum()),
        "total_grid_emissions_mef_upramp_kg": float(daily_df["emissions_grid_mef_upramp_kg"].sum()),
        "total_grid_emissions_mef_soft_kg": float(daily_df["emissions_grid_mef_soft_kg"].sum()),

        "total_direct_emissions_nh3_kg": float(daily_df["emissions_direct_nh3_kg"].sum()),
        "total_direct_emissions_meoh_kg": float(daily_df["emissions_direct_meoh_kg"].sum()),
        "total_direct_emissions_kg": float(daily_df["emissions_direct_total_kg"].sum()),

        "total_grid_import_kWh": float(daily_df["grid_import_kWh"].sum()),
        "total_res_used_kWh": float(daily_df["res_used_kWh"].sum()),
        "total_res_curtailed_kWh": float(daily_df["res_curtailed_kWh"].sum()),
        "avg_grid_share_percent": float(daily_df["grid_share_percent"].mean()),

        "total_unmet_H2_kg": float(daily_df["unmet_H2_kg"].sum()),
        "total_unmet_NH3_kg": float(daily_df["unmet_NH3_kg"].sum()),
        "total_unmet_MeOH_kg": float(daily_df["unmet_MeOH_kg"].sum()),

        "total_H2_sell_g_kg": float(daily_df["H2_sell_g_kg_day"].sum()),
        "total_H2_sell_ng_kg": float(daily_df["H2_sell_ng_kg_day"].sum()),
        "total_H2_sell_kg": float(daily_df["H2_sell_total_kg_day"].sum()),

        "total_NH3_sell_g_kg": float(daily_df["NH3_sell_g_kg_day"].sum()),
        "total_NH3_sell_ng_kg": float(daily_df["NH3_sell_ng_kg_day"].sum()),
        "total_NH3_sell_kg": float(daily_df["NH3_sell_total_kg_day"].sum()),

        "total_MeOH_sell_g_kg": float(daily_df["MeOH_sell_g_kg_day"].sum()),
        "total_MeOH_sell_ng_kg": float(daily_df["MeOH_sell_ng_kg_day"].sum()),
        "total_MeOH_sell_kg": float(daily_df["MeOH_sell_total_kg_day"].sum()),
    }
    pd.DataFrame([annual_summary]).to_csv(OUT_POST / "annual_summary.csv", index=False)


    return OUT_RUN


# M2 formulation
def _run_m2(config):
    # =====================  M2 =====================
    # min unmet demand >> min cost with daily emissions cap

    # ---------------- Header / metadata ----------------
    MODEL_TAG = "M2"
    FORMULATION_TAG = "M2_final_v2"
    SCENARIO_TAG = "S1"
    DEMAND_TAG = "D1"
    GREEN_TAG = "S1_green_split_with_rfnbo_hourly"

    # Choose one emissions accounting basis for the cap constraint:
    #   "aef"
    #   "mef_upramp"
    #   "mef_soft"
    EMISSIONS_KEY = config["signal"]

    # Daily emission-cap interpolation parameter.
    ALPHA = float(config.get("alpha", 0.5))

    # ---------------- Imports ----------------
    import gurobipy as gp
    from gurobipy import GRB

    # ---------------- Global configuration ----------------
    DATA_DIR = Path(config["model_ready"]).resolve().parent
    MR = Path(config["model_ready"]).resolve()

    # Required model-ready files
    RES_CSV         = MR / "res_hourly_2024.csv"
    DEM_CSV         = MR / "demand_hourly_2024.csv"
    PRICE_CSV       = MR / "grid_price_hourly_2024.csv"
    AEF_CSV         = MR / "grid_aef_hourly_2024.csv"
    MEF_UPRAMP_CSV  = MR / "grid_mef_upramp_hourly_2024.csv"
    MEF_SOFT_CSV    = MR / "grid_mef_soft_hourly_2024.csv"
    PARAMS_JSON     = MR / "params_scenario_all.json"

    # Bounds file
    M2_CAP_RANGES_CSV = Path(config["bounds"]).resolve()

    # Emission column names
    AEF_COL        = "grid_aef_predicted_kg_per_kwh"
    MEF_UPRAMP_COL = "grid_mef_upramp_pred_kg_per_kwh"
    MEF_SOFT_COL   = "grid_mef_soft_kg_per_kwh"

    # ---------------- Run controls ----------------
    N_DAYS_TO_RUN = int(config.get("days", 1))
    START_DAY_OF_YEAR = 1  # Resume requires explicit carryover, not supported by this entry point.
    MIPGAP            = 0.01
    TIME_LIMIT_S = float(config.get("time_limit", 350))
    OUTPUTFLAG = int(config.get("solver_output", 1))

    # ---------------- License handling ----------------

    # ---------------- Run ID builder ----------------
    def build_run_id():
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        return f"{MODEL_TAG}__{SCENARIO_TAG}__{DEMAND_TAG}__start{START_DAY_OF_YEAR:03d}__ndays{N_DAYS_TO_RUN:03d}__{stamp}"

    RUN_ID = build_run_id()

    # ---------------- Output paths ----------------
    OUT_BASE = Path(config["output"]).resolve()
    OUT_RUN   = OUT_BASE / RUN_ID
    OUT_POST  = OUT_RUN / "post"
    OUT_RUN.mkdir(parents=True, exist_ok=False)
    OUT_POST.mkdir(parents=True, exist_ok=True)


    # ---------------- Input loading ----------------

    res        = read_hourly_csv(RES_CSV)[["pv_mw","wind_mw","res_total_mw"]]
    dem        = read_hourly_csv(DEM_CSV)[["h2_demand_kgph","nh3_demand_kgph","meoh_demand_kgph"]]
    price      = read_hourly_csv(PRICE_CSV)[["da_elec_price_EURperMWh"]]
    aef        = read_hourly_csv(AEF_CSV)[[AEF_COL]]
    mef_upramp = read_hourly_csv(MEF_UPRAMP_CSV)[[MEF_UPRAMP_COL]]
    mef_soft   = read_hourly_csv(MEF_SOFT_CSV)[[MEF_SOFT_COL]]

    with open(PARAMS_JSON, "r") as f:
        params = json.load(f)

    # ---------------- Bounds file ----------------
    if not M2_CAP_RANGES_CSV.exists():
        raise FileNotFoundError(f"Missing M2 cap ranges file: {M2_CAP_RANGES_CSV}")

    m2_bounds_raw = pd.read_csv(M2_CAP_RANGES_CSV)

    required_cols = [
        "day_of_year", "scenario",
        "mef_upramp_lb_kg", "mef_upramp_up_kg",
        "mef_soft_lb_kg", "mef_soft_up_kg",
        "aef_lb_kg", "aef_up_kg"
    ]
    missing = [c for c in required_cols if c not in m2_bounds_raw.columns]
    if missing:
        raise ValueError(f"{M2_CAP_RANGES_CSV.name}: missing columns: {missing}. Found: {list(m2_bounds_raw.columns)}")

    m2_bounds_raw["day_of_year"] = m2_bounds_raw["day_of_year"].astype(int)
    m2_bounds_raw["scenario"] = m2_bounds_raw["scenario"].astype(str)

    m2_bounds = (
        m2_bounds_raw.loc[m2_bounds_raw["scenario"].eq(str(SCENARIO_TAG))].copy()
        .sort_values("day_of_year")
        .set_index("day_of_year")
    )

    if m2_bounds.empty:
        raise ValueError(
            f"M2 bounds file has no rows for scenario='{SCENARIO_TAG}'. "
            f"Available scenarios: {sorted(m2_bounds_raw['scenario'].unique().tolist())}"
        )

    def compute_daily_cap(lb_kg: float, ub_kg: float, alpha: float):
        lb = float(lb_kg)
        ub = float(ub_kg)

        is_negative_day = bool(ub < 0.0)
        is_cap_repair_day = bool((ub >= 0.0) and (ub < lb))

        if is_negative_day:
            return ub, np.nan, is_negative_day, is_cap_repair_day

        if is_cap_repair_day:
            return ub, np.nan, is_negative_day, is_cap_repair_day

        cap = lb + float(alpha) * (ub - lb)
        return cap, float(alpha), is_negative_day, is_cap_repair_day

    def alpha_tag(a: float) -> str:
        s = f"{a:.3f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")

    # ---------------- Input validation ----------------
    idx_2024 = pd.date_range("2024-01-01 00:00:00+00:00", "2024-12-31 23:00:00+00:00", freq="h", tz="UTC")


    res        = assert_full_hourly(res, "res")
    dem        = assert_full_hourly(dem, "dem")
    price      = assert_full_hourly(price, "price")
    aef        = assert_full_hourly(aef, "aef")
    mef_upramp = assert_full_hourly(mef_upramp, "mef_upramp")
    mef_soft   = assert_full_hourly(mef_soft, "mef_soft")

    # ---------------- Scalar parameters (aligned with M0-V2 / M1-V2) ----------------
    P_GRID_MAX_MW = 295.58

    # Battery
    P_BAT_MAX_MW  = 50.0
    E_BAT_MAX_MWH = 100.32
    ETA_CH        = 0.964
    ETA_DIS       = 0.964
    SOC_MIN       = 0.15
    SOC_MAX       = 0.95
    SOC_INIT      = 0.5

    # Electrolyzer
    P_ELZ_MIN = 8
    P_ELZ_MAX = 52.25
    H_ELZ_MAX = 914.39161
    A_elz = -0.075045052
    B_elz = 21.73040904
    C_elz = -16.14458074
    N_stack = 4
    omega_H2O = 0.015
    c_H2O = 2.24
    C_su_elz = 1392

    P_ELZ_MAX_TOT = N_stack * P_ELZ_MAX
    H_ELZ_MAX_TOT = N_stack * H_ELZ_MAX

    H_min = A_elz*(P_ELZ_MIN**2) + B_elz*P_ELZ_MIN + C_elz
    H_max = A_elz*(P_ELZ_MAX**2) + B_elz*P_ELZ_MAX + C_elz
    Q1 = (H_max - H_min) / (P_ELZ_MAX - P_ELZ_MIN)
    Q0 = H_min - Q1 * P_ELZ_MIN

    # NH3
    Q_NH3_CAP = 13472.04
    f_NH3_min = 0.61
    RR_NH3 = 0.2
    mH2_to_NH3 = 0.18
    mN2_to_NH3 = 0.87
    SEC_HB = 0.36
    SEC_ASU = 0.068
    C_su_NH3 = 2397.46
    de_NH3 = 0.00519

    # MeOH
    Q_MEOH_CAP = 5818.856
    f_MEOH_min = 0.1
    R_MEOH = 0.5 * Q_MEOH_CAP
    mH2_to_MEOH = 0.22
    SEC_MEOH = 161.75/1000.0
    mCO2_to_MEOH = 1.59
    c_CO2 = 0.2
    C_su_MEOH = 1159.37
    de_MEOH = 0.08

    # Cracker
    Q_CRK_CAP = 6062.416
    f_CRK_min = 0.05
    R_CRK = 0.5 * Q_CRK_CAP
    kH2_out = 0.12024
    kNH3_rec = 0.11848
    kH2_fuel = 0.035294
    SEC_CRK_E = 0.524667
    kSteam = 0.941176
    eSteam = 0.75
    omega_steam = 0.001
    C_su_CRK = 1000.0

    # Tanks
    H_TANK = 12.0
    ALPHA_TANK_FILL = 0.15

    S_H2_CAP   = H_TANK * H_ELZ_MAX_TOT
    S_NH3_CAP  = H_TANK * Q_NH3_CAP
    S_MEOH_CAP = H_TANK * Q_MEOH_CAP

    S_H2_INIT   = ALPHA_TANK_FILL * S_H2_CAP
    S_NH3_INIT  = ALPHA_TANK_FILL * S_NH3_CAP
    S_MEOH_INIT = ALPHA_TANK_FILL * S_MEOH_CAP

    # ---------------- Helpers ----------------

    # ---------------- Carry-over initialization ----------------
    carry = {
        "soc": SOC_INIT,
        "S_H2_g": 0.5*S_H2_INIT,
        "S_H2_ng": 0.5*S_H2_INIT,
        "S_NH3_g": 0.5*S_NH3_INIT,
        "S_NH3_ng": 0.5*S_NH3_INIT,
        "S_MEOH_g": 0.5*S_MEOH_INIT,
        "S_MEOH_ng": 0.5*S_MEOH_INIT,
        "y_elz_prev_mod": [0]*N_stack,
        "y_nh3_prev": 0,
        "y_meoh_prev": 0,
        "y_crk_prev": 0,
        "Q_nh3_prev": 0.0,
        "Q_meoh_prev": 0.0,
        "Q_crk_prev": 0.0,
    }

    daily_rows = []
    solver_rows = []
    state_rows = []
    all_hourly_frames = []

    # ---------------- Run loop ----------------
    t0_global = time.time()

    for d in range(START_DAY_OF_YEAR, START_DAY_OF_YEAR + N_DAYS_TO_RUN):
        day_hours = day_to_time_slice(d)

        if day_hours[0] < idx_2024[0] or day_hours[-1] > idx_2024[-1]:
            break


        pv_av    = res.loc[day_hours, "pv_mw"].values.astype(float)
        wd_av    = res.loc[day_hours, "wind_mw"].values.astype(float)
        pr_eur   = price.loc[day_hours, "da_elec_price_EURperMWh"].values.astype(float)
        dem_h2   = dem.loc[day_hours, "h2_demand_kgph"].values.astype(float)
        dem_nh3  = dem.loc[day_hours, "nh3_demand_kgph"].values.astype(float)
        dem_meoh = dem.loc[day_hours, "meoh_demand_kgph"].values.astype(float)

        aef_k        = aef.loc[day_hours, AEF_COL].values.astype(float)
        mef_upramp_k = mef_upramp.loc[day_hours, MEF_UPRAMP_COL].values.astype(float)
        mef_soft_k   = mef_soft.loc[day_hours, MEF_SOFT_COL].values.astype(float)

        # ---------------- Daily cap selection ----------------
        if d not in m2_bounds.index:
            raise ValueError(f"M2 bounds table missing day_of_year={d} for scenario='{SCENARIO_TAG}'")

        if EMISSIONS_KEY == "aef":
            lb_col, ub_col = "aef_lb_kg", "aef_up_kg"
        elif EMISSIONS_KEY == "mef_upramp":
            lb_col, ub_col = "mef_upramp_lb_kg", "mef_upramp_up_kg"
        elif EMISSIONS_KEY == "mef_soft":
            lb_col, ub_col = "mef_soft_lb_kg", "mef_soft_up_kg"
        else:
            raise ValueError("EMISSIONS_KEY must be one of: 'aef', 'mef_upramp', 'mef_soft'")

        lb_day = float(m2_bounds.loc[d, lb_col])
        ub_day = float(m2_bounds.loc[d, ub_col])
        EMIS_CAP_DAY_KG, ALPHA_USED_DAY, IS_NEGATIVE_DAY, IS_CAP_REPAIR_DAY = compute_daily_cap(lb_day, ub_day, ALPHA)

        # ---------------- Model ----------------
        m = gp.Model(f"{MODEL_TAG}_day{d:03d}")
        m.Params.MIPGap = MIPGAP
        m.Params.TimeLimit = TIME_LIMIT_S
        m.Params.OutputFlag = OUTPUTFLAG
        m.Params.NumericFocus = 1
        m.Params.MultiObjMethod = 2
        m.Params.MultiObjPre = 1

        T = range(24)
        T2 = range(25)

        S_sources = ["pv", "wd", "grid", "bat"]
        K_sinks = ["elz", "NH3", "MeOH", "crk", "bch"]
        S_res = ["pv", "wd"]

        # ---------------- Variables ----------------
        P = m.addVars(S_sources, K_sinks, T, lb=0.0, name="P_s2k")
        P_pv_cur = m.addVars(T, lb=0.0, name="P_pv_cur")
        P_wd_cur = m.addVars(T, lb=0.0, name="P_wd_cur")
        P_grid = m.addVars(T, lb=0.0, ub=P_GRID_MAX_MW, name="P_grid")

        P_ch  = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_ch")
        P_dis = m.addVars(T, lb=0.0, ub=P_BAT_MAX_MW, name="P_dis")
        SOC   = m.addVars(T2, lb=SOC_MIN, ub=SOC_MAX, name="SOC")

        Pk_g  = m.addVars(K_sinks, T, lb=0.0, name="P_sink_g")
        Pk_ng = m.addVars(K_sinks, T, lb=0.0, name="P_sink_ng")

        # Electrolyzer
        I_elz = range(N_stack)
        y_elz_m  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="y_elz_m")
        su_elz_m = m.addVars(I_elz, T, vtype=GRB.BINARY, name="su_elz_m")
        b_elz_g  = m.addVars(I_elz, T, vtype=GRB.BINARY, name="b_elz_g")

        P_elz_m_g   = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_g")
        P_elz_m_ng  = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_ng")
        P_elz_m_tot = m.addVars(I_elz, T, lb=0.0, ub=P_ELZ_MAX, name="P_elz_m_tot")

        H_elz_m    = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m")
        H_elz_m_g  = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_g")
        H_elz_m_ng = m.addVars(I_elz, T, lb=0.0, ub=H_ELZ_MAX, name="H_elz_m_ng")

        y_elz    = m.addVars(T, vtype=GRB.BINARY, name="y_elz")
        n_elz_on = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="n_elz_on")
        su_elz   = m.addVars(T, vtype=GRB.INTEGER, lb=0, ub=N_stack, name="su_elz")

        P_elz_g   = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_g")
        P_elz_ng  = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_ng")
        P_elz_tot = m.addVars(T, lb=0.0, ub=N_stack*P_ELZ_MAX, name="P_elz_tot")
        H_prod    = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod")
        H_prod_g  = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_g")
        H_prod_ng = m.addVars(T, lb=0.0, ub=N_stack*H_ELZ_MAX, name="H_prod_ng")
        W_elz     = m.addVars(T, lb=0.0, name="W_elz_m3ph")

        # NH3
        y_nh3   = m.addVars(T, vtype=GRB.BINARY, name="y_nh3")
        su_nh3  = m.addVars(T, vtype=GRB.BINARY, name="su_nh3")
        Q_nh3_g = m.addVars(T, lb=0.0, name="Q_nh3_g")
        Q_nh3_ng = m.addVars(T, lb=0.0, name="Q_nh3_ng")
        Q_nh3   = m.addVars(T, lb=0.0, name="Q_nh3")
        P_nh3_g = m.addVars(T, lb=0.0, name="P_nh3_g")
        P_nh3_ng = m.addVars(T, lb=0.0, name="P_nh3_ng")

        # MeOH
        y_meoh   = m.addVars(T, vtype=GRB.BINARY, name="y_meoh")
        su_meoh  = m.addVars(T, vtype=GRB.BINARY, name="su_meoh")
        Q_meoh_g = m.addVars(T, lb=0.0, name="Q_meoh_g")
        Q_meoh_ng = m.addVars(T, lb=0.0, name="Q_meoh_ng")
        Q_meoh   = m.addVars(T, lb=0.0, name="Q_meoh")
        P_meoh_g = m.addVars(T, lb=0.0, name="P_meoh_g")
        P_meoh_ng = m.addVars(T, lb=0.0, name="P_meoh_ng")
        CO2_buy  = m.addVars(T, lb=0.0, name="CO2_buy_kgph")

        # Cracker
        y_crk    = m.addVars(T, vtype=GRB.BINARY, name="y_crk")
        su_crk   = m.addVars(T, vtype=GRB.BINARY, name="su_crk")
        Q_crk_in_g = m.addVars(T, lb=0.0, name="Q_crk_in_g")
        Q_crk_in_ng = m.addVars(T, lb=0.0, name="Q_crk_in_ng")
        Q_crk_in = m.addVars(T, lb=0.0, name="Q_crk_in")

        H_crk_out_g  = m.addVars(T, lb=0.0, name="H_crk_out_g")
        H_crk_out_ng = m.addVars(T, lb=0.0, name="H_crk_out_ng")
        NH3_rec_g    = m.addVars(T, lb=0.0, name="NH3_rec_g")
        NH3_rec_ng   = m.addVars(T, lb=0.0, name="NH3_rec_ng")
        H_crk_fuel_g  = m.addVars(T, lb=0.0, name="H_crk_fuel_g")
        H_crk_fuel_ng = m.addVars(T, lb=0.0, name="H_crk_fuel_ng")
        Steam_g      = m.addVars(T, lb=0.0, name="Steam_g_kgph")
        Steam_ng     = m.addVars(T, lb=0.0, name="Steam_ng_kgph")
        P_crk_g      = m.addVars(T, lb=0.0, name="P_crk_g")
        P_crk_ng     = m.addVars(T, lb=0.0, name="P_crk_ng")
        W_steam      = m.addVars(T, lb=0.0, name="W_steam_m3ph")

        # Tanks
        S_H2_g   = m.addVars(T2, lb=0.0, name="S_H2_g")
        S_H2_ng  = m.addVars(T2, lb=0.0, name="S_H2_ng")
        S_NH3_g  = m.addVars(T2, lb=0.0, name="S_NH3_g")
        S_NH3_ng = m.addVars(T2, lb=0.0, name="S_NH3_ng")
        S_MEOH_g = m.addVars(T2, lb=0.0, name="S_MEOH_g")
        S_MEOH_ng = m.addVars(T2, lb=0.0, name="S_MEOH_ng")

        # Sales
        H_sell_g = m.addVars(T, lb=0.0, name="H_sell_g")
        H_sell_ng = m.addVars(T, lb=0.0, name="H_sell_ng")
        NH3_sell_g = m.addVars(T, lb=0.0, name="NH3_sell_g")
        NH3_sell_ng = m.addVars(T, lb=0.0, name="NH3_sell_ng")
        MEOH_sell_g = m.addVars(T, lb=0.0, name="MEOH_sell_g")
        MEOH_sell_ng = m.addVars(T, lb=0.0, name="MEOH_sell_ng")

        # H2 to units
        H_to_NH3_g = m.addVars(T, lb=0.0, name="H_to_NH3_g")
        H_to_NH3_ng = m.addVars(T, lb=0.0, name="H_to_NH3_ng")
        H_to_MEOH_g = m.addVars(T, lb=0.0, name="H_to_MEOH_g")
        H_to_MEOH_ng = m.addVars(T, lb=0.0, name="H_to_MEOH_ng")

        # Unmet
        u_H2   = m.addVars(T, lb=0.0, name="u_H2")
        u_NH3  = m.addVars(T, lb=0.0, name="u_NH3")
        u_MEOH = m.addVars(T, lb=0.0, name="u_MEOH")

        # ---------------- Constraints ----------------
        for t in T:
            m.addConstr(P_pv_cur[t] <= pv_av[t], name=f"ub_pv_cur[{t}]")
            m.addConstr(P_wd_cur[t] <= wd_av[t], name=f"ub_wd_cur[{t}]")

        for t in T:
            for k in K_sinks:
                m.addConstr(P["pv",  k, t] <= pv_av[t],      name=f"ub_P_pv_{k}[{t}]")
                m.addConstr(P["wd",  k, t] <= wd_av[t],      name=f"ub_P_wd_{k}[{t}]")
                m.addConstr(P["grid",k, t] <= P_GRID_MAX_MW, name=f"ub_P_grid_{k}[{t}]")
                m.addConstr(P["bat", k, t] <= P_BAT_MAX_MW,  name=f"ub_P_bat_{k}[{t}]")

        for t in T:
            m.addConstr(gp.quicksum(P["pv",k,t] for k in K_sinks) + P_pv_cur[t] == pv_av[t], name=f"pv_balance[{t}]")
            m.addConstr(gp.quicksum(P["wd",k,t] for k in K_sinks) + P_wd_cur[t] == wd_av[t], name=f"wd_balance[{t}]")

            m.addConstr(gp.quicksum(P["grid",k,t] for k in K_sinks) == P_grid[t], name=f"grid_alloc[{t}]")
            m.addConstr(gp.quicksum(P["bat",k,t] for k in K_sinks) == P_dis[t], name=f"bat_alloc[{t}]")

            m.addConstr(gp.quicksum(P[s,"bch",t] for s in S_sources) == P_ch[t], name=f"bch_sink[{t}]")
            m.addConstr(P["grid","bch",t] == 0.0, name=f"no_grid_to_bch[{t}]")
            m.addConstr(P["bat","bch",t]  == 0.0, name=f"no_bat_to_bch[{t}]")
            m.addConstr(P["pv","bch",t] + P["wd","bch",t] == P_ch[t], name=f"bch_from_res[{t}]")

            m.addSOS(GRB.SOS_TYPE1, [P_ch[t], P_dis[t]])
            m.addConstr(P_dis[t] <= ETA_DIS * (SOC[t] - SOC_MIN) * E_BAT_MAX_MWH, name=f"ub_Pdis_soc[{t}]")
            m.addConstr(P_ch[t]  <= (SOC_MAX - SOC[t]) * E_BAT_MAX_MWH / ETA_CH, name=f"ub_Pch_soc[{t}]")

            for k in K_sinks:
                m.addConstr(Pk_ng[k,t] == P["grid",k,t] + P["bat",k,t], name=f"Pk_ng_def[{k},{t}]")
                m.addConstr(Pk_g[k,t]  == gp.quicksum(P[s,k,t] for s in S_res), name=f"Pk_g_def[{k},{t}]")

        m.addConstr(SOC[0] == float(carry["soc"]), name="SOC_init_carry")
        for t in T:
            m.addConstr(
                SOC[t+1] == SOC[t] + (ETA_CH*P_ch[t] - (1.0/ETA_DIS)*P_dis[t]) / E_BAT_MAX_MWH,
                name=f"SOC_dyn[{t}]"
            )

        m.addConstrs((y_elz_m[i,t] >= y_elz_m[i+1,t] for i in range(N_stack-1) for t in T), name="elz_module_order")

        # Electrolyzer
        for t in T:
            m.addConstr(P_elz_g[t]  == Pk_g["elz",t],  name=f"P_elz_g_link[{t}]")
            m.addConstr(P_elz_ng[t] == Pk_ng["elz",t], name=f"P_elz_ng_link[{t}]")
            m.addConstr(P_elz_g[t]  == gp.quicksum(P_elz_m_g[i,t] for i in I_elz), name=f"P_elz_g_sum_mod[{t}]")
            m.addConstr(P_elz_ng[t] == gp.quicksum(P_elz_m_ng[i,t] for i in I_elz), name=f"P_elz_ng_sum_mod[{t}]")
            m.addConstr(P_elz_tot[t] == P_elz_g[t] + P_elz_ng[t], name=f"P_elz_tot_sum[{t}]")

            m.addConstr(n_elz_on[t] == gp.quicksum(y_elz_m[i,t] for i in I_elz), name=f"n_elz_on_def[{t}]")
            m.addConstr(n_elz_on[t] >= y_elz[t], name=f"y_elz_lb[{t}]")
            m.addConstr(n_elz_on[t] <= N_stack * y_elz[t], name=f"y_elz_ub[{t}]")
            m.addConstr(su_elz[t] == gp.quicksum(su_elz_m[i,t] for i in I_elz), name=f"su_elz_sum[{t}]")

            m.addConstr(H_prod[t]    == gp.quicksum(H_elz_m[i,t] for i in I_elz), name=f"H_prod_sum_mod[{t}]")
            m.addConstr(H_prod_g[t]  == gp.quicksum(H_elz_m_g[i,t] for i in I_elz), name=f"H_prod_g_sum_mod[{t}]")
            m.addConstr(H_prod_ng[t] == gp.quicksum(H_elz_m_ng[i,t] for i in I_elz), name=f"H_prod_ng_sum_mod[{t}]")
            m.addConstr(H_prod[t] == H_prod_g[t] + H_prod_ng[t], name=f"H_prod_split[{t}]")
            m.addConstr(W_elz[t] == omega_H2O * H_prod[t], name=f"W_elz[{t}]")

            for i in range(N_stack - 1):
                m.addConstr(b_elz_g[i,t] >= b_elz_g[i+1,t], name=f"elz_bg_order[{i},{t}]")

            for i in I_elz:
                m.addConstr(P_elz_m_tot[i,t] == P_elz_m_g[i,t] + P_elz_m_ng[i,t], name=f"P_elz_m_tot_def[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] <= P_ELZ_MAX * y_elz_m[i,t], name=f"elz_m_max[{i},{t}]")
                m.addConstr(P_elz_m_tot[i,t] >= P_ELZ_MIN * y_elz_m[i,t], name=f"elz_m_min[{i},{t}]")
                m.addConstr(b_elz_g[i,t] <= y_elz_m[i,t], name=f"b_elz_g_gate[{i},{t}]")
                m.addConstr(P_elz_m_g[i,t]  <= P_ELZ_MAX * b_elz_g[i,t], name=f"P_elz_m_g_cap[{i},{t}]")
                m.addConstr(P_elz_m_ng[i,t] <= P_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"P_elz_m_ng_cap[{i},{t}]")

                m.addConstr(H_elz_m[i,t] == H_elz_m_g[i,t] + H_elz_m_ng[i,t], name=f"H_elz_m_split[{i},{t}]")

                m.addQConstr(
                    H_elz_m[i,t] <= A_elz*(P_elz_m_tot[i,t]*P_elz_m_tot[i,t]) + B_elz*P_elz_m_tot[i,t] + C_elz*y_elz_m[i,t],
                    name=f"elz_m_quad[{i},{t}]"
                )
                m.addConstr(H_elz_m[i,t] >= Q1*P_elz_m_tot[i,t] + Q0*y_elz_m[i,t], name=f"elz_m_lin[{i},{t}]")

                m.addConstr(H_elz_m_g[i,t] <= H_ELZ_MAX * b_elz_g[i,t], name=f"H_elz_m_g_ub1[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] <= H_elz_m[i,t], name=f"H_elz_m_g_ub2[{i},{t}]")
                m.addConstr(H_elz_m_g[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*(1 - b_elz_g[i,t]), name=f"H_elz_m_g_lb[{i},{t}]")

                m.addConstr(H_elz_m_ng[i,t] <= H_ELZ_MAX * (y_elz_m[i,t] - b_elz_g[i,t]), name=f"H_elz_m_ng_ub1[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] <= H_elz_m[i,t], name=f"H_elz_m_ng_ub2[{i},{t}]")
                m.addConstr(H_elz_m_ng[i,t] >= H_elz_m[i,t] - H_ELZ_MAX*b_elz_g[i,t], name=f"H_elz_m_ng_lb[{i},{t}]")

                if t >= 1:
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_elz_m[i,t-1], name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_elz_m[i,t-1], name=f"su_elz_m3[{i},{t}]")
                else:
                    y_prev_i = int(carry["y_elz_prev_mod"][i])
                    m.addConstr(su_elz_m[i,t] <= y_elz_m[i,t], name=f"su_elz_m1[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] <= 1 - y_prev_i, name=f"su_elz_m2[{i},{t}]")
                    m.addConstr(su_elz_m[i,t] >= y_elz_m[i,t] - y_prev_i, name=f"su_elz_m3[{i},{t}]")

        # NH3
        for t in T:
            m.addConstr(Q_nh3[t] == Q_nh3_g[t] + Q_nh3_ng[t], name=f"Q_nh3_sum[{t}]")
            m.addConstr(Q_nh3[t] <= Q_NH3_CAP * y_nh3[t], name=f"nh3_max[{t}]")
            m.addConstr(Q_nh3[t] >= f_NH3_min * Q_NH3_CAP * y_nh3[t], name=f"nh3_min[{t}]")

            if t >= 1:
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_nh3[t-1])
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_nh3[t-1])
            else:
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(su_nh3[t] <= y_nh3[t])
                m.addConstr(su_nh3[t] <= 1 - y_prev)
                m.addConstr(su_nh3[t] >= y_nh3[t] - y_prev)

            m.addConstr(
                Q_nh3[t] <= f_NH3_min * Q_NH3_CAP + (Q_NH3_CAP - f_NH3_min * Q_NH3_CAP) * (1 - su_nh3[t]),
                name=f"nh3_start_minload[{t}]"
            )

            ramp_nh3 = RR_NH3 * Q_NH3_CAP
            if t == 0:
                Q_prev = float(carry["Q_nh3_prev"])
                y_prev = int(carry["y_nh3_prev"])
                m.addConstr(Q_nh3[t] - Q_prev <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rup0[{t}]")
                m.addConstr(Q_prev - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_prev), name=f"nh3_rdn0[{t}]")
            else:
                m.addConstr(Q_nh3[t] - Q_nh3[t-1] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rup[{t}]")
                m.addConstr(Q_nh3[t-1] - Q_nh3[t] <= ramp_nh3 + (Q_NH3_CAP - ramp_nh3) * (2 - y_nh3[t] - y_nh3[t-1]), name=f"nh3_rdn[{t}]")

            m.addConstr(P_nh3_g[t]  == (SEC_HB*Q_nh3_g[t]  + SEC_ASU*(mN2_to_NH3*Q_nh3_g[t]))  / 1000.0, name=f"P_nh3_g_map[{t}]")
            m.addConstr(P_nh3_ng[t] == (SEC_HB*Q_nh3_ng[t] + SEC_ASU*(mN2_to_NH3*Q_nh3_ng[t])) / 1000.0, name=f"P_nh3_ng_map[{t}]")
            m.addConstr(P_nh3_g[t]  == Pk_g["NH3",t],  name=f"P_nh3_g_link[{t}]")
            m.addConstr(P_nh3_ng[t] == Pk_ng["NH3",t], name=f"P_nh3_ng_link[{t}]")
            m.addConstr(H_to_NH3_g[t]  == mH2_to_NH3 * Q_nh3_g[t],  name=f"H_to_NH3_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] == mH2_to_NH3 * Q_nh3_ng[t], name=f"H_to_NH3_ng[{t}]")

        # MeOH
        for t in T:
            m.addConstr(Q_meoh[t] == Q_meoh_g[t] + Q_meoh_ng[t], name=f"Q_meoh_sum[{t}]")
            m.addConstr(Q_meoh[t] <= Q_MEOH_CAP * y_meoh[t], name=f"meoh_max[{t}]")
            m.addConstr(Q_meoh[t] >= f_MEOH_min * Q_MEOH_CAP * y_meoh[t], name=f"meoh_min[{t}]")

            if t >= 1:
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_meoh[t-1])
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_meoh[t-1])
            else:
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(su_meoh[t] <= y_meoh[t])
                m.addConstr(su_meoh[t] <= 1 - y_prev)
                m.addConstr(su_meoh[t] >= y_meoh[t] - y_prev)

            m.addConstr(
                Q_meoh[t] <= f_MEOH_min * Q_MEOH_CAP + (Q_MEOH_CAP - f_MEOH_min * Q_MEOH_CAP) * (1 - su_meoh[t]),
                name=f"meoh_start_minload[{t}]"
            )

            ramp_meoh = R_MEOH
            if t == 0:
                Q_prev = float(carry["Q_meoh_prev"])
                y_prev = int(carry["y_meoh_prev"])
                m.addConstr(Q_meoh[t] - Q_prev <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rup0[{t}]")
                m.addConstr(Q_prev - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_prev), name=f"meoh_rdn0[{t}]")
            else:
                m.addConstr(Q_meoh[t] - Q_meoh[t-1] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rup[{t}]")
                m.addConstr(Q_meoh[t-1] - Q_meoh[t] <= ramp_meoh + (Q_MEOH_CAP - ramp_meoh) * (2 - y_meoh[t] - y_meoh[t-1]), name=f"meoh_rdn[{t}]")

            m.addConstr(P_meoh_g[t]  == (SEC_MEOH * Q_meoh_g[t])  / 1000.0, name=f"P_meoh_g_map[{t}]")
            m.addConstr(P_meoh_ng[t] == (SEC_MEOH * Q_meoh_ng[t]) / 1000.0, name=f"P_meoh_ng_map[{t}]")
            m.addConstr(P_meoh_g[t]  == Pk_g["MeOH",t],  name=f"P_meoh_g_link[{t}]")
            m.addConstr(P_meoh_ng[t] == Pk_ng["MeOH",t], name=f"P_meoh_ng_link[{t}]")
            m.addConstr(CO2_buy[t] == mCO2_to_MEOH * Q_meoh[t], name=f"CO2_buy[{t}]")
            m.addConstr(H_to_MEOH_g[t]  == mH2_to_MEOH * Q_meoh_g[t],  name=f"H_to_MEOH_g[{t}]")
            m.addConstr(H_to_MEOH_ng[t] == mH2_to_MEOH * Q_meoh_ng[t], name=f"H_to_MEOH_ng[{t}]")

        # Cracker
        for t in T:
            m.addConstr(Q_crk_in[t] == Q_crk_in_g[t] + Q_crk_in_ng[t], name=f"Q_crk_sum[{t}]")
            m.addConstr(Q_crk_in[t] <= Q_CRK_CAP * y_crk[t], name=f"crk_max[{t}]")
            m.addConstr(Q_crk_in[t] >= f_CRK_min * Q_CRK_CAP * y_crk[t], name=f"crk_min[{t}]")

            if t >= 1:
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_crk[t-1])
                m.addConstr(su_crk[t] >= y_crk[t] - y_crk[t-1])
            else:
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(su_crk[t] <= y_crk[t])
                m.addConstr(su_crk[t] <= 1 - y_prev)
                m.addConstr(su_crk[t] >= y_crk[t] - y_prev)

            m.addConstr(
                Q_crk_in[t] <= f_CRK_min * Q_CRK_CAP + (Q_CRK_CAP - f_CRK_min * Q_CRK_CAP) * (1 - su_crk[t]),
                name=f"crk_start_minload[{t}]"
            )

            ramp_crk = R_CRK
            if t == 0:
                Q_prev = float(carry["Q_crk_prev"])
                y_prev = int(carry["y_crk_prev"])
                m.addConstr(Q_crk_in[t] - Q_prev <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rup0[{t}]")
                m.addConstr(Q_prev - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_prev), name=f"crk_rdn0[{t}]")
            else:
                m.addConstr(Q_crk_in[t] - Q_crk_in[t-1] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rup[{t}]")
                m.addConstr(Q_crk_in[t-1] - Q_crk_in[t] <= ramp_crk + (Q_CRK_CAP - ramp_crk) * (2 - y_crk[t] - y_crk[t-1]), name=f"crk_rdn[{t}]")

            m.addConstr(H_crk_out_g[t]  == kH2_out * Q_crk_in_g[t],  name=f"Hcrk_out_g[{t}]")
            m.addConstr(H_crk_out_ng[t] == kH2_out * Q_crk_in_ng[t], name=f"Hcrk_out_ng[{t}]")
            m.addConstr(NH3_rec_g[t]    == kNH3_rec * Q_crk_in_g[t],  name=f"NH3rec_g[{t}]")
            m.addConstr(NH3_rec_ng[t]   == kNH3_rec * Q_crk_in_ng[t], name=f"NH3rec_ng[{t}]")
            m.addConstr(H_crk_fuel_g[t]  == kH2_fuel * Q_crk_in_g[t],  name=f"Hcrk_fuel_g[{t}]")
            m.addConstr(H_crk_fuel_ng[t] == kH2_fuel * Q_crk_in_ng[t], name=f"Hcrk_fuel_ng[{t}]")
            m.addConstr(Steam_g[t]  == kSteam * Q_crk_in_g[t],  name=f"Steam_g[{t}]")
            m.addConstr(Steam_ng[t] == kSteam * Q_crk_in_ng[t], name=f"Steam_ng[{t}]")
            m.addConstr(W_steam[t]  == omega_steam * (Steam_g[t] + Steam_ng[t]), name=f"W_steam[{t}]")
            m.addConstr(P_crk_g[t]  == (SEC_CRK_E*Q_crk_in_g[t]  + eSteam*Steam_g[t])  / 1000.0, name=f"P_crk_g_map[{t}]")
            m.addConstr(P_crk_ng[t] == (SEC_CRK_E*Q_crk_in_ng[t] + eSteam*Steam_ng[t]) / 1000.0, name=f"P_crk_ng_map[{t}]")
            m.addConstr(P_crk_g[t]  == Pk_g["crk",t],  name=f"P_crk_g_link[{t}]")
            m.addConstr(P_crk_ng[t] == Pk_ng["crk",t], name=f"P_crk_ng_link[{t}]")

        # Tanks / demand
        m.addConstr(S_H2_g[0]   == float(carry["S_H2_g"]), name="S_H2_g_init")
        m.addConstr(S_H2_ng[0]  == float(carry["S_H2_ng"]), name="S_H2_ng_init")
        m.addConstr(S_NH3_g[0]  == float(carry["S_NH3_g"]), name="S_NH3_g_init")
        m.addConstr(S_NH3_ng[0] == float(carry["S_NH3_ng"]), name="S_NH3_ng_init")
        m.addConstr(S_MEOH_g[0] == float(carry["S_MEOH_g"]), name="S_MEOH_g_init")
        m.addConstr(S_MEOH_ng[0] == float(carry["S_MEOH_ng"]), name="S_MEOH_ng_init")

        for tt in T2:
            m.addConstr(S_H2_g[tt] + S_H2_ng[tt] <= S_H2_CAP, name=f"S_H2_cap[{tt}]")
            m.addConstr(S_NH3_g[tt] + S_NH3_ng[tt] <= S_NH3_CAP, name=f"S_NH3_cap[{tt}]")
            m.addConstr(S_MEOH_g[tt] + S_MEOH_ng[tt] <= S_MEOH_CAP, name=f"S_MEOH_cap[{tt}]")

        m.addConstr(S_H2_g[24] + S_H2_ng[24] >= ALPHA_TANK_FILL * S_H2_CAP, name="eod_hard_H2_frac")
        m.addConstr(S_NH3_g[24] + S_NH3_ng[24] >= ALPHA_TANK_FILL * S_NH3_CAP, name="eod_hard_NH3_frac")
        m.addConstr(S_MEOH_g[24] + S_MEOH_ng[24] >= ALPHA_TANK_FILL * S_MEOH_CAP, name="eod_hard_MeOH_frac")

        for t in T:
            m.addConstr(H_sell_g[t] + H_sell_ng[t] + u_H2[t] == dem_h2[t], name=f"dem_H2[{t}]")
            m.addConstr(NH3_sell_g[t] + NH3_sell_ng[t] + u_NH3[t] == dem_nh3[t], name=f"dem_NH3[{t}]")
            m.addConstr(MEOH_sell_g[t] + MEOH_sell_ng[t] + u_MEOH[t] == dem_meoh[t], name=f"dem_MEOH[{t}]")

            m.addConstr(
                S_H2_g[t+1] == S_H2_g[t] + H_prod_g[t] + H_crk_out_g[t] - H_to_NH3_g[t] - H_to_MEOH_g[t] - H_sell_g[t] - H_crk_fuel_g[t],
                name=f"S_H2_g_dyn[{t}]"
            )
            m.addConstr(
                S_H2_ng[t+1] == S_H2_ng[t] + H_prod_ng[t] + H_crk_out_ng[t] - H_to_NH3_ng[t] - H_to_MEOH_ng[t] - H_sell_ng[t] - H_crk_fuel_ng[t],
                name=f"S_H2_ng_dyn[{t}]"
            )
            m.addConstr(
                S_NH3_g[t+1] == S_NH3_g[t] + Q_nh3_g[t] + NH3_rec_g[t] - NH3_sell_g[t] - Q_crk_in_g[t],
                name=f"S_NH3_g_dyn[{t}]"
            )
            m.addConstr(
                S_NH3_ng[t+1] == S_NH3_ng[t] + Q_nh3_ng[t] + NH3_rec_ng[t] - NH3_sell_ng[t] - Q_crk_in_ng[t],
                name=f"S_NH3_ng_dyn[{t}]"
            )
            m.addConstr(
                S_MEOH_g[t+1] == S_MEOH_g[t] + Q_meoh_g[t] - MEOH_sell_g[t],
                name=f"S_MEOH_g_dyn[{t}]"
            )
            m.addConstr(
                S_MEOH_ng[t+1] == S_MEOH_ng[t] + Q_meoh_ng[t] - MEOH_sell_ng[t],
                name=f"S_MEOH_ng_dyn[{t}]"
            )

            m.addConstr(H_to_NH3_g[t] + H_to_MEOH_g[t] + H_sell_g[t] + H_crk_fuel_g[t] <= S_H2_g[t], name=f"dir_H2_g[{t}]")
            m.addConstr(H_to_NH3_ng[t] + H_to_MEOH_ng[t] + H_sell_ng[t] + H_crk_fuel_ng[t] <= S_H2_ng[t], name=f"dir_H2_ng[{t}]")
            m.addConstr(NH3_sell_g[t] + Q_crk_in_g[t] <= S_NH3_g[t], name=f"dir_NH3_g[{t}]")
            m.addConstr(NH3_sell_ng[t] + Q_crk_in_ng[t] <= S_NH3_ng[t], name=f"dir_NH3_ng[{t}]")
            m.addConstr(MEOH_sell_g[t] <= S_MEOH_g[t], name=f"dir_MEOH_g[{t}]")
            m.addConstr(MEOH_sell_ng[t] <= S_MEOH_ng[t], name=f"dir_MEOH_ng[{t}]")

        # ---------------- Emissions cap expression ----------------
        if EMISSIONS_KEY == "aef":
            grid_emis_cap_expr = gp.quicksum(1000.0 * P_grid[t] * float(aef_k[t]) for t in T)
        elif EMISSIONS_KEY == "mef_upramp":
            grid_emis_cap_expr = gp.quicksum(1000.0 * P_grid[t] * float(mef_upramp_k[t]) for t in T)
        elif EMISSIONS_KEY == "mef_soft":
            grid_emis_cap_expr = gp.quicksum(1000.0 * P_grid[t] * float(mef_soft_k[t]) for t in T)
        else:
            raise ValueError("EMISSIONS_KEY must be one of: 'aef', 'mef_upramp', 'mef_soft'")

        direct_emis_cap_expr = gp.quicksum(float(de_NH3) * Q_nh3[t] + float(de_MEOH) * Q_meoh[t] for t in T)
        total_emis_cap_expr = grid_emis_cap_expr + direct_emis_cap_expr
        m.addConstr(total_emis_cap_expr <= float(EMIS_CAP_DAY_KG), name="M2_emissions_cap_total_kg")

        # ---------------- Objectives ----------------
        unmet_total = gp.quicksum(u_H2[t] + u_NH3[t] + u_MEOH[t] for t in T)

        cost_total = gp.LinExpr()
        for t in T:
            cost_total += pr_eur[t] * P_grid[t]
            cost_total += c_H2O * (W_elz[t] + W_steam[t])
            cost_total += c_CO2 * CO2_buy[t]
            cost_total += C_su_elz * gp.quicksum(su_elz_m[i,t] for i in I_elz)
            cost_total += C_su_NH3 * su_nh3[t]
            cost_total += C_su_MEOH * su_meoh[t]
            cost_total += C_su_CRK * su_crk[t]

        m.setObjectiveN(unmet_total, index=0, priority=2, reltol=1e-4, abstol=1e-6, name="OBJ1_min_unmet")
        m.setObjectiveN(cost_total,  index=1, priority=1, reltol=1e-4, abstol=1e-6, name="OBJ2_min_cost")

        # ---------------- Solve ----------------
        m.optimize()

        status = int(safe_get_attr(m, "Status", -1))
        runtime = float(safe_get_attr(m, "Runtime", np.nan))
        nodecnt = float(safe_get_attr(m, "NodeCount", np.nan))
        solcnt  = int(safe_get_attr(m, "SolCount", -1))

        pass0 = get_objpass_metrics(m, 0)
        pass1 = get_objpass_metrics(m, 1)

        mip_gap = pass1["pass_mipgap"]
        obj_val = pass1["pass_objval"]
        obj_bnd = pass1["pass_objbound"]

        itercnt = safe_get_attr(m, "IterCount", np.nan)
        bariter = safe_get_attr(m, "BarIterCount", np.nan)
        work = safe_get_attr(m, "Work", np.nan)

        obj1_val = np.nan
        obj2_val = np.nan
        try:
            obj1_val = float(unmet_total.getValue())
        except Exception:
            pass
        try:
            obj2_val = float(cost_total.getValue())
        except Exception:
            pass

        is_optimal = (status == GRB.OPTIMAL)
        hit_time_limit = (status == GRB.TIME_LIMIT)
        has_incumbent = (solcnt is not None) and (solcnt >= 1)
        is_within_target_gap = bool(has_incumbent) and (mip_gap is not None) and (not np.isnan(mip_gap)) and (float(mip_gap) <= float(MIPGAP) + 1e-12)
        is_acceptable = bool(is_optimal or is_within_target_gap)

        solver_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "solver_status": status,
            "solver_status_name": status_name(status),

            "m2_emissions_key": str(EMISSIONS_KEY),
            "m2_alpha_used_day": (float(ALPHA_USED_DAY) if (ALPHA_USED_DAY is not None and not np.isnan(ALPHA_USED_DAY)) else np.nan),
            "m2_emis_cap_day_kg": float(EMIS_CAP_DAY_KG),

            "runtime_seconds": runtime,
            "time_limit_s": TIME_LIMIT_S,
            "mipgap_target": float(MIPGAP),
            "mip_gap": pass1["pass_mipgap"],
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),
            "obj_val": float(obj_val) if not np.isnan(obj_val) else np.nan,
            "obj_bound": float(obj_bnd) if not np.isnan(obj_bnd) else np.nan,

            "obj0_pass_status": pass0["pass_status"],
            "obj0_pass_status_name": pass0["pass_status_name"],
            "obj0_pass_runtime_s": pass0["pass_runtime"],
            "obj0_pass_objval": pass0["pass_objval"],
            "obj0_pass_objbound": pass0["pass_objbound"],
            "obj0_pass_mipgap": pass0["pass_mipgap"],

            "obj1_pass_status": pass1["pass_status"],
            "obj1_pass_status_name": pass1["pass_status_name"],
            "obj1_pass_runtime_s": pass1["pass_runtime"],
            "obj1_pass_objval": pass1["pass_objval"],
            "obj1_pass_objbound": pass1["pass_objbound"],
            "obj1_pass_mipgap": pass1["pass_mipgap"],

            "node_count": nodecnt,
            "sol_count": solcnt,
            "iter_count": float(itercnt) if not np.isnan(itercnt) else np.nan,
            "bar_iter_count": float(bariter) if not np.isnan(bariter) else np.nan,
            "work": float(work) if not np.isnan(work) else np.nan,
            "obj1_unmet_value": float(obj1_val) if not np.isnan(obj1_val) else np.nan,
            "obj2_cost_value": float(obj2_val) if not np.isnan(obj2_val) else np.nan,
        })

        if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            try:
                m.computeIIS()
                iis_path = day_dir / f"{MODEL_TAG}_day{d:03d}_iis.ilp"
                m.write(str(iis_path))
            except Exception:
                pass
            raise RuntimeError(f"Day {d}: solver status {status} (not optimal/time_limit).")

        # ---------------- Extract hourly results ----------------
        def val(v):
            return float(v.X)

        rows = []
        unmet_rows = []

        for t in T:
            ts = day_hours[t]
            pv_used = pv_av[t] - val(P_pv_cur[t])
            wd_used = wd_av[t] - val(P_wd_cur[t])
            grid_mw = val(P_grid[t])
            res_used_mw = pv_used + wd_used
            res_curt_mw = val(P_pv_cur[t]) + val(P_wd_cur[t])

            P_elz = val(P_elz_g[t]) + val(P_elz_ng[t])
            P_nh3 = val(P_nh3_g[t]) + val(P_nh3_ng[t])
            P_meo = val(P_meoh_g[t]) + val(P_meoh_ng[t])
            P_crk = val(P_crk_g[t]) + val(P_crk_ng[t])

            H_total = val(H_prod[t])
            H_g = val(H_prod_g[t])
            H_ng = val(H_prod_ng[t])

            nh3_total = val(Q_nh3[t])
            nh3_g = val(Q_nh3_g[t])
            nh3_ng = val(Q_nh3_ng[t])

            me_total = val(Q_meoh[t])
            me_g = val(Q_meoh_g[t])
            me_ng = val(Q_meoh_ng[t])

            h2_sell_g = val(H_sell_g[t])
            h2_sell_ng = val(H_sell_ng[t])
            nh3_sell_g = val(NH3_sell_g[t])
            nh3_sell_ng = val(NH3_sell_ng[t])
            meoh_sell_g = val(MEOH_sell_g[t])
            meoh_sell_ng = val(MEOH_sell_ng[t])

            u1 = val(u_H2[t]); u2 = val(u_NH3[t]); u3 = val(u_MEOH[t])

            def share(g, ng):
                tot = g + ng
                return (g / tot) if tot > 1e-9 else np.nan

            H2_min_to_NH3  = mH2_to_NH3 * f_NH3_min * Q_NH3_CAP * int(round(val(y_nh3[t])))
            H2_min_to_MeOH = mH2_to_MEOH * f_MEOH_min * Q_MEOH_CAP * int(round(val(y_meoh[t])))
            H2_min_to_CRK  = kH2_fuel * f_CRK_min * Q_CRK_CAP * int(round(val(y_crk[t])))
            H2_min_committed = H2_min_to_NH3 + H2_min_to_MeOH + H2_min_to_CRK
            H2_tank_start = val(S_H2_g[t]) + val(S_H2_ng[t])
            H2_free_margin = H2_tank_start - H2_min_committed

            elz_mod = {}
            for i in I_elz:
                elz_mod[f"elz_m{i}_y"]     = int(round(val(y_elz_m[i,t])))
                elz_mod[f"elz_m{i}_su"]    = int(round(val(su_elz_m[i,t])))
                elz_mod[f"elz_m{i}_b_g"]   = int(round(val(b_elz_g[i,t])))
                elz_mod[f"elz_m{i}_P_tot"] = float(val(P_elz_m_tot[i,t]))
                elz_mod[f"elz_m{i}_P_g"]   = float(val(P_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_P_ng"]  = float(val(P_elz_m_ng[i,t]))
                elz_mod[f"elz_m{i}_H_tot"] = float(val(H_elz_m[i,t]))
                elz_mod[f"elz_m{i}_H_g"]   = float(val(H_elz_m_g[i,t]))
                elz_mod[f"elz_m{i}_H_ng"]  = float(val(H_elz_m_ng[i,t]))

            rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "H2_tank_start_kg": H2_tank_start,
                "H2_min_committed_kgph": H2_min_committed,
                "H2_free_margin_kg": H2_free_margin,
                "pv_avail_mw": pv_av[t],
                "wind_avail_mw": wd_av[t],
                "pv_curt_mw": val(P_pv_cur[t]),
                "wind_curt_mw": val(P_wd_cur[t]),
                "res_used_mw": res_used_mw,
                "res_curtailed_mw": res_curt_mw,
                "grid_import_mw": grid_mw,
                "da_price_eur_per_mwh": pr_eur[t],
                "bat_ch_mw": val(P_ch[t]),
                "bat_dis_mw": val(P_dis[t]),
                "battery_soc": val(SOC[t]),
                "battery_soc_next": val(SOC[t+1]),
                "elz_grid_mw": val(P_elz_ng[t]),
                "nh3_grid_mw": val(P_nh3_ng[t]),
                "meoh_grid_mw": val(P_meoh_ng[t]),
                "crk_grid_mw": val(P_crk_ng[t]),
                "P_elz_mw": P_elz,
                "P_nh3_mw": P_nh3,
                "P_meoh_mw": P_meo,
                "P_crk_mw": P_crk,
                "elz_power_green_share": share(val(P_elz_g[t]), val(P_elz_ng[t])),
                "nh3_power_green_share": share(val(P_nh3_g[t]), val(P_nh3_ng[t])),
                "meoh_power_green_share": share(val(P_meoh_g[t]), val(P_meoh_ng[t])),
                "crk_power_green_share": share(val(P_crk_g[t]), val(P_crk_ng[t])),
                "H2_prod_kgph": H_total,
                "H2_prod_g_kgph": H_g,
                "H2_prod_ng_kgph": H_ng,
                "NH3_prod_kgph": nh3_total,
                "NH3_prod_g_kgph": nh3_g,
                "NH3_prod_ng_kgph": nh3_ng,
                "MeOH_prod_kgph": me_total,
                "MeOH_prod_g_kgph": me_g,
                "MeOH_prod_ng_kgph": me_ng,
                "crk_nh3_in_kgph": val(Q_crk_in[t]),
                "crk_h2_out_kgph": val(H_crk_out_g[t]) + val(H_crk_out_ng[t]),
                "crk_h2_fuel_kgph": val(H_crk_fuel_g[t]) + val(H_crk_fuel_ng[t]),
                "steam_kgph": val(Steam_g[t]) + val(Steam_ng[t]),
                "water_elz_m3ph": val(W_elz[t]),
                "water_steam_m3ph": val(W_steam[t]),
                "co2_buy_kgph": val(CO2_buy[t]),

                "H2_sell_g_kgph": h2_sell_g,
                "H2_sell_ng_kgph": h2_sell_ng,
                "H2_sell_kgph": h2_sell_g + h2_sell_ng,

                "NH3_sell_g_kgph": nh3_sell_g,
                "NH3_sell_ng_kgph": nh3_sell_ng,
                "NH3_sell_kgph": nh3_sell_g + nh3_sell_ng,

                "MeOH_sell_g_kgph": meoh_sell_g,
                "MeOH_sell_ng_kgph": meoh_sell_ng,
                "MeOH_sell_kgph": meoh_sell_g + meoh_sell_ng,

                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "tank_H2_g_kg": val(S_H2_g[t]),
                "tank_H2_ng_kg": val(S_H2_ng[t]),
                "tank_NH3_g_kg": val(S_NH3_g[t]),
                "tank_NH3_ng_kg": val(S_NH3_ng[t]),
                "tank_MeOH_g_kg": val(S_MEOH_g[t]),
                "tank_MeOH_ng_kg": val(S_MEOH_ng[t]),
                "b_ch": int(val(P_ch[t]) > 1e-6),

                "n_elz_on": int(round(val(n_elz_on[t]))),
                "y_elz": int(round(val(y_elz[t]))),
                "su_elz": int(round(val(su_elz[t]))),
                "y_nh3": int(round(val(y_nh3[t]))),
                "su_nh3": int(round(val(su_nh3[t]))),
                "y_meoh": int(round(val(y_meoh[t]))),
                "su_meoh": int(round(val(su_meoh[t]))),
                "y_crk": int(round(val(y_crk[t]))),
                "su_crk": int(round(val(su_crk[t]))),
                **existing_solution_exports(t, locals()),
                **elz_mod,
            })

            unmet_rows.append({
                "run_id": RUN_ID,
                "model_tag": MODEL_TAG,
                "scenario_tag": SCENARIO_TAG,
                "demand_tag": DEMAND_TAG,
                "datetime_utc": ts.isoformat(),
                "day_of_year": d,
                "hour_in_day": t,
                "unmet_h2_kgph": u1,
                "unmet_nh3_kgph": u2,
                "unmet_meoh_kgph": u3,
                "da_price_eur_per_mwh": pr_eur[t],
                "grid_import_mw": grid_mw,
                "res_curtailed_mw": res_curt_mw,
            })

        day_hourly = pd.DataFrame(rows)


        day_hourly_all = day_hourly.copy()
        day_hourly_all["day_id"] = int(d)
        all_hourly_frames.append(day_hourly_all)

        # ---------------- Daily accounting ----------------
        grid_cost_eur   = float(np.sum([pr_eur[t] * val(P_grid[t]) for t in T]))
        water_cost_eur  = float(c_H2O * np.sum([val(W_elz[t]) + val(W_steam[t]) for t in T]))
        co2_cost_eur    = float(c_CO2 * np.sum([val(CO2_buy[t]) for t in T]))
        startup_elz_eur = float(C_su_elz * np.sum([val(su_elz[t]) for t in T]))
        startup_nh3_eur = float(C_su_NH3 * np.sum([val(su_nh3[t]) for t in T]))
        startup_meoh_eur = float(C_su_MEOH * np.sum([val(su_meoh[t]) for t in T]))
        startup_crk_eur = float(C_su_CRK * np.sum([val(su_crk[t]) for t in T]))

        total_unmet_kg = float(np.sum(day_hourly["unmet_h2_kgph"].values + day_hourly["unmet_nh3_kgph"].values + day_hourly["unmet_meoh_kgph"].values))
        total_cost = float(obj2_val) if not np.isnan(obj2_val) else float(grid_cost_eur + water_cost_eur + co2_cost_eur + startup_elz_eur + startup_nh3_eur + startup_meoh_eur + startup_crk_eur)

        grid_emis_aef = float(np.sum([val(P_grid[t]) * 1000.0 * aef_k[t] for t in T]))
        grid_emis_mef_upramp = float(np.sum([val(P_grid[t]) * 1000.0 * mef_upramp_k[t] for t in T]))
        grid_emis_mef_soft = float(np.sum([val(P_grid[t]) * 1000.0 * mef_soft_k[t] for t in T]))

        direct_emis_nh3 = float(np.sum([de_NH3 * val(Q_nh3[t]) for t in T]))
        direct_emis_meoh = float(np.sum([de_MEOH * val(Q_meoh[t]) for t in T]))
        direct_emis_total = direct_emis_nh3 + direct_emis_meoh

        if EMISSIONS_KEY == "aef":
            realized_emis_cap_basis = grid_emis_aef + direct_emis_total
        elif EMISSIONS_KEY == "mef_upramp":
            realized_emis_cap_basis = grid_emis_mef_upramp + direct_emis_total
        elif EMISSIONS_KEY == "mef_soft":
            realized_emis_cap_basis = grid_emis_mef_soft + direct_emis_total
        else:
            raise ValueError("EMISSIONS_KEY must be one of: 'aef', 'mef_upramp', 'mef_soft'")

        demand_H2_kg_day   = float(np.sum(dem_h2))
        demand_NH3_kg_day  = float(np.sum(dem_nh3))
        demand_MeOH_kg_day = float(np.sum(dem_meoh))

        elz_kwh_day  = float(np.sum(day_hourly["P_elz_mw"].values * 1000.0))
        nh3_kwh_day  = float(np.sum(day_hourly["P_nh3_mw"].values * 1000.0))
        meoh_kwh_day = float(np.sum(day_hourly["P_meoh_mw"].values * 1000.0))
        crk_kwh_day  = float(np.sum(day_hourly["P_crk_mw"].values * 1000.0))
        units_kwh_day = elz_kwh_day + nh3_kwh_day + meoh_kwh_day + crk_kwh_day

        res_gen_kwh = float(np.sum((pv_av + wd_av) * 1000.0))
        res_used_kwh = float(np.sum((pv_av - day_hourly["pv_curt_mw"].values + wd_av - day_hourly["wind_curt_mw"].values) * 1000.0))
        res_curt_kwh = float(np.sum((day_hourly["pv_curt_mw"].values + day_hourly["wind_curt_mw"].values) * 1000.0))
        grid_import_kwh = float(np.sum(day_hourly["grid_import_mw"].values * 1000.0))
        grid_share_percent = 100.0 * grid_import_kwh / max(grid_import_kwh + res_used_kwh, 1e-9)

        H2_g_sum = float(np.sum(day_hourly["H2_prod_g_kgph"].values))
        H2_ng_sum = float(np.sum(day_hourly["H2_prod_ng_kgph"].values))
        NH3_g_sum = float(np.sum(day_hourly["NH3_prod_g_kgph"].values))
        NH3_ng_sum = float(np.sum(day_hourly["NH3_prod_ng_kgph"].values))
        Me_g_sum = float(np.sum(day_hourly["MeOH_prod_g_kgph"].values))
        Me_ng_sum = float(np.sum(day_hourly["MeOH_prod_ng_kgph"].values))

        def green_share(sum_g, sum_ng):
            tot = sum_g + sum_ng
            return (sum_g / tot) if tot > 1e-9 else np.nan

        daily_rows.append({
            "run_id": RUN_ID,
            "model_tag": MODEL_TAG,
            "scenario_tag": SCENARIO_TAG,
            "demand_tag": DEMAND_TAG,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "has_incumbent": bool(has_incumbent),
            "is_optimal": bool(is_optimal),
            "hit_time_limit": bool(hit_time_limit),
            "mip_gap": float(mip_gap) if mip_gap is not None else np.nan,
            "is_within_target_gap": bool(is_within_target_gap),
            "is_acceptable": bool(is_acceptable),

            "total_cost_eur": total_cost,
            "obj1_total_unmet_kg": total_unmet_kg,
            "obj2_operating_cost_eur": total_cost,

            "res_generation_kWh": res_gen_kwh,
            "res_used_kWh": res_used_kwh,
            "res_curtailed_kWh": res_curt_kwh,
            "grid_import_kWh": grid_import_kwh,
            "grid_share_percent": grid_share_percent,

            "battery_soc_avg": float(np.mean(day_hourly["battery_soc"].values)),
            "tank_H2_level_avg_kg": float(np.mean(day_hourly["tank_H2_g_kg"].values + day_hourly["tank_H2_ng_kg"].values)),
            "tank_NH3_level_avg_kg": float(np.mean(day_hourly["tank_NH3_g_kg"].values + day_hourly["tank_NH3_ng_kg"].values)),
            "tank_MeOH_level_avg_kg": float(np.mean(day_hourly["tank_MeOH_g_kg"].values + day_hourly["tank_MeOH_ng_kg"].values)),

            "H2_prod_green_share": green_share(H2_g_sum, H2_ng_sum),
            "NH3_prod_green_share": green_share(NH3_g_sum, NH3_ng_sum),
            "MeOH_prod_green_share": green_share(Me_g_sum, Me_ng_sum),

            # grid emissions only
            "emissions_grid_aef_kg": grid_emis_aef,
            "emissions_grid_mef_upramp_kg": grid_emis_mef_upramp,
            "emissions_grid_mef_soft_kg": grid_emis_mef_soft,

            # direct emissions separate
            "emissions_direct_nh3_kg": direct_emis_nh3,
            "emissions_direct_meoh_kg": direct_emis_meoh,
            "emissions_direct_total_kg": direct_emis_total,

            "unmet_H2_kg": float(np.sum(day_hourly["unmet_h2_kgph"].values)),
            "unmet_NH3_kg": float(np.sum(day_hourly["unmet_nh3_kgph"].values)),
            "unmet_MeOH_kg": float(np.sum(day_hourly["unmet_meoh_kgph"].values)),

            "demand_H2_kg_day": demand_H2_kg_day,
            "demand_NH3_kg_day": demand_NH3_kg_day,
            "demand_MeOH_kg_day": demand_MeOH_kg_day,

            "elz_consumption_kWh_day": elz_kwh_day,
            "nh3_consumption_kWh_day": nh3_kwh_day,
            "meoh_consumption_kWh_day": meoh_kwh_day,
            "crk_consumption_kWh_day": crk_kwh_day,
            "units_total_consumption_kWh_day": units_kwh_day,

            # sales accounting
            "H2_sell_g_kg_day": float(np.sum(day_hourly["H2_sell_g_kgph"].values)),
            "H2_sell_ng_kg_day": float(np.sum(day_hourly["H2_sell_ng_kgph"].values)),
            "H2_sell_total_kg_day": float(np.sum(day_hourly["H2_sell_kgph"].values)),

            "NH3_sell_g_kg_day": float(np.sum(day_hourly["NH3_sell_g_kgph"].values)),
            "NH3_sell_ng_kg_day": float(np.sum(day_hourly["NH3_sell_ng_kgph"].values)),
            "NH3_sell_total_kg_day": float(np.sum(day_hourly["NH3_sell_kgph"].values)),

            "MeOH_sell_g_kg_day": float(np.sum(day_hourly["MeOH_sell_g_kgph"].values)),
            "MeOH_sell_ng_kg_day": float(np.sum(day_hourly["MeOH_sell_ng_kgph"].values)),
            "MeOH_sell_total_kg_day": float(np.sum(day_hourly["MeOH_sell_kgph"].values)),

            "eod_H2_total_kg": float(val(S_H2_g[24]) + val(S_H2_ng[24])),
            "eod_NH3_total_kg": float(val(S_NH3_g[24]) + val(S_NH3_ng[24])),
            "eod_MeOH_total_kg": float(val(S_MEOH_g[24]) + val(S_MEOH_ng[24])),
            "eod_req_H2_kg": float(ALPHA_TANK_FILL * S_H2_CAP),
            "eod_req_NH3_kg": float(ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_req_MeOH_kg": float(ALPHA_TANK_FILL * S_MEOH_CAP),
            "eod_margin_H2_kg": float((val(S_H2_g[24]) + val(S_H2_ng[24])) - ALPHA_TANK_FILL * S_H2_CAP),
            "eod_margin_NH3_kg": float((val(S_NH3_g[24]) + val(S_NH3_ng[24])) - ALPHA_TANK_FILL * S_NH3_CAP),
            "eod_margin_MeOH_kg": float((val(S_MEOH_g[24]) + val(S_MEOH_ng[24])) - ALPHA_TANK_FILL * S_MEOH_CAP),

            "cost_grid_eur": grid_cost_eur,
            "cost_water_eur": water_cost_eur,
            "cost_co2_eur": co2_cost_eur,
            "cost_startup_elz_eur": startup_elz_eur,
            "cost_startup_nh3_eur": startup_nh3_eur,
            "cost_startup_meoh_eur": startup_meoh_eur,
            "cost_startup_crk_eur": startup_crk_eur,
            "m2_cap_repair_flag": int(IS_CAP_REPAIR_DAY),
            "m2_emissions_key": str(EMISSIONS_KEY),
            "m2_alpha_used_day": (float(ALPHA_USED_DAY) if (ALPHA_USED_DAY is not None and not np.isnan(ALPHA_USED_DAY)) else np.nan),
            "m2_emis_cap_day_kg": float(EMIS_CAP_DAY_KG),
            "m2_realized_total_emis_cap_basis_kg": float(realized_emis_cap_basis),
            "m2_cap_slack_kg": float(EMIS_CAP_DAY_KG - realized_emis_cap_basis),
        })

        # ---------------- Carry-over update ----------------
        carry["soc"] = val(SOC[24])
        carry["S_H2_g"] = val(S_H2_g[24]); carry["S_H2_ng"] = val(S_H2_ng[24])
        carry["S_NH3_g"] = val(S_NH3_g[24]); carry["S_NH3_ng"] = val(S_NH3_ng[24])
        carry["S_MEOH_g"] = val(S_MEOH_g[24]); carry["S_MEOH_ng"] = val(S_MEOH_ng[24])
        carry["y_elz_prev_mod"] = [int(round(val(y_elz_m[i,23]))) for i in I_elz]
        carry["y_nh3_prev"]  = int(round(val(y_nh3[23])))
        carry["y_meoh_prev"] = int(round(val(y_meoh[23])))
        carry["y_crk_prev"]  = int(round(val(y_crk[23])))
        carry["Q_nh3_prev"]  = float(val(Q_nh3[23]))
        carry["Q_meoh_prev"] = float(val(Q_meoh[23]))
        carry["Q_crk_prev"]  = float(val(Q_crk_in[23]))

        state_rows.append({
            "run_id": RUN_ID,
            "day_of_year": d,
            "date_utc": str(day_hours[0].date()),
            "SOC_end": carry["soc"],
            "S_H2_g_end": carry["S_H2_g"],
            "S_H2_ng_end": carry["S_H2_ng"],
            "S_NH3_g_end": carry["S_NH3_g"],
            "S_NH3_ng_end": carry["S_NH3_ng"],
            "S_MEOH_g_end": carry["S_MEOH_g"],
            "S_MEOH_ng_end": carry["S_MEOH_ng"],
            "y_elz_prev_mod_end": carry["y_elz_prev_mod"],
            "y_nh3_end": carry["y_nh3_prev"],
            "y_meoh_end": carry["y_meoh_prev"],
            "y_crk_end": carry["y_crk_prev"],
            "Q_nh3_end": carry["Q_nh3_prev"],
            "Q_meoh_end": carry["Q_meoh_prev"],
            "Q_crk_end": carry["Q_crk_prev"],
        })

        eps = 1e-6
        sim_cd = float(np.sum((day_hourly["bat_ch_mw"].values > eps) & (day_hourly["bat_dis_mw"].values > eps)))
        soc_viol = float(np.sum((day_hourly["battery_soc"].values < SOC_MIN - 1e-6) | (day_hourly["battery_soc"].values > SOC_MAX + 1e-6)))
        unmet_any = float(np.sum((day_hourly["unmet_h2_kgph"].values > eps) | (day_hourly["unmet_nh3_kgph"].values > eps) | (day_hourly["unmet_meoh_kgph"].values > eps)))


    # ---------------- End-of-run aggregation ----------------
    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        raise RuntimeError("No days were solved (daily_df is empty).")

    solver_df = pd.DataFrame(solver_rows)
    state_df  = pd.DataFrame(state_rows)

    if len(all_hourly_frames) > 0:
        hourly_all_df = pd.concat(all_hourly_frames, ignore_index=True)
        hourly_all_df.to_csv(OUT_POST / "hourly_timeseries_all_days.csv", index=False)
    else:
        hourly_all_df = pd.DataFrame()

    daily_df.to_csv(OUT_POST / "daily_summary.csv", index=False)
    solver_df.to_csv(OUT_POST / "solver_log_daily.csv", index=False)
    state_df.to_csv(OUT_POST / "state_trace.csv", index=False)

    annual_summary = {
        "run_id": RUN_ID,
        "model_tag": MODEL_TAG,
        "scenario_tag": SCENARIO_TAG,
        "demand_tag": DEMAND_TAG,
        "start_day_of_year": START_DAY_OF_YEAR,
        "n_days_run": int(len(daily_df)),
        "m2_emissions_key": str(EMISSIONS_KEY),
        "m2_alpha_configured": float(ALPHA),
        "m2_emis_cap_day_kg_sum": float(daily_df["m2_emis_cap_day_kg"].sum()),
        "m2_alpha_used_day_mean": float(daily_df["m2_alpha_used_day"].mean()),
        "m2_alpha_used_day_count": int(daily_df["m2_alpha_used_day"].notna().sum()),
        "m2_cap_repair_days_count": int(daily_df["m2_cap_repair_flag"].sum()),
        "total_cost_eur": float(daily_df["total_cost_eur"].sum()),

        "total_grid_emissions_aef_kg": float(daily_df["emissions_grid_aef_kg"].sum()),
        "total_grid_emissions_mef_upramp_kg": float(daily_df["emissions_grid_mef_upramp_kg"].sum()),
        "total_grid_emissions_mef_soft_kg": float(daily_df["emissions_grid_mef_soft_kg"].sum()),

        "total_direct_emissions_nh3_kg": float(daily_df["emissions_direct_nh3_kg"].sum()),
        "total_direct_emissions_meoh_kg": float(daily_df["emissions_direct_meoh_kg"].sum()),
        "total_direct_emissions_kg": float(daily_df["emissions_direct_total_kg"].sum()),

        "total_grid_import_kWh": float(daily_df["grid_import_kWh"].sum()),
        "total_res_used_kWh": float(daily_df["res_used_kWh"].sum()),
        "total_res_curtailed_kWh": float(daily_df["res_curtailed_kWh"].sum()),
        "avg_grid_share_percent": float(daily_df["grid_share_percent"].mean()),

        "total_unmet_H2_kg": float(daily_df["unmet_H2_kg"].sum()),
        "total_unmet_NH3_kg": float(daily_df["unmet_NH3_kg"].sum()),
        "total_unmet_MeOH_kg": float(daily_df["unmet_MeOH_kg"].sum()),

        "total_H2_sell_g_kg": float(daily_df["H2_sell_g_kg_day"].sum()),
        "total_H2_sell_ng_kg": float(daily_df["H2_sell_ng_kg_day"].sum()),
        "total_H2_sell_kg": float(daily_df["H2_sell_total_kg_day"].sum()),

        "total_NH3_sell_g_kg": float(daily_df["NH3_sell_g_kg_day"].sum()),
        "total_NH3_sell_ng_kg": float(daily_df["NH3_sell_ng_kg_day"].sum()),
        "total_NH3_sell_kg": float(daily_df["NH3_sell_total_kg_day"].sum()),

        "total_MeOH_sell_g_kg": float(daily_df["MeOH_sell_g_kg_day"].sum()),
        "total_MeOH_sell_ng_kg": float(daily_df["MeOH_sell_ng_kg_day"].sum()),
        "total_MeOH_sell_kg": float(daily_df["MeOH_sell_total_kg_day"].sum()),

    }
    pd.DataFrame([annual_summary]).to_csv(OUT_POST / "annual_summary.csv", index=False)


    return OUT_RUN


SIGNAL_FILES = {
    "aef": ("grid_aef_hourly_2024.csv", "grid_aef_predicted_kg_per_kwh"),
    "mef_soft": ("grid_mef_soft_hourly_2024.csv", "grid_mef_soft_kg_per_kwh"),
    "mef_upramp": ("grid_mef_upramp_hourly_2024.csv", "grid_mef_upramp_pred_kg_per_kwh"),
}


def run_optimization(model, model_ready, output, signal="aef", alpha=0.5,
                     days=366, bounds=None, solver_output=True):
    """Run one selected model from 1 January, carrying states between days.

    Returns the new run folder, whose post/ subfolder holds five CSV files.
    M0 is independent of the chosen signal; all three emissions are evaluated.
    M2 needs daily M0/M1 references consistent with the inputs and plant settings.
    """
    model = str(model).upper()
    if model not in {"M0", "M1", "M2"}:
        raise ValueError("model must be M0, M1 or M2")
    if signal not in SIGNAL_FILES:
        raise ValueError("signal must be aef, mef_soft or mef_upramp")
    if not isinstance(days, int) or not 1 <= days <= 366:
        raise ValueError("days must be an integer from 1 to 366")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    model_ready = Path(model_ready).resolve()
    output = Path(output).resolve()
    if "Csv files of input and output" in output.parts:
        raise ValueError("Write new runs outside the supplied input/result folders")
    config = {"model_ready": model_ready, "output": output, "days": days,
              "signal": signal, "alpha": alpha, "time_limit": 350,
              "solver_output": int(solver_output)}
    if model == "M2":
        bounds = Path(bounds or model_ready / "M2_S1_emission_bounds_input_file.csv").resolve()
        reference = pd.read_csv(bounds)
        selected = reference.loc[reference.scenario.eq("S1")].set_index("day_of_year")
        needed = [f"{signal}_lb_kg", f"{signal}_up_kg"]
        if selected.index.duplicated().any() or not set(range(1, days + 1)).issubset(selected.index):
            raise ValueError("M2 reference file must cover each requested day once")
        if not np.isfinite(selected.loc[range(1, days + 1), needed].to_numpy(float)).all():
            raise ValueError("M2 reference values for the selected signal must be finite")
        config["bounds"] = bounds
    return {"M0": _run_m0, "M1": _run_m1, "M2": _run_m2}[model](config)


def make_emission_bounds(m0_daily, m1_daily_by_signal, output):
    """Construct daily grid-plus-direct M0/M1 references for one or more signals.

    References retain their own sequential state histories. Endpoints are not
    reordered; the original M2 cap function handles reversed/negative cases.
    Signals absent from the mapping have blank columns and cannot be selected
    in run_optimization until corresponding references have been supplied.
    """
    def read_daily(path):
        d = pd.read_csv(path)
        if d.day_of_year.duplicated().any():
            raise ValueError("Reference daily summaries must have one row per day")
        return d.set_index("day_of_year").sort_index()
    base = read_daily(m0_daily)
    result = pd.DataFrame(index=base.index)
    result["scenario"] = "S1"
    for signal in SIGNAL_FILES:
        result[f"{signal}_lb_kg"] = np.nan
        result[f"{signal}_up_kg"] = np.nan
    for signal, file in m1_daily_by_signal.items():
        if signal not in SIGNAL_FILES:
            raise ValueError(f"Unknown signal: {signal}")
        ref = read_daily(file)
        if not ref.index.equals(base.index):
            raise ValueError("M0 and M1 references must cover identical days")
        col = f"emissions_grid_{signal}_kg"
        values = [ref[col] + ref.emissions_direct_total_kg,
                  base[col] + base.emissions_direct_total_kg]
        if not np.isfinite(np.column_stack(values)).all():
            raise ValueError("Reference emissions must be finite")
        result[f"{signal}_lb_kg"], result[f"{signal}_up_kg"] = values
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Choose a new reference filename: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.reset_index().to_csv(path, index=False)
    return path
