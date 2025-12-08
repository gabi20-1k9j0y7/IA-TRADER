# src/data.py
# Procesamiento de datos: Indicadores técnicos (Etapa 1) y Opciones Financieras (Etapa 2).

from __future__ import annotations
import os, typing as t, warnings

# Gestión de warnings de librerías externas
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated as an API.*")

import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.stats import linregress
from alpha_vantage.timeseries import TimeSeries
from pandas.tseries.offsets import BDay

# ============================
# Secrets / configuración
# ============================

def _get_secret(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st  # type: ignore
        if "secrets" in dir(st) and name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        pass
    return os.environ.get(name, default)

ALPHAVANTAGE_API_KEY = _get_secret("ALPHAVANTAGE_API_KEY", None)

# Configuración de ventanas temporales
WINDOW_MAX = 1260
PAD_BDAYS  = 90   

# ============================
# Utilidades comunes
# ============================
TECH_COLS = [
    "open","high","low","close","volume",
    "logret_1","overnight_ret","roc_5","rsi_14","macd_hist","atr_14_pct","vol_z_21",
    "dist_sma20","roc_21","dist_sma50","dist_sma200","dist_sma400",
    "bb_width_20_2","sma_slope_20",
    "from_52w_high","from_52w_low",
    "vol_ratio_21_252","vol_ratio_252_1260",
    "trend_ok","drawdown_long","roc_1260","trend_slope_3y","from_ath",
]

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)

def _safe_write_csv(df: pd.DataFrame, path: str, index_label: str = "Date") -> None:
    _ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=True, index_label=index_label, date_format="%d-%m-%y")
    os.replace(tmp, path)

def _normalize_index_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df

# ============================
# Auxiliares
# ============================

def _rolling_slope(x_ndarray: np.ndarray) -> float:
    x = np.asarray(x_ndarray, dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2: return np.nan
    xx = np.arange(m.sum(), dtype=float)
    return linregress(xx, x[m]).slope


def _bb_width_from_df(bbands_df: pd.DataFrame) -> pd.Series:
    if not isinstance(bbands_df, pd.DataFrame) or bbands_df.empty:
        return pd.Series(np.nan, index=[])
    cols = bbands_df.columns
    if "BBB_20_2.0" in cols: return bbands_df["BBB_20_2.0"]
    upper = bbands_df[[c for c in cols if c.startswith("BBU_")]].iloc[:, 0]
    lower = bbands_df[[c for c in cols if c.startswith("BBL_")]].iloc[:, 0]
    mid   = bbands_df[[c for c in cols if c.startswith("BBM_")]].iloc[:, 0]
    return (upper - lower) / mid.replace(0, np.nan)


def _apply_manual_splits(df: pd.DataFrame, splits: list[tuple[str, float]]) -> pd.DataFrame:
    """Ajuste retroactivo de precios por splits."""
    if not splits: return df
    df = df.copy(); df["adj_factor"] = 1.0
    for date_str, factor in sorted(splits, key=lambda x: pd.to_datetime(x[0])):
        d = pd.to_datetime(date_str)
        df.loc[df.index < d, "adj_factor"] *= float(factor)
    for col in ["open","high","low","close"]:
        df[col] = df[col] / df["adj_factor"]
    df["volume"] = df["volume"] * df["adj_factor"]
    return df.drop(columns="adj_factor")

# ============================
# Alpha Vantage (incremental + caché + SMART UPDATE)
# ============================

def _av_get_daily(ticker: str, outputsize: str = "full") -> pd.DataFrame:
    ts = TimeSeries(key=ALPHAVANTAGE_API_KEY or "", output_format="pandas")
    df, _ = ts.get_daily(symbol=ticker, outputsize=outputsize)
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns={
        "1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"
    })
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[["open","high","low","close","volume"]].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return _normalize_index_dates(df)


def fetch_prices_incremental(ticker: str, cache_file: str, years_hist: int = 10) -> pd.DataFrame:
    """
    Descarga y mantiene una caché idempotente.
    [MODIFICADO] Lógica 'Smart Update' para evitar bucles en festivos/intradía.
    """
    cache = pd.DataFrame()
    if os.path.exists(cache_file):
        cache = pd.read_csv(cache_file, index_col=0)
        cache.index = pd.to_datetime(cache.index, format="%d-%m-%y", errors="coerce")
        cache = _normalize_index_dates(cache)

    need_update = True
    if not cache.empty:
        last_dt = cache.index.max()
        today = pd.Timestamp.today().normalize()
        
        # Verificación de datos obsoletos
        if last_dt < today:
            # Lógica de exclusión intradía/festivo
            diff_days = (today - last_dt).days
            current_hour = pd.Timestamp.now().hour
            
            # Evitar llamadas innecesarias a API en horario de mercado abierto
            if diff_days <= 3 and current_hour < 23:
                need_update = False
            else:
                need_update = True
        else:
            need_update = False

    if cache.empty:
        merged = _av_get_daily(ticker, "full")
        if ticker.upper() == "AMZN" and (merged.index < "2022-06-06").any():
            merged = _apply_manual_splits(merged, [("2022-06-06", 20)])
            
    elif need_update:
        fresh = _av_get_daily(ticker, "compact")
        if ticker.upper() == "AMZN" and (fresh.index < "2022-06-06").any():
            fresh = _apply_manual_splits(fresh, [("2022-06-06", 20)])
        merged = pd.concat([cache, fresh]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = cache

    _safe_write_csv(merged, cache_file, index_label="Date")

    cut = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_hist)
    out = merged[merged.index >= cut]
    return _normalize_index_dates(out)

# ============================
# Maestro (DD-MM-YY / DD/MM/YY)
# ============================

def _parse_dates_flex(series: pd.Series) -> pd.Series:
    """Parseo flexible de fechas (múltiples formatos soportados)"""
    s = pd.to_datetime(series, format="%d-%m-%y", errors="coerce")
    m = s.isna()
    if m.any():
        s.loc[m] = pd.to_datetime(series.where(m), format="%d/%m/%y", errors="coerce")
    m = s.isna()
    if m.any():
        s.loc[m] = pd.to_datetime(series.where(m), format="%Y-%m-%d", errors="coerce")
    m = s.isna()
    if m.any():
        s.loc[m] = pd.to_datetime(series.where(m), errors="coerce", dayfirst=True)
    return s


def _read_master_anydate(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    raw = pd.read_csv(path)
    date_col = "Date" if "Date" in raw.columns else ("date" if "date" in raw.columns else None)
    if not date_col: raise ValueError(f"El maestro no tiene columna 'Date'/'date': {path}")
    raw[date_col] = _parse_dates_flex(raw[date_col])
    df = raw.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df

# ====================
# Etapa 1 — Técnicos
# ====================

def build_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = ohlcv.copy(); feats.index.name = "Date"
    feats = feats.replace([np.inf, -np.inf], np.nan)

    close = feats["close"].astype(float)
    high  = feats["high"].astype(float)
    low   = feats["low"].astype(float)
    opn   = feats["open"].astype(float)
    vol   = feats["volume"].astype("float64")  

    feats["logret_1"]      = np.log(close / close.shift(1))
    feats["overnight_ret"] = np.log(opn / close.shift(1))

    feats["roc_5"]  = ta.roc(close, length=5)
    feats["rsi_14"] = ta.rsi(close, length=14)
    _macd = ta.macd(close, fast=12, slow=26, signal=9)
    feats["macd_hist"] = _macd["MACDh_12_26_9"] if isinstance(_macd, pd.DataFrame) and "MACDh_12_26_9" in _macd.columns else np.nan

    atr14 = ta.atr(high, low, close, length=14)
    feats["atr_14_pct"] = (atr14 / close) * 100.0

    # Cálculo optimizado de volatilidad (Z-Score)
    vol_mu  = vol.rolling(21, min_periods=21).apply(lambda x: np.mean(x), raw=True)
    vol_sig = vol.rolling(21, min_periods=21).apply(lambda x: np.std(x, ddof=1), raw=True).replace(0, np.nan)
    feats["vol_z_21"] = (vol - vol_mu) / vol_sig

    sma20  = ta.sma(close, length=20)
    sma50  = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    sma400 = ta.sma(close, length=400)

    feats["dist_sma20"]  = (close / sma20) - 1
    feats["roc_21"]      = ta.roc(close, length=21)
    feats["dist_sma50"]  = (close / sma50) - 1
    feats["dist_sma200"] = (close / sma200) - 1
    feats["dist_sma400"] = (close / sma400) - 1

    _bbands = ta.bbands(close, length=20, std=2)
    feats["bb_width_20_2"] = _bb_width_from_df(_bbands)

    feats["sma_slope_20"] = sma20.rolling(window=5, min_periods=5).apply(_rolling_slope, raw=True)

    roll_max_252 = close.rolling(252, min_periods=252).max()
    roll_min_252 = close.rolling(252, min_periods=252).min()
    feats["from_52w_high"] = (close / roll_max_252) - 1
    feats["from_52w_low"]  = (close / roll_min_252) - 1

    logret   = feats["logret_1"]
    vol_21   = logret.rolling(21,   min_periods=21).std()
    vol_252  = logret.rolling(252,  min_periods=252).std()
    vol_1260 = logret.rolling(1260, min_periods=1260).std()
    feats["vol_ratio_21_252"]   = vol_21 / vol_252
    feats["vol_ratio_252_1260"] = vol_252 / vol_1260

    feats["trend_ok"] = (close > sma200).astype("int8")
    roll_max_1260 = close.rolling(1260, min_periods=1).max()
    feats["drawdown_long"] = ((close - roll_max_1260) / roll_max_1260).rolling(1260, min_periods=1).min()

    feats["roc_1260"]       = ta.roc(close, length=1260)
    feats["trend_slope_3y"] = np.log(close).rolling(756, min_periods=756).apply(_rolling_slope, raw=True)
    feats["from_ath"]       = (close / close.expanding().max()) - 1

    missing = [c for c in TECH_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas técnicas: {missing}")
    return feats[TECH_COLS].copy()

# ============================
# Etapa 1 — Actualización maestro 
# ============================

def update_master_technicals(
    master_csv_path: str,
    ticker: str,
    on_progress: t.Callable[[int, str], None] | None = None,
    write_back: bool = True,
    years_hist: int = 10,
) -> dict:
    def _p(p, m):
        if on_progress: on_progress(int(p), m)

    abs_master = os.path.abspath(master_csv_path)
    master_dir = os.path.dirname(master_csv_path) or "."
    cache_file = os.path.join(master_dir, f"{ticker.upper()}_alphavantage_cache.csv")

    _p(5, "Leyendo maestro…")
    master = _read_master_anydate(master_csv_path)
    prev_last = master.index.max() if not master.empty else None
    before_rows = int(master.shape[0]) if not master.empty else 0

    _p(20, "Actualizando precios (Alpha Vantage)…")
    prices = fetch_prices_incremental(ticker, cache_file=cache_file, years_hist=years_hist)
    if prices.empty: raise RuntimeError("No hay precios descargados de Alpha Vantage.")
    latest_price_date = prices.index.max()

    _p(45, "Calculando indicadores técnicos…")
    if prev_last is not None:
        start_prices = (prev_last - BDay(WINDOW_MAX + PAD_BDAYS)).date()
        prices = prices[prices.index >= pd.Timestamp(start_prices)]
    tech = build_technical_indicators(prices)

    to_add = tech if master.empty else tech[tech.index > prev_last]

    nan_cols_new = []
    if len(to_add):
        last_rows = to_add.tail(5)
        nan_cols_new = [c for c in TECH_COLS if last_rows[c].isna().any()]

    if len(to_add):
        if master.empty:
            master_updated = to_add
        else:
            new_block = pd.DataFrame(index=to_add.index, columns=master.columns, dtype="float64")
            for c in TECH_COLS:
                if c in new_block.columns:
                    new_block[c] = to_add[c]
            master_updated = pd.concat([master, new_block]).sort_index()
    else:
        master_updated = master

    after_rows = int(master_updated.shape[0]) if not master_updated.empty else 0
    added_rows = after_rows - before_rows

    if write_back:
        _p(92, "Guardando maestro…")
        _safe_write_csv(master_updated, master_csv_path, index_label="Date")

    now_last = master_updated.index.max() if not master_updated.empty else None
    _p(100, f"Listo. Añadidas {added_rows} filas.")

    return {
        "added_rows": int(added_rows),
        "first_added": str(to_add.index.min().date()) if len(to_add) else None,
        "last_added": str(to_add.index.max().date()) if len(to_add) else None,
        "last_in_master_before": str(prev_last.date()) if prev_last is not None else None,
        "last_in_master_after": str(now_last.date()) if now_last is not None else None,
        "latest_price": str(latest_price_date.date()),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "master_path_abs": abs_master,
        "cache_path_abs": os.path.abspath(cache_file),
        "tech_nan_cols_in_new": nan_cols_new,
    }

# ============================
# Etapa 2 — Opciones (tail fill + escala + derivadas + FORWARD FILL)
# ============================

def update_master_options_from_csvs(
    master_csv_path: str,
    ticker: str,
    on_progress: t.Callable[[int, str], None] | None = None,
    write_back: bool = True,
) -> dict:
    def _p(p, m):
        if on_progress: on_progress(int(p), m)

    abs_master = os.path.abspath(master_csv_path)
    master_dir = os.path.dirname(master_csv_path) or "."

    # Carga de dataset principal
    _p(5, "Leyendo maestro…")
    master = _read_master_anydate(master_csv_path)
    if master.empty:
        return {"updated_cells": 0, "derived_cells": 0, "reason": "master_empty", "master_path_abs": abs_master}

    # Configuración de mapeo CSV externo -> Columnas
    T = ticker.upper()
    files_map: dict[str, tuple[str, str]] = {
        f"{T}_hv_yz_30.csv":    ("hv_yz_30_ext",    "hv_yz_30"),
        f"{T}_hv_yz_270.csv":   ("hv_yz_270_ext",   "hv_yz_270"),
        f"{T}_iv_mean_30.csv":  ("iv_atm_30_ext",   "iv_atm_30"),
        f"{T}_iv_mean_90.csv":  ("iv_atm_90_ext",   "iv_atm_90"),
        f"{T}_iv_mean_360.csv": ("iv_atm_360_ext",  "iv_atm_360"),
        f"{T}_iv_skew_30.csv":  ("iv_skew_30_ext",  "iv_skew_30"),
        f"{T}_pcr_oi_30.csv":   ("pc_ratio_oi_30_ext",  "pcr_oi_30"),
        f"{T}_pcr_v_30.csv":    ("pc_ratio_vol_30_ext", "pcr_vol_30"),
    }

    for col, _ in files_map.values():
        if col not in master.columns:
            master[col] = np.nan

    # Helpers de procesamiento local
    def _parse_dates_flex_series(df: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_datetime(df[col], format="%d-%m-%y", errors="coerce")
        m = s.isna()
        if m.any(): s.loc[m] = pd.to_datetime(df[col].where(m), format="%d/%m/%y", errors="coerce")
        m = s.isna()
        if m.any(): s.loc[m] = pd.to_datetime(df[col].where(m), format="%Y-%m-%d", errors="coerce")
        m = s.isna()
        if m.any(): s.loc[m] = pd.to_datetime(df[col].where(m), errors="coerce", dayfirst=True)
        return s

    def _read_series(path: str) -> pd.Series:
        df = pd.read_csv(path)
        date_col = None
        for cand in ["Date", "date", "DATE"]:
            if cand in df.columns: date_col = cand; break
        if date_col is None: date_col = df.columns[0]
        parsed = _parse_dates_flex_series(df, date_col)
        df = df.assign(**{date_col: parsed})
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        val_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not val_cols:
            for c in df.columns:
                if c != date_col: df[c] = pd.to_numeric(df[c], errors="coerce")
            val_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not val_cols: raise ValueError("No hay columna numérica en " + os.path.basename(path))
        s = df[val_cols[0]].astype(float).copy()
        s.name = "value"
        return s

    def _maybe_to_decimal(s: pd.Series, key: str) -> pd.Series:
        if key.startswith(("hv_", "iv_")) and ("skew" not in key):
            v = s.dropna()
            if len(v):
                if v.median() > 1.5 or v.max() > 3: return s / 100.0
        return s

    # Relleno incremental de datos (Tail Fill)
    _p(25, "Leyendo CSVs y alineando…")
    found_csvs, missing_csvs, scaled_cols = [], [], []
    updated_cols: list[str] = []
    updated_cells = 0
    touched_dates: set[pd.Timestamp] = set()

    def _tail_nan_index(series: pd.Series) -> list[pd.Timestamp]:
        idxs = []
        for i in range(len(series)-1, -1, -1):
            v = series.iat[i]
            if pd.isna(v): idxs.append(series.index[i])
            else: break
        idxs.reverse()
        return idxs

    for fname, (col, key) in files_map.items():
        fpath = os.path.join(master_dir, fname)
        if not os.path.exists(fpath):
            missing_csvs.append(fname)
            continue
        try:
            s = _read_series(fpath)
            s = _maybe_to_decimal(s, key)
            s = s.reindex(master.index)
            tail_idx = _tail_nan_index(master[col])
            if not tail_idx:
                found_csvs.append(fname)
                continue
            wrote_any = False
            for d in tail_idx:
                val = s.loc[d] if d in s.index else np.nan
                if pd.notna(val) and pd.isna(master.at[d, col]):
                    master.at[d, col] = float(val)
                    updated_cells += 1
                    touched_dates.add(pd.Timestamp(d))
                    wrote_any = True
            if wrote_any and col not in updated_cols:
                updated_cols.append(col)
            found_csvs.append(fname)
            v = s.dropna()
            if key.startswith(("hv_", "iv_")) and ("skew" not in key) and len(v) and (v.median() < 1.0 and v.max() <= 2.0):
                scaled_cols.append(col)
        except Exception:
            found_csvs.append(fname)

    # Forward Fill

    _p(75, "Rellenando huecos de opciones (Forward Fill)...")
    cols_to_fill = [c for c, _ in files_map.values()]
    cols_present = [c for c in cols_to_fill if c in master.columns]
    
    if cols_present:
        master[cols_present] = master[cols_present].ffill()
    # ----------------------------------

    # 4) Cálculo de métricas derivadas
    _p(80, "Calculando derivadas (solo donde falte)…")
    derived_cells = 0
    
    if updated_cells == 0:
        touched_dates.update(master.index[-5:]) # Revisar últimos 5 días por si acaso

    if touched_dates:
        if "iv_ts_ratio_30_360" not in master.columns: master["iv_ts_ratio_30_360"] = np.nan
        if "iv_minus_hv30" not in master.columns: master["iv_minus_hv30"] = np.nan

        for d in sorted(touched_dates):
            if d not in master.index: continue
            
            if pd.isna(master.at[d, "iv_ts_ratio_30_360"]):
                a = master.at[d, "iv_atm_30_ext"]  if "iv_atm_30_ext"  in master.columns else np.nan
                b = master.at[d, "iv_atm_360_ext"] if "iv_atm_360_ext" in master.columns else np.nan
                if pd.notna(a) and pd.notna(b) and b != 0:
                    master.at[d, "iv_ts_ratio_30_360"] = float(a) / float(b)
                    derived_cells += 1
            if pd.isna(master.at[d, "iv_minus_hv30"]):
                a = master.at[d, "iv_atm_30_ext"] if "iv_atm_30_ext" in master.columns else np.nan
                c = master.at[d, "hv_yz_30_ext"]  if "hv_yz_30_ext"  in master.columns else np.nan
                if pd.notna(a) and pd.notna(c):
                    master.at[d, "iv_minus_hv30"] = float(a) - float(c)
                    derived_cells += 1

    # 5) Persistencia de datos
    if write_back:
        _p(95, "Guardando maestro…")
        _safe_write_csv(master, master_csv_path, index_label="Date")

    _p(100, f"Listo. Celdas completadas: {updated_cells}; derivadas: {derived_cells}.")
    return {
        "updated_cells": int(updated_cells),
        "updated_cols": updated_cols,
        "derived_cells": int(derived_cells),
        "touched_dates": [str(pd.Timestamp(d).date()) for d in sorted(touched_dates) if d in master.index],
        "found_csvs": found_csvs,
        "missing_csvs": missing_csvs,
        "scaled_cols": sorted(set(scaled_cols)),
        "master_path_abs": abs_master,
    }
