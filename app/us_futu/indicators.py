from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MRMCMacdConfig:
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


def _ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def _shift_dynamic(series: pd.Series, offsets: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    steps = pd.to_numeric(offsets, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for idx, step in enumerate(steps):
        if np.isnan(step):
            continue
        ref_idx = idx - int(step)
        if 0 <= ref_idx < len(values):
            out[idx] = values[ref_idx]
    return pd.Series(out, index=series.index)


def _extreme_dynamic(series: pd.Series, windows: pd.Series, *, mode: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    spans = pd.to_numeric(windows, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for idx, span in enumerate(spans):
        if np.isnan(span) or span <= 0:
            continue
        window = int(span)
        start = max(0, idx - window + 1)
        section = values[start : idx + 1]
        if len(section) == 0 or np.isnan(section).all():
            continue
        out[idx] = np.nanmin(section) if mode == "min" else np.nanmax(section)
    return pd.Series(out, index=series.index)


def _barslast(condition: pd.Series) -> pd.Series:
    flags = condition.fillna(False).astype(bool).to_numpy()
    out = np.full(len(flags), np.nan, dtype=float)
    last_true = None
    for idx, flag in enumerate(flags):
        if flag:
            last_true = idx
            out[idx] = 0.0
        elif last_true is not None:
            out[idx] = float(idx - last_true)
    return pd.Series(out, index=condition.index)


def _rolling_count(condition: pd.Series, window: int) -> pd.Series:
    return condition.fillna(False).astype(int).rolling(window, min_periods=1).sum()


def build_mrmc_nx_indicators(
    frame: pd.DataFrame,
    config: MRMCMacdConfig | None = None,
) -> pd.DataFrame:
    config = config or MRMCMacdConfig()
    df = frame.sort_values("date").copy()
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])

    diff = _ema(df["close"], config.fast_period) - _ema(df["close"], config.slow_period)
    dea = _ema(diff, config.signal_period)
    macd = (diff - dea) * 2

    n1 = _barslast((macd.shift(1) >= 0) & (macd < 0))
    mm1 = _barslast((macd.shift(1) <= 0) & (macd > 0))

    cc1 = _extreme_dynamic(df["close"], n1 + 1, mode="min")
    cc2 = _shift_dynamic(cc1, mm1 + 1)
    cc3 = _shift_dynamic(cc2, mm1 + 1)
    difl1 = _extreme_dynamic(diff, n1 + 1, mode="min")
    difl2 = _shift_dynamic(difl1, mm1 + 1)
    difl3 = _shift_dynamic(difl2, mm1 + 1)

    ch1 = _extreme_dynamic(df["close"], mm1 + 1, mode="max")
    ch2 = _shift_dynamic(ch1, n1 + 1)
    ch3 = _shift_dynamic(ch2, n1 + 1)
    difh1 = _extreme_dynamic(diff, mm1 + 1, mode="max")
    difh2 = _shift_dynamic(difh1, n1 + 1)
    difh3 = _shift_dynamic(difh2, n1 + 1)

    aaa = (cc1 < cc2) & (difl1 > difl2) & (macd.shift(1) < 0) & (diff < 0)
    bbb = (cc1 < cc3) & (difl1 < difl2) & (difl1 > difl3) & (macd.shift(1) < 0) & (diff < 0)
    ccc = (aaa | bbb) & (diff < 0)
    lll = (~ccc.shift(1).fillna(False)) & ccc

    xxx = (aaa.shift(1).fillna(False) & (difl1 <= difl2) & (diff < dea)) | (
        bbb.shift(1).fillna(False) & (difl1 <= difl3) & (diff < dea)
    )
    jjj = ccc.shift(1).fillna(False) & (diff.shift(1).abs() >= diff.abs() * 1.01)
    blbl = jjj.shift(1).fillna(False) & ccc & (diff.shift(1).abs() * 1.01 <= diff.abs())
    dxdx = (~jjj.shift(1).fillna(False)) & jjj

    ref_jjj_mm1_1 = _shift_dynamic(jjj.astype(float), mm1 + 1).fillna(0).astype(bool)
    ref_jjj_mm1 = _shift_dynamic(jjj.astype(float), mm1).fillna(0).astype(bool)
    djgxx = (
        ((df["close"] < cc2) | (df["close"] < cc1))
        & (ref_jjj_mm1_1 | ref_jjj_mm1)
        & (~lll.shift(1).fillna(False))
        & (_rolling_count(jjj, 24) >= 1)
    )
    djxx = ~(_rolling_count(djgxx.shift(1).fillna(False), 2) >= 1) & djgxx
    dxx = (xxx | djxx) & (~ccc)

    zjdbl = (ch1 > ch2) & (difh1 < difh2) & (macd.shift(1) > 0) & (diff > 0)
    gxdbl = (ch1 > ch3) & (difh1 > difh2) & (difh1 < difh3) & (macd.shift(1) > 0) & (diff > 0)
    dbbl = (zjdbl | gxdbl) & (diff > 0)
    dbl = (~dbbl.shift(1).fillna(False)) & dbbl & (diff > dea)
    dblxs = (zjdbl.shift(1).fillna(False) & (difh1 >= difh2) & (diff > dea)) | (
        gxdbl.shift(1).fillna(False) & (difh1 >= difh3) & (diff > dea)
    )
    dbjg = dbbl.shift(1).fillna(False) & (diff.shift(1) >= diff * 1.01)
    dbjgxc = (~dbjg.shift(1).fillna(False)) & dbjg
    dbjgbl = dbjg.shift(1).fillna(False) & dbbl & (diff.shift(1) * 1.01 <= diff)

    ref_dbjg_n1_1 = _shift_dynamic(dbjg.astype(float), n1 + 1).fillna(0).astype(bool)
    ref_dbjg_n1 = _shift_dynamic(dbjg.astype(float), n1).fillna(0).astype(bool)
    zzzzz = (
        ((df["close"] > ch2) | (df["close"] > ch1))
        & (ref_dbjg_n1_1 | ref_dbjg_n1)
        & (~dbl.shift(1).fillna(False))
        & (_rolling_count(dbjg, 23) >= 1)
    )
    yyyyy = ~(_rolling_count(zzzzz.shift(1).fillna(False), 2) >= 1) & zzzzz
    wwwww = (dblxs | yyyyy) & (~dbbl)

    blue_upper = _ema(df["high"], 24)
    blue_lower = _ema(df["low"], 23)
    yellow_upper = _ema(df["high"], 89)
    yellow_lower = _ema(df["low"], 90)
    blue_mid = (blue_upper + blue_lower) / 2
    yellow_mid = (yellow_upper + yellow_lower) / 2

    result = df.copy()
    result["diff"] = diff
    result["dea"] = dea
    result["macd"] = macd
    result["n1"] = n1
    result["mm1"] = mm1
    result["ccc"] = ccc
    result["lll"] = lll
    result["xxx"] = xxx
    result["jjj"] = jjj
    result["blbl"] = blbl
    result["dxx"] = dxx
    result["dbbl"] = dbbl
    result["dbl"] = dbl
    result["dbjg"] = dbjg
    result["dbjgbl"] = dbjgbl
    result["wwwww"] = wwwww
    result["mrmc_bottom_signal"] = dxdx
    result["mrmc_sell_signal"] = dbjgxc
    result["blue_upper"] = blue_upper
    result["blue_lower"] = blue_lower
    result["yellow_upper"] = yellow_upper
    result["yellow_lower"] = yellow_lower
    result["blue_mid"] = blue_mid
    result["yellow_mid"] = yellow_mid
    result["blue_above_yellow"] = (blue_mid > yellow_mid) & (blue_lower > yellow_lower)
    result["close_above_blue"] = result["close"] > blue_upper
    result["close_below_blue"] = result["close"] < blue_lower
    result["close_above_yellow"] = result["close"] > yellow_upper
    result["close_below_yellow"] = result["close"] < yellow_lower
    return result
