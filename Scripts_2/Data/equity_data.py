from __future__ import print_function, division
import os
import os.path as op
import numpy as np
import pandas as pd
import time
from Data import dgp_config as dcf
from tqdm import tqdm


def get_processed_US_data_by_year(year, df):
    #df = processed_US_data()

    df = df[
        df.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
    ].copy()
    return df


def get_spy_freq_rets(freq):
    assert freq in ["week", "month", "quarter"]
    spy = pd.read_csv(
        os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.csv"),
        parse_dates=["date"],
    )
    spy.rename(columns={"date": "Date"}, inplace=True)
    spy = spy.set_index("Date")
    return spy


def get_period_end_dates(period):
    assert period in ["week", "month", "quarter"]
    spy = get_spy_freq_rets(period)
    return spy.index


# def getProcessedData(country = "USA")
def processed_US_data():
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)

        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finish loading processed data in {(time.time() - since) / 60:.2f} min")

        return df.copy()

    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "us_920101-200731.csv")
    print("Reading raw data from {}".format(raw_us_data_path))

    since = time.time()
    df = pd.read_csv(
        raw_us_data_path,
        parse_dates=["date"],
        dtype={
            "PERMNO": str,
            "BIDLO": np.float64,
            "ASKHI": np.float64,
            "PRC": np.float64,
            "VOL": np.float64,
            "SHROUT": np.float64,
            "OPENPRC": np.float64,
            "RET": object,
            "EXCHCD": np.float64,
        },
        compression="gzip",
        header=0,
    )
    print(f"finish reading data in {(time.time() - since) / 60:.2f} s")
    df = process_raw_data_helper(df)

    df.reset_index().to_feather(processed_us_data_path)
    return df.copy()


def process_raw_data_helper(df):
    # ---------- basic cleaning / renaming ----------
    df = df.rename(
        columns={
            "date":  "Date",
            "PERMNO":"StockID",
            "BIDLO": "Low",
            "ASKHI": "High",
            "PRC":   "Close",
            "VOL":   "Vol",
            "SHROUT":"Shares",
            "OPENPRC":"Open",
            "RET":   "Ret",
        }
    )
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"]     = df["Ret"].astype(str)

    df = df.replace({
        "Close": {0: np.nan}, "Open":  {0: np.nan},
        "High":  {0: np.nan}, "Low":   {0: np.nan},
        "Ret":   {"C": np.nan, "B": np.nan, "A": np.nan, ".": np.nan},
        "Vol":   {0: np.nan, -99: np.nan},
    })

    if "Shares" not in df.columns:
        df["Shares"] = 0

    df["Ret"] = df["Ret"].astype(np.float64)
    df = df.dropna(subset=["Ret"])

    df[["Close", "Open", "High", "Low", "Vol", "Shares"]] = df[
        ["Close", "Open", "High", "Low", "Vol", "Shares"]
    ].abs()

    # sort & index -----------------------------------------------------
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    # ---------- BACKWARD adjustment per stock ------------------------
    def apply_backward_adjust(stock_df):
        # stock_df index: (Date, StockID); ascending by Date
        stock_df = stock_df.reset_index()

        if stock_df.empty or stock_df.loc[len(stock_df)-1, "Close"] in (0, np.nan):
            return pd.DataFrame([], columns=stock_df.columns).set_index(["Date", "StockID"])

        adjusted = stock_df.copy()
        last_idx = len(adjusted) - 1

        # Set last day's AdjClose equal to its raw close (anchor)
        adjusted.at[last_idx, "AdjClose"] = adjusted.at[last_idx, "Close"]

        # Walk *backwards* from second-last row to first
        for i in range(last_idx - 1, -1, -1):
            next_adj_close = adjusted.at[i + 1, "AdjClose"]
            ret_next       = adjusted.at[i + 1, "Ret"]           # use next day's return
            adj_close      = next_adj_close / (1 + ret_next)     # back-adjust step

            raw_close = adjusted.at[i, "Close"]
            ratio     = adj_close / raw_close if raw_close else np.nan

            adjusted.at[i, "AdjClose"] = adj_close
            adjusted.at[i, "Open"]  *= ratio
            adjusted.at[i, "High"]  *= ratio
            adjusted.at[i, "Low"]   *= ratio
            adjusted.at[i, "Close"]  = adj_close                 # overwrite Close

        adjusted.set_index(["Date", "StockID"], inplace=True)
        return adjusted

    print("Applying BACKWARD price adjustment...")
    adjusted_parts = []
    for sid, g in tqdm(df.groupby(level="StockID"), desc="Adjusting by StockID"):
        adjusted_parts.append(apply_backward_adjust(g))

    df = pd.concat(adjusted_parts).sort_index()
    return df


def add_derived_features(df):
    df["MarketCap"] = np.abs(df["Close"] * df["Shares"])
    
    df["log_ret"] = np.log(1 + df.Ret)
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )
    
    #
    for freq in ["week", "month", "quarter"]:
        period_end_dates = get_period_end_dates(freq)
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        freq_df["freq_ret"] = freq_df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        print(
            f"Freq: {freq}: {len(freq_df)}/{len(df)} preriod_end_dates from \
                        {period_end_dates[0]}, {period_end_dates[1]},  to {period_end_dates[-1]}"
        )
        df[f"Ret_{freq}"] = freq_df["freq_ret"]
        num_nan = np.sum(pd.isna(df[f"Ret_{freq}"]))
        print(f"df Ret_{freq} {len(df) - num_nan}/{len(df)} not nan")

    for i in [5, 20, 60, 65, 180, 250, 260]:
        print(f"Calculating {i}d return")
        df[f"Ret_{i}d"] = df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )
    return df


def get_period_ret(period, country="USA"):
#
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_ret.pq")
    period_ret = pd.read_parquet(period_ret_path)   # 
    period_ret.set_index(["Date", "StockID"], inplace=True)
    period_ret.sort_index(inplace=True)
    return period_ret




# def getProcessedData(country = "CHN")
def processed_CN_data():
    """
    Load pre-processed China A-share daily data from a Feather file if it exists;
    otherwise read the raw CSV (Wind/CSMAR-style), clean it with
    process_cn_raw_data_helper(), cache it to Feather, and return a copy.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (Date, StockID) and ready for downstream use.
    """
    processed_cn_data_path = op.join(dcf.PROCESSED_DATA_DIR, "cn_ret.feather")

    # ---------- 1. Fast path: already processed ----------
    if op.exists(processed_cn_data_path):
        print(f"Loading processed data from {processed_cn_data_path}")
        since = time.time()
        df = pd.read_feather(processed_cn_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finish loading processed data in {(time.time() - since) / 60:.2f} min")
        return df.copy()

    # ---------- 2. Slow path: read raw file and preprocess ----------
    raw_cn_data_path = op.join(dcf.RAW_DATA_DIR, "CSMAR/cn_93-25.csv")  # adjust name
    print(f"Reading raw data from {raw_cn_data_path}")

    since = time.time()
    df = pd.read_csv(
        raw_cn_data_path,
        parse_dates=["Trddt"],
        dtype={
            "Stkcd":       str,
            "Opnprc":      np.float64,
            "Hiprc":       np.float64,
            "Loprc":       np.float64,
            "Clsprc":      np.float64,
            "Dnshrtrd":    np.float64,
            "Dnvaltrd":    np.float64,
            "Dretwd":      object,
            "Dsmvosd":     np.float64,
            "Markettype":  np.float64,
            "Adjprcwd":    np.float64,      # ← NEW
        },
        header=0,
    )
    print(f"Finish reading data in {(time.time() - since):.2f} s")

    # ---------- 3. Clean & feature-engineer ----------
    df = process_cn_raw_data_helper(df)

    # ---------- 4. Cache to Feather ----------
    df.reset_index().to_feather(processed_cn_data_path)
    return df.copy()

def process_cn_raw_data_helper(df):
    # 1️⃣ rename --------------------------------------------------------
    df = df.rename(
        columns={
            "Trddt": "Date", "Stkcd": "StockID",
            "Loprc": "Low",  "Hiprc": "High",
            "Clsprc": "Close","Opnprc": "Open",
            "Dnshrtrd": "Vol",
            "Dretwd":  "Ret",
        }
    )

    # 2️⃣ keep only essentials (add Markettype & Adjprcwd for scaling) --
    keep_cols = [
        "Date", "StockID", "Low", "High", "Close", "Open",
        "Vol", "Ret", "Dsmvosd", "Markettype", "Adjprcwd"
    ]
    df = df.loc[:, df.columns.intersection(keep_cols)]

    # 3️⃣ basic dtypes --------------------------------------------------
    df["Date"]    = pd.to_datetime(df["Date"])
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"]     = pd.to_numeric(df["Ret"], errors="coerce")

    # 🆕 3b.  create EXCHCD and drop Markettype -------------------------
    df["EXCHCD"]  = pd.to_numeric(df["Markettype"], errors="coerce")
    df = df.drop(columns="Markettype")

    # 🆕 3c.  replace raw OHLC with adjusted OHLC -----------------------
    if "Adjprcwd" in df.columns:
        ratio = df["Adjprcwd"] / df["Close"]
        df["Close"] = df["Adjprcwd"]                      # adjusted Close
        df["Open"]  = df["Open"]  * ratio
        df["High"]  = df["High"]  * ratio
        df["Low"]   = df["Low"]   * ratio
    df = df.drop(columns="Adjprcwd", errors="ignore")

    # 4️⃣ derive Shares -------------------------------------------------
    if "Dsmvosd" in df.columns:
        shares_raw = df["Dsmvosd"] / df["Close"] / ratio if 'ratio' in locals() else \
                     df["Dsmvosd"] / df["Close"]
        df["Shares"] = shares_raw
    else:
        df["Shares"] = 0.0
    df = df.drop(columns="Dsmvosd", errors="ignore")

    # 5️⃣ sentinels 6️⃣ dropna 7️⃣ abs() 8️⃣ index (unchanged) ----------
    df = df.replace({
        "Close": {0: np.nan}, "Open":  {0: np.nan},
        "High":  {0: np.nan}, "Low":   {0: np.nan},
        "Vol":   {0: np.nan, -99: np.nan},
    })
    df = df.dropna(subset=["Ret"])
    num_cols = ["Close", "Open", "High", "Low", "Vol", "Shares"]
    df[num_cols] = df[num_cols].abs()
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    return df