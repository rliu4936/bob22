from multiprocessing import Process, Manager
import time
import gc
from Data import generate_chart as gc
from Data import equity_data as eqd

chart_type = "bar"
ws = 20
freq = "week"
ma_lags = [ws]
vb = True

def get_us_ret_by_year(year, semaphore):
    with semaphore:
        print(f"[{year}] Loading us_ret...")
        us_ret = eqd.processed_US_data()
        mask = us_ret.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
        df = us_ret[mask].copy()
        del us_ret
        print(f"[{year}] Finished loading and cleared us_ret")
        return df

def process_year(year, semaphore):
    print(f"Start processing year {year}")
    process_data = get_us_ret_by_year(year, semaphore)
    dgp_obj = gc.GenerateStockData(
        "USA", year, ws, freq,
        chart_freq=1, ma_lags=ma_lags, volume_bar=vb,
        need_adjust_price=True, allow_tqdm=False,
        chart_type=chart_type, process_data=process_data
    )
    dgp_obj.save_annual_data()
    dgp_obj.save_annual_ts_data()
    print(f"[{year}] Done.")

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

if __name__ == "__main__":
    year_list = list(range(1992, 1995))  # Replace with any list of years
    with Manager() as manager:
        semaphore = manager.Semaphore(1)
        for year_chunk in chunked(year_list, 6):
            processes = []
            for year in year_chunk:
                p = Process(target=process_year, args=(year, semaphore))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()