from Data import generate_chart as gc

class GenerateYearStockData:
    def __init__(
        self,
        country,
        year_list,              # List[int] -- multiple years
        window_size,
        freq,
        chart_freq=1,
        ma_lags=None,
        volume_bar=False,
        need_adjust_price=True,
        allow_tqdm=True,
        chart_type="bar",
        us_ret=None,
    ):
        self.country = country
        self.year_list = year_list
        self.window_size = window_size
        self.freq = freq
        self.chart_freq = chart_freq
        self.ma_lags = ma_lags
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.allow_tqdm = allow_tqdm
        self.chart_type = chart_type
        self.us_ret = us_ret

    def generate_all_years(self, image_mode=True, ts_mode=True):
        for year in self.year_list:
            print(f"\n=== Processing Year {year} ===")
            generator = gc.GenerateStockData(
                country=self.country,
                year=year,
                window_size=self.window_size,
                freq=self.freq,
                chart_freq=self.chart_freq,
                ma_lags=self.ma_lags,
                volume_bar=self.volume_bar,
                need_adjust_price=self.need_adjust_price,
                allow_tqdm=self.allow_tqdm,
                chart_type=self.chart_type,
                us_ret=self.us_ret,
            )
            if image_mode:
                generator.save_annual_data()
            if ts_mode:
                generator.save_annual_ts_data()