# -*- coding: utf-8 -*-
import pandas as pd


class CorrelationEngine:
    """
    封装所有相关性分析方法的类。
    """

    def __init__(self):
        print("相关性分析引擎初始化成功。")

    def align_series(self, price_series, sentiment_series):
        """对齐价格和情感两个时间序列。"""
        print("开始对齐价格和情感时间序列...")
        df = pd.concat([price_series, sentiment_series], axis=1)

        # 使用 .ffill() 并重新赋值，以避免警告
        df[sentiment_series.name] = df[sentiment_series.name].ffill()
        df.dropna(inplace=True)

        print("时间序列对齐完成。")
        if df.empty:
            print("警告：对齐后数据为空，请检查价格和评论数据的时间范围是否有重叠。")
            return pd.Series(dtype=float), pd.Series(dtype=float)

        return df[price_series.name], df[sentiment_series.name]

    def calculate_rolling_correlation(self, series1, series2, window=30):
        """计算滚动相关性。"""
        print(f"计算窗口大小为 {window} 的滚动相关性...")
        return series1.rolling(window=window).corr(series2)

    def calculate_lagged_correlation(self, series1, series2, max_lag=10):
        """计算滞后相关性。"""
        print(f"计算最大滞后为 {max_lag} 天的滞后相关性...")
        lags = range(-max_lag, max_lag + 1)
        corrs = [series1.corr(series2.shift(lag)) for lag in lags]
        return pd.Series(corrs, index=lags)
