# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TemporalSentimentAnalysis:
    """
    时间序列情感模式分析
    """

    def __init__(self):
        print("时间序列情感分析器初始化完成")

    def intraday_sentiment_pattern(self, df):
        """
        日内情感模式分析 - 分析不同时段的情感特征
        """
        print("开始日内情感模式分析...")

        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        # 定义时段
        def get_time_period(hour):
            if 9 <= hour < 12:
                return '上午'
            elif 12 <= hour < 14:
                return '午间'
            elif 14 <= hour < 16:
                return '下午'
            elif 16 <= hour < 20:
                return '傍晚'
            elif 20 <= hour < 24:
                return '晚间'
            else:
                return '深夜'

        df['time_period'] = df['hour'].apply(get_time_period)

        # 统计各时段情感
        period_sentiment = df.groupby('time_period').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'cleaned_text': lambda x: len(' '.join(x).split())  # 总词数
        }).round(4)

        return period_sentiment

    def weekend_vs_weekday_sentiment(self, df):
        """
        工作日 vs 周末情感对比分析
        """
        print("开始工作日/周末情感对比分析...")

        df['weekday'] = pd.to_datetime(df['timestamp']).dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6])  # 周六周日

        weekday_sentiment = df.groupby('is_weekend').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'cleaned_text': 'count'
        }).round(4)

        weekday_sentiment.index = ['工作日', '周末']

        return weekday_sentiment

    def sentiment_seasonality_analysis(self, daily_sentiment_df):
        """
        情感季节性分析
        """
        print("开始情感季节性分析...")

        df = daily_sentiment_df.copy()
        df.index = pd.to_datetime(df.index)

        # 月度模式
        df['month'] = df.index.month
        monthly_pattern = df.groupby('month')['daily_sentiment_score'].mean()

        # 季度模式
        df['quarter'] = df.index.quarter
        quarterly_pattern = df.groupby('quarter')['daily_sentiment_score'].mean()

        # 年末效应（12月 vs 其他月份）
        year_end_effect = {
            '12月平均情感': df[df['month'] == 12]['daily_sentiment_score'].mean(),
            '其他月份平均情感': df[df['month'] != 12]['daily_sentiment_score'].mean()
        }

        return {
            'monthly': monthly_pattern,
            'quarterly': quarterly_pattern,
            'year_end_effect': year_end_effect
        }

    def sentiment_momentum_analysis(self, daily_sentiment_df, window=5):
        """
        情感动量分析
        """
        print(f"开始情感动量分析（窗口期={window}天）...")

        df = daily_sentiment_df.copy()

        # 计算情感移动平均
        df['sentiment_ma'] = df['daily_sentiment_score'].rolling(window=window).mean()

        # 计算情感动量（当前值与移动平均的差值）
        df['sentiment_momentum'] = (
            df['daily_sentiment_score'] - df['sentiment_ma']
        )

        # 计算情感变化率
        df['sentiment_change_rate'] = df['daily_sentiment_score'].pct_change()

        # 计算情感波动率（滚动标准差）
        df['sentiment_volatility'] = df['daily_sentiment_score'].rolling(window=window).std()

        print("情感动量分析完成")
        return df

class NetworkSentimentAnalysis:
    """
    网络情感传播分析
    """

    def __init__(self):
        print("网络情感传播分析器初始化完成")

    def sentiment_contagion_analysis(self, df, time_window='1H'):
        """
        情感传染分析 - 分析情感在时间窗口内的传播
        """
        print(f"开始情感传染分析（时间窗口：{time_window}）...")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()

        # 按时间窗口分组
        grouped = df.groupby(pd.Grouper(freq=time_window))

        contagion_metrics = []
        for period, group in grouped:
            if len(group) > 1:
                # 计算该时段内情感的一致性
                sentiment_consistency = 1 - (group['sentiment_score'].std() / (group['sentiment_score'].abs().mean() + 1e-6))

                # 计算情感极化程度
                positive_ratio = (group['sentiment_score'] > 0).mean()
                negative_ratio = (group['sentiment_score'] < 0).mean()
                polarization = abs(positive_ratio - negative_ratio)

                contagion_metrics.append({
                    'period': period,
                    'comment_count': len(group),
                    'avg_sentiment': group['sentiment_score'].mean(),
                    'sentiment_consistency': sentiment_consistency,
                    'polarization': polarization
                })

        return pd.DataFrame(contagion_metrics)

    def echo_chamber_detection(self, df):
        """
        回音室效应检测 - 检测同质化讨论
        """
        print("开始回音室效应检测...")

        # 计算文本相似度（简化版本）
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # 按日期分组
        daily_groups = df.groupby(df['timestamp'].dt.date)

        echo_chamber_scores = []
        for date, group in daily_groups:
            if len(group) >= 5:  # 至少5条评论才分析
                texts = group['cleaned_text'].tolist()

                # 计算文本相似度
                vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)

                    # 计算平均相似度（排除对角线）
                    n = len(similarity_matrix)
                    avg_similarity = (similarity_matrix.sum() - n) / (n * (n - 1))

                    # 情感一致性
                    sentiment_std = group['sentiment_score'].std()
                    sentiment_consistency = 1 / (1 + sentiment_std)

                    echo_chamber_scores.append({
                        'date': date,
                        'text_similarity': avg_similarity,
                        'sentiment_consistency': sentiment_consistency,
                        'echo_chamber_score': (avg_similarity + sentiment_consistency) / 2
                    })
                except:
                    continue

        return pd.DataFrame(echo_chamber_scores)

class PredictiveSentimentAnalysis:
    """
    预测性情感分析
    """

    def __init__(self):
        print("预测性情感分析器初始化完成")

    def sentiment_momentum_signals(self, daily_sentiment_df):
        """
        基于情感动量的交易信号，增加动量相关指标
        """
        print("开始生成情感动量信号...")

        df = daily_sentiment_df.copy()

        # 计算多个周期的移动平均
        df['sentiment_ma_3'] = df['daily_sentiment_score'].rolling(3).mean()
        df['sentiment_ma_7'] = df['daily_sentiment_score'].rolling(7).mean()
        df['sentiment_ma_14'] = df['daily_sentiment_score'].rolling(14).mean()

        # 添加动量分析指标
        df['sentiment_ma'] = df['daily_sentiment_score'].rolling(window=5).mean()
        df['sentiment_momentum'] = df['daily_sentiment_score'] - df['sentiment_ma']
        df['sentiment_change_rate'] = df['daily_sentiment_score'].pct_change()
        df['sentiment_volatility'] = df['daily_sentiment_score'].rolling(window=5).std()

        # 生成信号
        conditions = [
            # 强烈看多信号
            (df['daily_sentiment_score'] > df['sentiment_ma_3']) &
            (df['sentiment_ma_3'] > df['sentiment_ma_7']) &
            (df['sentiment_ma_7'] > df['sentiment_ma_14']) &
            (df['daily_sentiment_score'] > 0.3),

            # 温和看多信号
            (df['daily_sentiment_score'] > df['sentiment_ma_7']) &
            (df['daily_sentiment_score'] > 0.1),

            # 强烈看空信号
            (df['daily_sentiment_score'] < df['sentiment_ma_3']) &
            (df['sentiment_ma_3'] < df['sentiment_ma_7']) &
            (df['sentiment_ma_7'] < df['sentiment_ma_14']) &
            (df['daily_sentiment_score'] < -0.3),

            # 温和看空信号
            (df['daily_sentiment_score'] < df['sentiment_ma_7']) &
            (df['daily_sentiment_score'] < -0.1)
        ]

        choices = ['强烈看多', '温和看多', '强烈看空', '温和看空']
        df['sentiment_signal'] = np.select(conditions, choices, default='中性')

        return df

    def sentiment_reversal_detection(self, daily_sentiment_df, threshold=0.5):
        """
        情感反转点检测
        """
        print(f"开始情感反转点检测（阈值：{threshold}）...")

        df = daily_sentiment_df.copy()

        # 计算情感变化率
        df['sentiment_change'] = df['daily_sentiment_score'].diff()
        df['sentiment_change_pct'] = df['daily_sentiment_score'].pct_change()

        # 检测反转点
        df['is_reversal'] = (abs(df['sentiment_change']) > threshold) & \
                           (df['sentiment_change'] * df['sentiment_change'].shift(1) < 0)

        # 标记反转类型
        conditions = [
            df['is_reversal'] & (df['sentiment_change'] > 0),
            df['is_reversal'] & (df['sentiment_change'] < 0)
        ]
        choices = ['向上反转', '向下反转']
        df['reversal_type'] = np.select(conditions, choices, default='无反转')

        reversal_points = df[df['is_reversal']].copy()

        return df, reversal_points

# 主要的执行函数
def run_comprehensive_sentiment_analysis(processed_comment_df, daily_sentiment_df):
    """
    运行全面的情感分析
    """
    results = {}

    # 1. 时间模式分析
    temporal_analyzer = TemporalSentimentAnalysis()
    results['intraday_pattern'] = temporal_analyzer.intraday_sentiment_pattern(processed_comment_df)
    results['weekday_pattern'] = temporal_analyzer.weekend_vs_weekday_sentiment(processed_comment_df)
    results['seasonality'] = temporal_analyzer.sentiment_seasonality_analysis(daily_sentiment_df)
    results['momentum_analysis'] = temporal_analyzer.sentiment_momentum_analysis(daily_sentiment_df)

    # 2. 网络传播分析
    network_analyzer = NetworkSentimentAnalysis()
    results['contagion'] = network_analyzer.sentiment_contagion_analysis(processed_comment_df)
    results['echo_chamber'] = network_analyzer.echo_chamber_detection(processed_comment_df)

    # 3. 预测性分析
    predictive_analyzer = PredictiveSentimentAnalysis()
    results['momentum_signals'] = predictive_analyzer.sentiment_momentum_signals(daily_sentiment_df)
    results['reversal_signals'], results['reversal_points'] = predictive_analyzer.sentiment_reversal_detection(daily_sentiment_df)

    return results
