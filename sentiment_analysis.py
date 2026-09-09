# -*- coding: utf-8 -*-
import os
import math
from transformers import pipeline
import pandas as pd
import numpy as np
import config
from util import DataSaver


class SentimentAnalyzer:
    """
    封装情感分析功能，支持增量处理。
    """

    def __init__(self, model_path):
        print("正在加载情感分析模型...")
        self.pipeline = pipeline('sentiment-analysis', model=model_path, tokenizer=model_path, device=-1)
        print("情感分析模型加载成功。")

    def load_existing_sentiment_data(self, stock_code):
        """加载已存在的情感分析结果"""
        sentiment_file_path = os.path.join(config.PROCESSED_DATA_PATH, f'{stock_code}_comments_with_sentiment.csv')

        if os.path.exists(sentiment_file_path):
            try:
                existing_df = pd.read_csv(sentiment_file_path)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                print(f"加载已有情感分析数据: {len(existing_df)} 条记录")
                return existing_df
            except Exception as e:
                print(f"加载已有情感数据时出错: {e}")
                return pd.DataFrame()
        else:
            print("未找到已有的情感分析数据文件")
            return pd.DataFrame()

    def filter_new_comments(self, processed_comment_df, existing_sentiment_df):
        """过滤出需要新分析的评论，支持补充老数据的增量分析"""
        if existing_sentiment_df.empty:
            print("没有已有数据，所有评论都需要分析")
            return processed_comment_df

        processed_comment_df['timestamp'] = pd.to_datetime(processed_comment_df['timestamp'])

        if 'comment_text' in existing_sentiment_df.columns:
            existing_keys = set(
                existing_sentiment_df['comment_text'].astype(str) + '|' +
                existing_sentiment_df['timestamp'].astype(str)
            )

            processed_keys = (
                processed_comment_df['comment_text'].astype(str) + '|' +
                processed_comment_df['timestamp'].astype(str)
            )

            mask = ~processed_keys.isin(existing_keys)
            new_comments = processed_comment_df[mask]

            print(f"已有情感分析数据: {len(existing_sentiment_df)} 条")
            print(f"待处理评论数据: {len(processed_comment_df)} 条")
            print(f"发现新评论: {len(new_comments)} 条需要分析")

            if len(new_comments) > 0:
                time_range = f"{new_comments['timestamp'].min()} 到 {new_comments['timestamp'].max()}"
                print(f"新数据时间范围: {time_range}")

        else:
            print("使用时间过滤方式进行增量分析")
            latest_analyzed_time = existing_sentiment_df['timestamp'].max()
            new_comments = processed_comment_df[processed_comment_df['timestamp'] > latest_analyzed_time]
            print(f"发现 {len(new_comments)} 条新评论需要分析")

        return new_comments

    def analyze_batch(self, texts):
        """对文本列表进行情感分析，返回-1, 0, 1分值。"""
        if not isinstance(texts, list):
            texts = list(texts)

        if len(texts) == 0:
            return []

        print(f"开始对 {len(texts)} 条文本进行情感分析...")
        results = self.pipeline(texts, batch_size=8, truncation=True)
        sentiment_scores = []

        for res in results:
            if res['label'] == 'Positive':
                sentiment_scores.append(1)
            elif res['label'] == 'Negative':
                sentiment_scores.append(-1)
            else:
                sentiment_scores.append(0)

        print("情感分析完成。")
        return sentiment_scores

    def merge_sentiment_data(self, existing_df, new_analyzed_df):
        """合并已有数据和新分析的数据"""
        if existing_df.empty:
            return new_analyzed_df

        if new_analyzed_df.empty:
            return existing_df

        merged_df = pd.concat([existing_df, new_analyzed_df], ignore_index=True)
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)

        print(f"数据合并完成，总计 {len(merged_df)} 条记录")
        return merged_df

    def analyze_with_incremental_processing(self, processed_comment_df, stock_code):
        """增量情感分析处理"""
        print("--- 开始增量情感分析 ---")

        existing_sentiment_df = self.load_existing_sentiment_data(stock_code)
        new_comments_df = self.filter_new_comments(processed_comment_df, existing_sentiment_df)

        if not new_comments_df.empty:
            new_sentiment_scores = self.analyze_batch(list(new_comments_df['cleaned_text']))
            new_comments_df = new_comments_df.copy()
            new_comments_df['sentiment_score'] = new_sentiment_scores
        else:
            print("没有新评论需要分析")
            new_comments_df = pd.DataFrame()

        complete_sentiment_df = self.merge_sentiment_data(existing_sentiment_df, new_comments_df)
        return complete_sentiment_df, len(new_comments_df)

    def aggregate_sentiment_daily(self, df):
        """将每条评论的情感分数按天聚合成每日情感指数，包含统计指标。"""
        print("开始按天聚合情感指数...")

        if 'timestamp' not in df.columns or 'sentiment_score' not in df.columns:
            if 'comment_date' in df.columns:
                df = df.rename(columns={'comment_date': 'timestamp'})
            else:
                raise ValueError("DataFrame必须包含 'timestamp' 和 'sentiment_score' 列")

        if df.empty:
            print("警告：在聚合情感时，没有有效的日期数据。")
            return pd.DataFrame(columns=['daily_sentiment_score', 'daily_sentiment_std', 'daily_comment_count'])

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if '点赞数' in df.columns:
            df['weighted_sentiment'] = df['sentiment_score'] * (1 + np.log2(df['点赞数'] + 1))
        else:
            df['weighted_sentiment'] = df['sentiment_score']

        grouped = df.set_index('timestamp').groupby(pd.Grouper(freq='D'))['weighted_sentiment']

        daily_sentiment = pd.DataFrame({
            'daily_sentiment_score': grouped.mean(),
            'daily_sentiment_std': grouped.std().fillna(0),
            'daily_comment_count': grouped.count(),
            'daily_sentiment_min': grouped.min(),
            'daily_sentiment_max': grouped.max()
        })

        print(f"每日情感指数聚合完成，生成 {len(daily_sentiment)} 天的数据，包含统计指标")
        return daily_sentiment
