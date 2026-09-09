# -*- coding: utf-8 -*-
import re
import jieba
import pandas as pd
from pystopwords import stopwords
from datetime import datetime, timedelta
import config
import os

class DataPreprocessor:
    """
    封装所有数据预处理逻辑。
    """
    def __init__(self):
        self.stop_words = stopwords(langs='zh')
        print("数据预处理器初始化成功，已加载中文停用词表。")

    def _parse_timestamp(self, ts_str):
        """
        智能解析多种时间格式，包括股吧API格式。
        支持格式: 
        - 股吧API: "2025-06-27 22:48:43" 
        - 雪球格式: "今天 10:30", "1小时前", "06-25 15:00", "2023-12-31 23:59"
        """
        if not isinstance(ts_str, str):
            return None

        now = datetime.now()

        # 首先尝试解析股吧API的完整时间戳格式
        try:
            # 尝试解析 "YYYY-MM-DD HH:MM:SS" 格式
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').date()
        except ValueError:
            pass

        # 雪球格式的时间解析（保持兼容性）
        if '今天' in ts_str or '分钟前' in ts_str or '小时前' in ts_str:
            return now.date()

        # 匹配 "MM-DD" 格式, e.g., "06-20 15:30"
        match = re.match(r'(\d{2}-\d{2})', ts_str)
        if match:
            try:
                date_part = match.group(1)
                # 假设是当年的日期
                return datetime.strptime(f'{now.year}-{date_part}', '%Y-%m-%d').date()
            except ValueError:
                return None  # 日期格式错误

        # 匹配 "YYYY-MM-DD" 格式
        match = re.match(r'(\d{4}-\d{2}-\d{2})', ts_str)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y-%m-%d').date()
            except ValueError:
                return None

        return None  # 所有格式均不匹配

    def clean_text(self, text):
        """清洗单条评论文本。"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\$\S+\$', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        # text = re.sub(r'[a-zA-Z0-9]', '', text)   不需要过滤英文和数字
        # text = re.sub(r'[^\w\s]', '', text)   不需要过滤标点符号
        return " ".join(text.split())       # 空格格式化。

    def segment_and_filter_stopwords(self, text):
        """对文本进行标点过滤和分词并过滤停用词。"""
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        # print(f"分词前文本: {text}")
        words = jieba.cut(text)
        return " ".join(words)

    def process_comments(self, df):
        """对整个评论DataFrame进行预处理，支持股吧API和雪球数据格式。"""
        print("开始预处理评论数据...")

        # 检测数据源格式并标准化列名
        if 'timestamp' in df.columns:
            # 股吧API格式
            time_column = 'timestamp'
            print("检测到股吧API数据格式")
        elif 'time' in df.columns:
            # 雪球格式（保持向后兼容）
            time_column = 'time'
            print("检测到雪球数据格式")
        else:
            raise ValueError("未找到时间列（'timestamp' 或 'time'）")

        # 1. 标准化时间戳
        df['comment_date'] = df[time_column].apply(self._parse_timestamp)
        df.dropna(subset=['comment_date'], inplace=True)
        df['comment_date'] = pd.to_datetime(df['comment_date'])

        # 2. 处理点赞数（股吧API可能有此字段）
        if 'post_like_count' in df.columns:
            df['点赞数'] = df['post_like_count'].fillna(0)
        else:
            df['点赞数'] = 0  # 默认值

        # 3. 清洗和处理文本
        df['cleaned_text'] = df['comment_text'].apply(self.clean_text)
        df['single_word'] = df['cleaned_text'].apply(self.segment_and_filter_stopwords)
        df.dropna(subset=['single_word'], inplace=True)
        df = df[df['single_word'] != '']
        
        print(f"评论数据预处理完成，有效数据：{len(df)} 条")
        return df

    def process_prices(self, df):
        """对价格DataFrame进行预处理。"""
        print("开始预处理价格数据...")
        # 修正: 明确指定Tushare返回的日期格式，提高代码健壮性
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df.sort_values('trade_date', inplace=True)
        df.set_index('trade_date', inplace=True)
        df['pct_change'] = df['close'].pct_change()
        df.dropna(subset=['pct_change'], inplace=True)
        print("价格数据预处理完成。")
        return df

    def preprocess_price_data(self, stock_code):
        """预处理价格数据的统一接口"""
        raw_path = os.path.join(config.RAW_DATA_PATH, f'{stock_code}_price_raw.csv')
        processed_path = os.path.join(config.PROCESSED_DATA_PATH, f'{stock_code}_price_processed.csv')
        
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            processed_df = self.process_prices(df)
            processed_df.to_csv(processed_path, encoding='utf-8-sig')
            print(f"价格数据预处理完成，保存到: {processed_path}")
        else:
            print(f"原始价格数据文件不存在: {raw_path}")
    
    def preprocess_comment_data(self, stock_code):
        """预处理评论数据的统一接口"""
        raw_path = os.path.join(config.RAW_DATA_PATH, f'{stock_code}_comment_raw.csv')
        processed_path = os.path.join(config.PROCESSED_DATA_PATH, f'{stock_code}_comments_processed.csv')
        
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            processed_df = self.process_comments(df)
            processed_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
            print(f"评论数据预处理完成，保存到: {processed_path}")
        else:
            print(f"原始评论数据文件不存在: {raw_path}")
