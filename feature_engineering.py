# -*- coding: utf-8 -*-
"""
特征工程模块
整合所有分析结果为按日的特征表格
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import config

class FeatureEngineer:
    """
    特征工程器 - 整合所有分析结果
    """

    def __init__(self):
        print("特征工程器初始化完成")

    def create_comprehensive_daily_features(self,
                                          price_df,
                                          daily_sentiment_df,
                                          advanced_results,
                                          temporal_results):
        """
        创建综合的每日特征表格

        参数:
        - price_df: 股价数据
        - daily_sentiment_df: 每日情感数据
        - advanced_results: 高级情感分析结果
        - temporal_results: 时间序列分析结果

        返回:
        - comprehensive_df: 综合特征DataFrame
        """
        print("开始创建综合每日特征表格...")

        # 1. 基础价格特征
        price_features = self._extract_price_features(price_df)
        print(f"✅ 提取价格特征: {len(price_features.columns)} 列")

        # 2. 基础情感特征
        sentiment_features = self._extract_sentiment_features(daily_sentiment_df)
        print(f"✅ 提取情感特征: {len(sentiment_features.columns)} 列")

        # 3. 高级情感特征
        advanced_features = self._extract_advanced_features(advanced_results)
        print(f"✅ 提取高级特征: {len(advanced_features.columns)} 列")

        # 4. 时间序列特征
        temporal_features = self._extract_temporal_features(temporal_results)
        print(f"✅ 提取时间特征: {len(temporal_features.columns)} 列")

        # 5. 技术指标特征
        technical_features = self._extract_technical_features(price_df)
        print(f"✅ 提取技术特征: {len(technical_features.columns)} 列")

        # 6. 合并所有特征
        comprehensive_df = self._merge_all_features([
            price_features,
            sentiment_features,
            advanced_features,
            temporal_features,
            technical_features
        ])

        # 7. 添加时间特征
        comprehensive_df = self._add_time_features(comprehensive_df)

        # 8. 计算衍生特征
        comprehensive_df = self._calculate_derived_features(comprehensive_df)

        print(f"🎉 综合特征表格创建完成！总共 {len(comprehensive_df.columns)} 列，{len(comprehensive_df)} 行")

        return comprehensive_df

    def _extract_price_features(self, price_df):
        """提取价格相关特征 - 适应不同的列名格式"""
        df = price_df.copy()

        # 确保日期为索引
        if 'date' in df.columns:
            df = df.set_index('date')

        price_features = pd.DataFrame(index=df.index)

        # 基础价格数据 - 处理可能的列名变化
        price_features['开盘价'] = df['open']
        price_features['最高价'] = df['high']
        price_features['最低价'] = df['low']
        price_features['收盘价'] = df['close']

        # 成交量 - 尝试不同的列名
        volume_col = None
        for col_name in ['volume', 'vol', '成交量', 'turnover']:
            if col_name in df.columns:
                volume_col = col_name
                break

        if volume_col:
            price_features['成交量'] = df[volume_col]
            price_features['成交额'] = df.get('amount', df[volume_col] * df['close'])
        else:
            print("⚠️ 警告：未找到成交量相关列，将使用默认值")
            price_features['成交量'] = 0
            price_features['成交额'] = 0

        # 价格变化
        price_features['日收益率'] = df['pct_change']
        price_features['价格变化额'] = df['close'].diff()
        price_features['振幅'] = (df['high'] - df['low']) / df['close'].shift(1)

        # 成交量特征 - 只有在有成交量数据时才计算
        if volume_col:
            price_features['成交量变化率'] = df[volume_col].pct_change()
            price_features['量价比'] = df[volume_col] / df['close']
        else:
            price_features['成交量变化率'] = 0
            price_features['量价比'] = 0

        return price_features

    def _extract_sentiment_features(self, daily_sentiment_df):
        """提取基础情感特征"""
        df = daily_sentiment_df.copy()

        # 确保日期为索引
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        sentiment_features = pd.DataFrame(index=df.index)

        # 基础情感指标
        sentiment_features['每日情感分数'] = df['daily_sentiment_score']
        sentiment_features['评论数量'] = df['daily_comment_count']  # 修正字段名
        sentiment_features['正面评论数'] = df.get('positive_count', 0)
        sentiment_features['负面评论数'] = df.get('negative_count', 0)
        sentiment_features['中性评论数'] = df.get('neutral_count', 0)

        # 情感强度
        sentiment_features['情感强度'] = df['daily_sentiment_score'].abs()
        sentiment_features['情感方向'] = np.sign(df['daily_sentiment_score'])

        # 情感变化 - 添加前缀避免冲突
        sentiment_features['基础_情感变化'] = df['daily_sentiment_score'].diff()
        sentiment_features['基础_情感变化率'] = df['daily_sentiment_score'].pct_change()

        return sentiment_features

    def _extract_advanced_features(self, advanced_results):
        """提取高级情感分析特征"""
        features_list = []

        # 1. 主题情感特征
        if 'topic_sentiment' in advanced_results:
            topic_features = self._process_topic_sentiment(advanced_results['topic_sentiment'])
            features_list.append(topic_features)

        # 2. 情感强度特征
        if 'intensity_stats' in advanced_results:
            # 这个通常是统计数据，我们需要从原始数据重新计算每日强度
            pass

        # 3. 极端情感事件
        if 'extreme_events' in advanced_results:
            extreme_features = self._process_extreme_events(advanced_results['extreme_events'])
            features_list.append(extreme_features)

        # 4. 背离分析特征
        if 'divergence_df' in advanced_results:
            divergence_features = self._process_divergence_features(advanced_results['divergence_df'])
            features_list.append(divergence_features)

        # 合并所有高级特征
        if features_list:
            # 找到共同的日期索引
            common_index = features_list[0].index
            for df in features_list[1:]:
                common_index = common_index.intersection(df.index)

            advanced_features = pd.DataFrame(index=common_index)
            for df in features_list:
                advanced_features = advanced_features.join(df, how='left')
        else:
            # 创建空的DataFrame
            advanced_features = pd.DataFrame()

        return advanced_features

    def _process_topic_sentiment(self, topic_sentiment_dict):
        """处理主题情感数据"""
        topic_features = pd.DataFrame()

        for topic, sentiment_series in topic_sentiment_dict.items():
            col_name = f'{topic}_情感'
            topic_features[col_name] = sentiment_series

        # 计算主题情感的统计特征
        if len(topic_features.columns) > 0:
            topic_features['主题情感_均值'] = topic_features.mean(axis=1)
            topic_features['主题情感_标准差'] = topic_features.std(axis=1)
            topic_features['主题情感_最大值'] = topic_features.max(axis=1)
            topic_features['主题情感_最小值'] = topic_features.min(axis=1)

        return topic_features

    def _process_extreme_events(self, extreme_events_df):
        """处理极端情感事件"""
        extreme_features = pd.DataFrame(index=extreme_events_df.index)

        extreme_features['极端正面事件数'] = extreme_events_df['extreme_positive_count']
        extreme_features['极端负面事件数'] = extreme_events_df['extreme_negative_count']
        extreme_features['极端事件总数'] = (extreme_events_df['extreme_positive_count'] +
                                    extreme_events_df['extreme_negative_count'])
        extreme_features['极端事件净值'] = (extreme_events_df['extreme_positive_count'] -
                                    extreme_events_df['extreme_negative_count'])

        return extreme_features

    def _process_divergence_features(self, divergence_df):
        """处理背离分析特征"""
        divergence_features = pd.DataFrame(index=divergence_df.index)

        divergence_features['情感_标准化'] = divergence_df['sentiment']
        divergence_features['收益率_标准化'] = divergence_df['returns']
        divergence_features['情感方向'] = divergence_df['sentiment_direction']
        divergence_features['价格方向'] = divergence_df['price_direction']
        divergence_features['是否背离'] = divergence_df['is_divergence'].astype(int)

        return divergence_features

    def _extract_temporal_features(self, temporal_results):
        """提取时间序列特征"""
        features_list = []

        # 1. 情感动量特征
        if 'momentum_analysis' in temporal_results:
            momentum_features = self._process_momentum_features(temporal_results['momentum_analysis'])
            features_list.append(momentum_features)
        elif 'momentum_signals' in temporal_results:
            momentum_features = self._process_momentum_features(temporal_results['momentum_signals'])
            features_list.append(momentum_features)

        # 2. 反转信号特征
        if 'reversal_signals' in temporal_results:
            reversal_features = self._process_reversal_features(temporal_results['reversal_signals'])
            features_list.append(reversal_features)

        # 合并时间特征
        if features_list:
            common_index = features_list[0].index
            for df in features_list[1:]:
                common_index = common_index.intersection(df.index)

            temporal_features = pd.DataFrame(index=common_index)
            for df in features_list:
                temporal_features = temporal_features.join(df, how='left')
        else:
            temporal_features = pd.DataFrame()

        return temporal_features

    def _process_momentum_features(self, momentum_df):
        """处理动量特征"""
        momentum_features = pd.DataFrame(index=momentum_df.index)

        momentum_features['情感移动平均'] = momentum_df['sentiment_ma']
        momentum_features['情感动量'] = momentum_df['sentiment_momentum']
        momentum_features['动量_情感变化率'] = momentum_df['sentiment_change_rate']  # 添加前缀避免冲突
        momentum_features['情感波动率'] = momentum_df['sentiment_volatility']

        # 如果有交易信号
        if 'sentiment_signal' in momentum_df.columns:
            momentum_features['交易信号'] = momentum_df['sentiment_signal']
            # 将交易信号转换为数值
            signal_map = {'强烈看多': 2, '温和看多': 1, '中性': 0, '温和看空': -1, '强烈看空': -2}
            momentum_features['交易信号_数值'] = momentum_df['sentiment_signal'].map(signal_map).fillna(0)

        return momentum_features

    def _process_reversal_features(self, reversal_df):
        """处理反转特征"""
        reversal_features = pd.DataFrame(index=reversal_df.index)

        reversal_features['是否反转'] = reversal_df['is_reversal'].astype(int)

        if 'reversal_type' in reversal_df.columns:
            reversal_features['反转类型'] = reversal_df['reversal_type']
            # 转换为数值
            reversal_map = {'向上反转': 1, '向下反转': -1, '无反转': 0}
            reversal_features['反转类型_数值'] = reversal_df['reversal_type'].map(reversal_map).fillna(0)

        return reversal_features

    def _extract_technical_features(self, price_df):
        """提取技术指标特征 - 适应不同的列名格式"""
        df = price_df.copy()

        if 'date' in df.columns:
            df = df.set_index('date')

        technical_features = pd.DataFrame(index=df.index)

        # 移动平均线
        technical_features['MA5'] = df['close'].rolling(5).mean()
        technical_features['MA10'] = df['close'].rolling(10).mean()
        technical_features['MA20'] = df['close'].rolling(20).mean()

        # 价格相对位置
        technical_features['价格_MA5_比'] = df['close'] / technical_features['MA5']
        technical_features['价格_MA20_比'] = df['close'] / technical_features['MA20']

        # 成交量移动平均 - 尝试不同的列名
        volume_col = None
        for col_name in ['volume', 'vol', '成交量', 'turnover']:
            if col_name in df.columns:
                volume_col = col_name
                break

        if volume_col:
            technical_features['成交量_MA5'] = df[volume_col].rolling(5).mean()
            technical_features['成交量_相对强度'] = df[volume_col] / technical_features['成交量_MA5']
        else:
            technical_features['成交量_MA5'] = 0
            technical_features['成交量_相对强度'] = 0

        # RSI (简化版)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        technical_features['RSI'] = 100 - (100 / (1 + rs))

        # 布林带
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        technical_features['布林上轨'] = rolling_mean + (rolling_std * 2)
        technical_features['布林下轨'] = rolling_mean - (rolling_std * 2)
        technical_features['布林位置'] = (df['close'] - technical_features['布林下轨']) / (technical_features['布林上轨'] - technical_features['布林下轨'])

        return technical_features

    def _merge_all_features(self, feature_dfs):
        """合并所有特征DataFrame"""
        # 过滤掉空的DataFrame
        valid_dfs = [df for df in feature_dfs if not df.empty]

        if not valid_dfs:
            return pd.DataFrame()

        # 找到共同的日期范围
        common_index = valid_dfs[0].index
        for df in valid_dfs[1:]:
            common_index = common_index.intersection(df.index)

        # 基于共同日期范围合并
        comprehensive_df = pd.DataFrame(index=common_index)

        for df in valid_dfs:
            comprehensive_df = comprehensive_df.join(df, how='left')

        return comprehensive_df

    def _add_time_features(self, df):
        """添加时间特征"""
        if df.empty:
            return df

        df = df.copy()

        # 时间特征
        df['年'] = df.index.year
        df['月'] = df.index.month
        df['日'] = df.index.day
        df['星期'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['是否周末'] = (df.index.dayofweek >= 5).astype(int)
        df['季度'] = df.index.quarter
        df['年内第几天'] = df.index.dayofyear

        return df

    def _calculate_derived_features(self, df):
        """计算衍生特征"""
        if df.empty:
            return df

        df = df.copy()

        # 1. 价格动量特征
        if '收盘价' in df.columns:
            df['价格动量_3日'] = df['收盘价'].rolling(3).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 3 else np.nan)
            df['价格动量_5日'] = df['收盘价'].rolling(5).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 5 else np.nan)

        # 2. 情感-价格相关性特征
        if '每日情感分数' in df.columns and '日收益率' in df.columns:
            # 滚动相关性
            window = 10
            df['情感价格相关性_10日'] = df['每日情感分数'].rolling(window).corr(df['日收益率'])

            # 情感与收益率的乘积（同向性指标）
            df['情感收益同向性'] = df['每日情感分数'] * df['日收益率']

        # 3. 综合信号特征
        if '交易信号_数值' in df.columns and 'RSI' in df.columns:
            df['综合信号'] = (df['交易信号_数值'] + (df['RSI'] - 50) / 25) / 2  # 标准化RSI并结合

        # 4. 风险特征
        if '情感波动率' in df.columns and '振幅' in df.columns:
            df['综合波动率'] = (df['情感波动率'].fillna(0) + df['振幅'].fillna(0)) / 2

        return df

    def save_features(self, comprehensive_df, filename=None):
        """保存特征表格"""
        if filename is None:
            filename = f"{config.STOCK_CODE}_comprehensive_features.csv"

        filepath = os.path.join('data/processed', filename)
        comprehensive_df.to_csv(filepath, encoding='utf-8-sig')
        print(f"✅ 特征表格已保存到: {filepath}")

        return filepath

    def print_feature_summary(self, comprehensive_df):
        """打印特征总结"""
        print("\n" + "="*80)
        print("📊 综合特征表格总结")
        print("="*80)

        print(f"📅 时间范围: {comprehensive_df.index.min().strftime('%Y-%m-%d')} 至 {comprehensive_df.index.max().strftime('%Y-%m-%d')}")
        print(f"📊 数据行数: {len(comprehensive_df)} 天")
        print(f"📈 特征列数: {len(comprehensive_df.columns)} 个")

        print(f"\n🔍 特征分类统计:")

        # 按特征类型分类
        feature_categories = {
            '价格特征': ['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '日收益率', '价格变化额', '振幅', '成交量变化率', '量价比'],
            '情感特征': ['每日情感分数', '评论数量', '正面评论数', '负面评论数', '中性评论数', '情感强度', '情感方向', '情感变化', '情感变化率'],
            '主题特征': [col for col in comprehensive_df.columns if '_情感' in col or '主题情感' in col],
            '极端事件': [col for col in comprehensive_df.columns if '极端' in col],
            '技术指标': ['MA5', 'MA10', 'MA20', 'RSI', '布林上轨', '布林下轨', '布林位置'] + [col for col in comprehensive_df.columns if 'MA' in col or '价格_' in col],
            '动量特征': [col for col in comprehensive_df.columns if '动量' in col or '波动' in col],
            '时间特征': ['年', '月', '日', '星期', '是否周末', '季度', '年内第几天'],
            '衍生特征': [col for col in comprehensive_df.columns if '相关性' in col or '同向性' in col or '综合' in col],
            '交易信号': [col for col in comprehensive_df.columns if '信号' in col or '反转' in col or '背离' in col]
        }

        for category, features in feature_categories.items():
            existing_features = [f for f in features if f in comprehensive_df.columns]
            if existing_features:
                print(f"  {category}: {len(existing_features)} 个")
                print(f"    {', '.join(existing_features[:3])}{'...' if len(existing_features) > 3 else ''}")

        print(f"\n📋 数据质量:")
        missing_rate = comprehensive_df.isnull().sum().sum() / (len(comprehensive_df) * len(comprehensive_df.columns))
        print(f"  缺失值比例: {missing_rate:.2%}")
        print(f"  完整数据行数: {comprehensive_df.dropna().shape[0]} 行")

        print("\n" + "="*80)


def create_comprehensive_features(price_df, daily_sentiment_df, advanced_results, temporal_results):
    """
    创建综合特征表格的主函数

    返回:
    - comprehensive_df: 综合特征DataFrame
    """
    engineer = FeatureEngineer()

    # 创建综合特征
    comprehensive_df = engineer.create_comprehensive_daily_features(
        price_df, daily_sentiment_df, advanced_results, temporal_results
    )

    # 打印特征总结
    engineer.print_feature_summary(comprehensive_df)

    # 保存特征表格
    filepath = engineer.save_features(comprehensive_df)

    return comprehensive_df, filepath
