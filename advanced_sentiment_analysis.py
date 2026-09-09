# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class AdvancedSentimentAnalysis:
    """
    高级情感分析器，提供多维度的情感分析功能
    使用关键词匹配方式，简单高效
    """

    def __init__(self):
        # 定义不同主题的关键词
        self.topic_keywords = {
            '基本面': ['业绩', '财报', '营收', '利润', '增长', '亏损', '盈利', '净利润', '毛利率', 'ROE', 'PE', 'PB', '财务', '资产', '负债', '现金流', '估值'],
            '技术面': ['突破', '支撑', '阻力', '均线', 'MACD', 'KDJ', '成交量', '放量', '缩量', '回调', '反弹', '趋势', '形态', '指标', '买点', '卖点'],
            '政策面': ['政策', '利好', '利空', '监管', '央行', '降准', '降息', '政府', '法规', '新规', '税收', '补贴', '扶持', '限制'],
            '市场情绪': ['恐慌', '贪婪', '看多', '看空', '抄底', '逃顶', '割肉', '加仓', '减仓', '满仓', '信心', '悲观', '乐观', '焦虑'],
            '消息面': ['公告', '重组', '并购', '分红', '送股', '增持', '减持', '停牌', '复牌', '退市', '新闻', '传言', '辟谣', '澄清']
        }

        # 情感强度词汇
        self.intensity_words = {
            '极强': ['暴涨', '暴跌', '崩盘', '狂欢', '血亏', '巨亏', '翻倍', '腰斩', '爆仓', '跳水'],
            '强': ['大涨', '大跌', '飙升', '重挫', '飞天', '砸盘', '拉升', '杀跌', '冲高'],
            '中': ['上涨', '下跌', '回调', '反弹', '调整', '震荡', '盘整'],
            '弱': ['微涨', '微跌', '震荡', '横盘', '窄幅', '小幅']
        }

        print("高级情感分析器初始化完成（关键词匹配模式）")

    def topic_sentiment_analysis(self, df):
        """
        基于主题的情感分析 - 支持词向量和关键词匹配
        增加详细的匹配统计信息
        """
        print("开始主题情感分析...")
        print("使用关键词匹配模式进行主题识别")

        topic_sentiment = defaultdict(list)

        # 添加匹配统计
        match_stats = {
            'total_texts': 0,
            'keyword_matched': 0,
            'unmatched': 0,
            'keyword_details': defaultdict(list)
        }

        for idx, row in df.iterrows():
            text = row['cleaned_text']
            sentiment = row['sentiment_score']
            date = row['timestamp']
            match_stats['total_texts'] += 1

            matched = False

            # 直接使用关键词匹配
            for topic, keywords in self.topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topic_sentiment[topic].append({
                        'date': date,
                        'sentiment': sentiment,
                        'text': text,
                        'confidence': 1.0,  # 关键词匹配设为满分
                        'match_type': 'keyword'
                    })
                    match_stats['keyword_matched'] += 1
                    # 找出匹配的关键词
                    matched_keywords = [kw for kw in keywords if kw in text]
                    match_stats['keyword_details'][topic].append({
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'matched_keywords': matched_keywords
                    })
                    matched = True
                    break

            if not matched:
                match_stats['unmatched'] += 1

        # 打印详细统计信息
        self._print_match_statistics(match_stats)

        # 聚合每个主题的日均情感
        topic_daily_sentiment = {}
        for topic, records in topic_sentiment.items():
            if records:
                topic_df = pd.DataFrame(records)
                topic_df['date'] = pd.to_datetime(topic_df['date']).dt.date

                daily_avg = topic_df.groupby('date')['sentiment'].mean()

                topic_daily_sentiment[topic] = daily_avg

        print(f"完成 {len(topic_daily_sentiment)} 个主题的情感分析")
        return topic_daily_sentiment

    def _print_match_statistics(self, stats):
        """打印匹配统计信息"""
        print("\n" + "="*60)
        print("📊 主题匹配统计报告")
        print("="*60)

        total = stats['total_texts']
        keyword_count = stats['keyword_matched']
        unmatched_count = stats['unmatched']

        print(f"📝 总评论数: {total}")
        print(f"🔤 关键词匹配: {keyword_count} ({keyword_count/total*100:.1f}%)")
        print(f"❌ 未匹配: {unmatched_count} ({unmatched_count/total*100:.1f}%)")

        # 关键词匹配详情
        if keyword_count > 0:
            print(f"\n🔤 关键词匹配详情:")
            for topic, details in stats['keyword_details'].items():
                print(f"  • {topic}: {len(details)}条")
                # 显示前2个例子
                for i, example in enumerate(details[:2]):
                    keywords_str = ', '.join(example['matched_keywords'])
                    print(f"    - \"{example['text']}\" (关键词: {keywords_str})")

        print("="*60 + "\n")

    def _detect_intensity(self, text):
        """
        检测情感强度 - 使用关键词匹配
        """
        # 按强度从高到低检查，优先匹配强度高的词汇
        for level in ['极强', '强', '中', '弱']:
            words = self.intensity_words[level]
            if any(word in text for word in words):
                return level
        return '中'  # 默认中等强度

    def sentiment_intensity_analysis(self, df):
        """
        情感强度分析
        """
        print("开始情感强度分析...")

        df['intensity_level'] = df['cleaned_text'].apply(self._detect_intensity)
        df['weighted_sentiment'] = df.apply(
            lambda row: self._apply_intensity_weight(row['sentiment_score'], row['intensity_level']),
            axis=1
        )

        # 按强度级别统计
        intensity_stats = df.groupby('intensity_level').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'weighted_sentiment': ['mean', 'std']
        }).round(4)

        print("情感强度分析完成")
        return df, intensity_stats

    def _apply_intensity_weight(self, sentiment, intensity):
        """根据强度调整情感分数"""
        weights = {'极强': 2.0, '强': 1.5, '中': 1.0, '弱': 0.5}
        return sentiment * weights.get(intensity, 1.0)

    def user_influence_analysis(self, df):
        """
        用户影响力分析（基于点赞数）
        """
        print("开始用户影响力分析...")

        if '点赞数' not in df.columns:
            print("缺少点赞数据，跳过影响力分析")
            return df

        # 根据点赞数计算影响力权重
        df['influence_weight'] = np.log2(df['点赞数'] + 1)  # 对数变换避免极值影响
        df['influenced_sentiment'] = df['sentiment_score'] * df['influence_weight']

        # 计算加权平均情感
        daily_influenced_sentiment = df.groupby(df['timestamp'].dt.date).apply(
            lambda x: np.average(x['sentiment_score'], weights=x['influence_weight'])
            if x['influence_weight'].sum() > 0 else x['sentiment_score'].mean()
        )

        print("用户影响力分析完成")
        return df, daily_influenced_sentiment

    def sentiment_momentum_analysis(self, daily_sentiment_df, window=5):
        """
        情感动量分析
        """
        print(f"开始情感动量分析（窗口期={window}天）...")

        # 计算情感移动平均
        daily_sentiment_df['sentiment_ma'] = daily_sentiment_df['daily_sentiment_score'].rolling(window=window).mean()

        # 计算情感动量（当前值与移动平均的差值）
        daily_sentiment_df['sentiment_momentum'] = (
            daily_sentiment_df['daily_sentiment_score'] - daily_sentiment_df['sentiment_ma']
        )

        # 计算情感变化率
        daily_sentiment_df['sentiment_change_rate'] = daily_sentiment_df['daily_sentiment_score'].pct_change()

        # 计算情感波动率（滚动标准差）
        daily_sentiment_df['sentiment_volatility'] = daily_sentiment_df['daily_sentiment_score'].rolling(window=window).std()

        print("情感动量分析完成")
        return daily_sentiment_df

    def extreme_sentiment_detection(self, df, threshold=1.5):
        """
        极端情感检测
        """
        print(f"开始极端情感检测（阈值={threshold}）...")

        # 计算情感分数的标准差
        sentiment_std = df['sentiment_score'].std()
        sentiment_mean = df['sentiment_score'].mean()

        # 标记极端情感
        df['is_extreme_positive'] = df['sentiment_score'] > (sentiment_mean + threshold * sentiment_std)
        df['is_extreme_negative'] = df['sentiment_score'] < (sentiment_mean - threshold * sentiment_std)

        # 统计极端情感事件
        extreme_positive_events = df[df['is_extreme_positive']].groupby(df['timestamp'].dt.date).size()
        extreme_negative_events = df[df['is_extreme_negative']].groupby(df['timestamp'].dt.date).size()

        extreme_events_df = pd.DataFrame({
            'extreme_positive_count': extreme_positive_events,
            'extreme_negative_count': extreme_negative_events
        }).fillna(0)

        print(f"检测到 {extreme_positive_events.sum()} 个极端正面情感事件")
        print(f"检测到 {extreme_negative_events.sum()} 个极端负面情感事件")

        return df, extreme_events_df

    def sentiment_divergence_analysis(self, sentiment_df, price_df):
        """
        情感-价格背离分析
        """
        print("开始情感-价格背离分析...")

        # 对齐数据
        from correlation_analysis import CorrelationEngine
        engine = CorrelationEngine()

        aligned_sentiment, aligned_returns = engine.align_series(
            sentiment_df['daily_sentiment_score'],
            price_df['pct_change']
        )

        # 计算3日移动平均来平滑数据
        sentiment_ma3 = aligned_sentiment.rolling(3).mean()
        returns_ma3 = aligned_returns.rolling(3).mean()

        # 检测背离：情感和价格变化方向相反
        sentiment_direction = np.sign(sentiment_ma3.diff())
        price_direction = np.sign(returns_ma3.diff())

        divergence = (sentiment_direction * price_direction) < 0  # 方向相反

        divergence_df = pd.DataFrame({
            'sentiment': aligned_sentiment,
            'returns': aligned_returns,
            'sentiment_direction': sentiment_direction,
            'price_direction': price_direction,
            'is_divergence': divergence
        })

        divergence_events = divergence_df[divergence_df['is_divergence']].index
        print(f"检测到 {len(divergence_events)} 个情感-价格背离事件")

        return divergence_df, divergence_events

    def generate_sentiment_report(self, results_dict):
        """
        生成综合情感分析报告
        """
        print("生成综合情感分析报告...")

        report_lines = [
            "# 股票评论情感分析综合报告",
            f"## 报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 主题情感分析结果"
        ]

        if 'topic_sentiment' in results_dict:
            for topic, sentiment_series in results_dict['topic_sentiment'].items():
                avg_sentiment = sentiment_series.mean()
                report_lines.append(f"- **{topic}**：平均情感分数 {avg_sentiment:.3f}")

        if 'intensity_stats' in results_dict:
            report_lines.extend([
                "",
                "## 2. 情感强度分析",
                "| 强度级别 | 平均情感 | 标准差 | 数量 |",
                "|---------|---------|--------|------|"
            ])

            for level, stats in results_dict['intensity_stats'].iterrows():
                mean_val = stats[('sentiment_score', 'mean')]
                std_val = stats[('sentiment_score', 'std')]
                count_val = int(stats[('sentiment_score', 'count')])
                report_lines.append(f"| {level} | {mean_val:.3f} | {std_val:.3f} | {count_val} |")

        if 'extreme_events' in results_dict:
            extreme_df = results_dict['extreme_events']
            total_positive = extreme_df['extreme_positive_count'].sum()
            total_negative = extreme_df['extreme_negative_count'].sum()
            report_lines.extend([
                "",
                "## 3. 极端情感事件",
                f"- 极端正面情感事件：{total_positive} 次",
                f"- 极端负面情感事件：{total_negative} 次"
            ])

        if 'divergence_events' in results_dict:
            divergence_count = len(results_dict['divergence_events'])
            report_lines.extend([
                "",
                "## 4. 情感-价格背离分析",
                f"- 检测到背离事件：{divergence_count} 次",
                f"- 背离事件日期：{list(results_dict['divergence_events'].strftime('%Y-%m-%d'))}"
            ])

        report_content = "\n".join(report_lines)

        # 保存报告
        with open('reports/advanced_sentiment_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report_content)

        print("综合情感分析报告已保存到: reports/advanced_sentiment_analysis_report.md")
        return report_content


# 使用示例函数
def run_advanced_sentiment_analysis(processed_comment_df, daily_sentiment_df, processed_price_df):
    """
    运行高级情感分析的主函数
    """
    analyzer = AdvancedSentimentAnalysis()
    results = {}

    # 1. 主题情感分析
    results['topic_sentiment'] = analyzer.topic_sentiment_analysis(processed_comment_df)

    # 2. 情感强度分析
    enhanced_df, results['intensity_stats'] = analyzer.sentiment_intensity_analysis(processed_comment_df)

    # 3. 用户影响力分析
    if '点赞数' in processed_comment_df.columns:
        influenced_df, results['influenced_sentiment'] = analyzer.user_influence_analysis(enhanced_df)

    # 4. 情感动量分析
    results['momentum_df'] = analyzer.sentiment_momentum_analysis(daily_sentiment_df.copy())

    # 5. 极端情感检测
    extreme_df, results['extreme_events'] = analyzer.extreme_sentiment_detection(enhanced_df)

    # 6. 情感-价格背离分析
    results['divergence_df'], results['divergence_events'] = analyzer.sentiment_divergence_analysis(
        daily_sentiment_df, processed_price_df
    )

    # 7. 生成综合报告
    analyzer.generate_sentiment_report(results)

    return results
