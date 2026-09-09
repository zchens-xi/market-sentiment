"""
对data/raw 目录下的原始数据进行分析处理，生成报告和中间项文件，包括：
1. 数据清洗和预处理
2. 情感分析
3. 相关性分析
4. 可视化
5. 报告生成

可以添加更多分析步骤，独立运行然后再合并到这里来。
所以为什么不一开始就独立运行。
"""

# -*- coding: utf-8 -*-
# 1. 屏蔽无关的技术警告
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')

# 2. 导入项目模块
import config
from data_ingestion import DataSaver
from data_preprocessing import DataPreprocessor
from sentiment_analysis import SentimentAnalyzer
from correlation_analysis import CorrelationEngine
from visualization import Visualizer
from report_generator import ReportGenerator
import pandas as pd
import numpy as np
import json

# 将HTML模板直接定义在主文件中，方便管理
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        h1, h2 { color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }
        .card { padding: 20px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1); }
        .metric { font-size: 1.5em; font-weight: bold; color: #007bff; }
        footer { text-align: center; margin-top: 30px; color: #6c757d; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>本报告旨在分析 <b>{{ stock_name }} ({{ stock_code }})</b> 在 <b>{{ start_date }}</b> 到 <b>{{ end_date }}</b> 期间，公众评论情感与股票价格变动之间的相关性。</p>

        <h2>关键指标摘要</h2>
        <div class="card">
            <p>情感与收益率整体相关系数: <span class="metric">{{ '%.4f'|format(corr_metrics.overall) if corr_metrics.overall is not none else 'N/A' }}</span></p>
            <p>情感领先价格相关性峰值: <span class="metric">{{ '%.4f'|format(corr_metrics.max_lead_corr) if corr_metrics.max_lead_corr is not none else 'N/A' }}</span> (领先 {{ corr_metrics.max_lead_day }} 天)</p>
            <p>价格影响情绪相关性峰值: <span class="metric">{{ '%.4f'|format(corr_metrics.max_lag_corr) if corr_metrics.max_lag_corr is not none else 'N/A' }}</span> (滞后 {{ corr_metrics.max_lag_day }} 天)</p>
        </div>

        <h2>可视化分析</h2>
        <div class="grid">
            <div class="card">{{ figures.price_sentiment | safe }}</div>
            <div class="card">{{ figures.scatterplot | safe }}</div>
            <div class="card">{{ figures.rolling_corr | safe }}</div>
            <div class="card">{{ figures.lagged_corr | safe }}</div>
        </div>

        <footer>报告生成于 {{ generated_at }}</footer>
    </div>
</body>
</html>
"""


def load_raw_data(stock_code):
    """从CSV文件加载原始的价格和评论数据。"""
    print("--- 步骤 1: 加载原始数据 ---")
    price_path = os.path.join(config.RAW_DATA_PATH, f'{stock_code}_price_raw.csv')
    comment_path = os.path.join(config.RAW_DATA_PATH, f'{stock_code}_comment_raw.csv')

    if not os.path.exists(price_path) or not os.path.exists(comment_path):
        print(f"[错误] 原始数据文件不存在。请先运行 prepare_data.py 来生成数据。")
        print(f"  - 期望的价格文件: {price_path}")
        print(f"  - 期望的评论文件: {comment_path}")
        return None, None

    try:
        raw_price_df = pd.read_csv(price_path)
        raw_comment_df = pd.read_csv(comment_path)
        print("原始数据加载成功。")
        return raw_price_df, raw_comment_df
    except Exception as e:
        print(f"[错误] 加载原始数据文件时出错: {e}")
        return None, None


def preprocess_and_analyze_sentiment(raw_price_df, raw_comment_df, stock_code):
    """预处理数据并进行情感分析，并保存中间结果。"""
    print("--- 步骤 2: 数据预处理与情感分析 ---")
    saver = DataSaver()

    # 2.1 数据预处理
    preprocessor = DataPreprocessor()
    processed_price_df = preprocessor.process_prices(raw_price_df.copy())
    saver.save_to_csv(processed_price_df.reset_index(), config.PROCESSED_DATA_PATH, f'{stock_code}_price_processed.csv')

    processed_comment_df = preprocessor.process_comments(raw_comment_df.copy())
    saver.save_to_csv(processed_comment_df, config.PROCESSED_DATA_PATH, f'{stock_code}_comments_processed.csv')

    if processed_comment_df.empty:
        print("错误：评论数据预处理后为空。")
        return None, None

    # 2.2 情感分析
    sentiment_analyzer = SentimentAnalyzer(model_path=config.SENTIMENT_MODEL_PATH)
    processed_comment_df['sentiment_score'] = sentiment_analyzer.analyze(list(processed_comment_df['cleaned_text']))
    saver.save_to_csv(processed_comment_df, config.PROCESSED_DATA_PATH, f'{stock_code}_comments_with_sentiment.csv')

    daily_sentiment_df = sentiment_analyzer.aggregate_sentiment_daily(processed_comment_df)
    saver.save_to_csv(daily_sentiment_df.reset_index(), config.PROCESSED_DATA_PATH, f'{stock_code}_daily_sentiment.csv')

    if daily_sentiment_df.empty:
        print("错误：没有可分析的情感数据。")
        return None, None

    print("数据预处理和情感分析完成，中间文件已保存。")
    return processed_price_df, daily_sentiment_df


def perform_correlation_analysis(processed_price_df, daily_sentiment_df, stock_code):
    """执行相关性分析，并保存中间结果。"""
    print("--- 步骤 3: 相关性分析 ---")
    saver = DataSaver()
    engine = CorrelationEngine()
    price_return_series = processed_price_df['pct_change']
    sentiment_series = daily_sentiment_df['daily_sentiment_score']

    aligned_return, aligned_sentiment = engine.align_series(price_return_series, sentiment_series)

    aligned_df = pd.DataFrame({'aligned_return': aligned_return, 'aligned_sentiment': aligned_sentiment})
    saver.save_to_csv(aligned_df.reset_index(), config.PROCESSED_DATA_PATH, f'{stock_code}_aligned_return_sentiment.csv')

    if aligned_return.empty or aligned_sentiment.empty:
        print("错误：价格和情感数据对齐后为空，无法进行相关性分析。")
        return None, None, None

    overall_corr = aligned_return.corr(aligned_sentiment)
    with open(os.path.join(config.PROCESSED_DATA_PATH, f'{stock_code}_overall_correlation.txt'), 'w') as f:
        f.write(str(overall_corr))
    print(f"Overall correlation saved.")

    rolling_corr = engine.calculate_rolling_correlation(aligned_return, aligned_sentiment, window=30).dropna()
    saver.save_to_csv(rolling_corr.to_frame(name='rolling_corr').reset_index(), config.PROCESSED_DATA_PATH, f'{stock_code}_rolling_correlation.csv')

    lagged_corr = engine.calculate_lagged_correlation(aligned_return, aligned_sentiment, max_lag=10)
    saver.save_to_csv(lagged_corr.to_frame(name='lagged_corr').reset_index(), config.PROCESSED_DATA_PATH, f'{stock_code}_lagged_correlation.csv')

    print("相关性分析完成，中间文件已保存。")
    return overall_corr, rolling_corr, lagged_corr


def generate_final_report(processed_price_df, daily_sentiment_df, corr_metrics):
    """生成可视化图表和最终的HTML报告。"""
    print("--- 步骤 4: 生成分析报告 ---")
    # 4.1 数据可视化
    visualizer = Visualizer()
    engine = CorrelationEngine()  # For alignment

    price_close_series = processed_price_df['close']
    sentiment_series = daily_sentiment_df['daily_sentiment_score']
    aligned_price, aligned_sentiment_for_plot = engine.align_series(price_close_series, sentiment_series)
    aligned_return, aligned_sentiment_for_scatter = engine.align_series(processed_price_df['pct_change'],
                                                                        sentiment_series)

    figures = {
        'price_sentiment': visualizer.create_sentiment_vs_price_fig(aligned_price, aligned_sentiment_for_plot,
                                                                    config.STOCK_NAME),
        'scatterplot': visualizer.create_correlation_scatterplot_fig(aligned_sentiment_for_scatter, aligned_return,
                                                                     config.STOCK_NAME),
        'rolling_corr': visualizer.create_rolling_correlation_fig(corr_metrics['rolling'], config.STOCK_NAME,
                                                                  window=30),
        'lagged_corr': visualizer.create_lagged_correlation_fig(corr_metrics['lagged'], config.STOCK_NAME)
    }

    # 4.2 准备报告上下文
    overall_corr = corr_metrics['overall']
    lagged_corr = corr_metrics['lagged']
    lead_corr = lagged_corr[lagged_corr.index > 0]
    lag_corr_only = lagged_corr[lagged_corr.index < 0]

    report_context = {
        'title': f'{config.STOCK_NAME} 情感与价格相关性分析报告',
        'stock_name': config.STOCK_NAME,
        'stock_code': config.STOCK_CODE,
        'start_date': config.START_DATE,
        'end_date': config.END_DATE,
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'corr_metrics': {
            'overall': overall_corr,
            'max_lead_corr': lead_corr.max() if not lead_corr.empty and lead_corr.notna().any() else None,
            'max_lead_day': lead_corr.idxmax() if not lead_corr.empty and lead_corr.notna().any() else 'N/A',
            'max_lag_corr': lag_corr_only.max() if not lag_corr_only.empty and lag_corr_only.notna().any() else None,
            'max_lag_day': lag_corr_only.idxmax() if not lag_corr_only.empty and lag_corr_only.notna().any() else 'N/A',
        },
        'figures': figures
    }

    # 4.3 生成HTML报告
    report_generator = ReportGenerator(report_path=config.REPORT_PATH, template_str=HTML_TEMPLATE)
    report_generator.generate_report(config.HTML_REPORT_FILENAME, report_context)
    print("报告生成完毕。")


def main():
    """主函数，编排整个分析流程。"""
    print("===== 开始执行股票评论与价格相关性分析项目 (网页报告版) =====\n")

    # 步骤 1: 从文件加载预先准备好的数据
    raw_price_df, raw_comment_df = load_raw_data(stock_code=config.STOCK_CODE)
    if raw_price_df is None or raw_comment_df is None:
        print("\n[失败] 数据加载阶段出错，项目终止。")
        return

    # 步骤 2: 预处理和情感分析
    processed_price_df, daily_sentiment_df = preprocess_and_analyze_sentiment(raw_price_df, raw_comment_df, config.STOCK_CODE)
    if processed_price_df is None or daily_sentiment_df is None:
        print("\n[失败] 数据处理或情感分析阶段出错，项目终止。")
        return

    # 步骤 3: 相关性分析
    overall_corr, rolling_corr, lagged_corr = perform_correlation_analysis(processed_price_df, daily_sentiment_df, config.STOCK_CODE)
    if overall_corr is None:
        print("\n[失败] 相关性分析阶段出错，项目终止。")
        return

    # 保存相关性分析结果api
    correlation_metrics = {
        'overall': overall_corr,
        'rolling': rolling_corr,
        'lagged': lagged_corr
    }

    # 步骤 4: 生成报告
    generate_final_report(processed_price_df, daily_sentiment_df, correlation_metrics)

    print("\n===== 项目执行完毕！请在 reports 目录下查看 stock_analysis_report.html =====\n")


if __name__ == '__main__':
    main()
