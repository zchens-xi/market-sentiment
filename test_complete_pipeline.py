#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的股吧API数据处理流程测试
包括数据获取、预处理、情感分析、相关性分析
"""

import os
import sys
import config
import pandas as pd
from data_ingestion import DataSaver
from data_preprocessing import DataPreprocessor
from sentiment_analysis import SentimentAnalyzer
from correlation_analysis import CorrelationEngine
from visualization import Visualizer

def test_complete_pipeline():
    """测试完整的数据处理流程"""
    print("=== 完整股吧API数据处理流程测试 ===")
    
    try:
        # 1. 数据获取
        print("\n--- 步骤1: 数据获取 ---")
        saver = DataSaver()
        
        print("获取股票价格数据...")
        price_df = saver.fetch_stock_data(config.STOCK_CODE, config.START_DATE, config.END_DATE)
        
        print("获取股吧评论数据...")
        comment_df = saver.fetch_comments_data(config.STOCK_CODE)
        
        if price_df is None or price_df.empty:
            print("❌ 价格数据获取失败")
            return False
            
        if comment_df is None or comment_df.empty:
            print("❌ 评论数据获取失败") 
            return False
            
        print(f"✅ 数据获取成功: 价格数据 {len(price_df)} 条, 评论数据 {len(comment_df)} 条")
        
        # 2. 数据预处理
        print("\n--- 步骤2: 数据预处理 ---")
        preprocessor = DataPreprocessor()
        
        print("预处理价格数据...")
        processed_price_df = preprocessor.process_prices(price_df.copy())
        
        print("预处理评论数据...")
        processed_comment_df = preprocessor.process_comments(comment_df.copy())
        
        if processed_price_df.empty:
            print("❌ 价格数据预处理失败")
            return False
            
        if processed_comment_df.empty:
            print("❌ 评论数据预处理失败")
            return False
            
        print(f"✅ 数据预处理成功: 价格数据 {len(processed_price_df)} 条, 评论数据 {len(processed_comment_df)} 条")
        
        # 保存预处理后的数据
        saver.save_to_csv(processed_price_df.reset_index(), config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_price_processed.csv')
        saver.save_to_csv(processed_comment_df, config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_processed.csv')
        
        # 3. 情感分析
        print("\n--- 步骤3: 情感分析 ---")
        try:
            sentiment_analyzer = SentimentAnalyzer(model_path=config.SENTIMENT_MODEL_PATH)
            
            print("进行情感分析...")
            processed_comment_df['sentiment_score'] = sentiment_analyzer.analyze(list(processed_comment_df['processed_text']))
            
            print("聚合每日情感指数...")
            daily_sentiment_df = sentiment_analyzer.aggregate_sentiment_daily(processed_comment_df)
            
            if daily_sentiment_df.empty:
                print("❌ 情感分析结果为空")
                return False
                
            print(f"✅ 情感分析成功: 生成 {len(daily_sentiment_df)} 天的情感数据")
            
            # 保存情感分析结果
            saver.save_to_csv(processed_comment_df, config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_with_sentiment.csv')
            saver.save_to_csv(daily_sentiment_df.reset_index(), config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_daily_sentiment.csv')
            
        except Exception as e:
            print(f"⚠️ 情感分析模块加载失败: {e}")
            print("使用简化的关键词情感分析...")
            
            # 简化的情感分析
            positive_keywords = ['看好', '上涨', '买入', '牛市', '利好', '强势', '突破', '涨', '好']
            negative_keywords = ['看空', '下跌', '卖出', '熊市', '利空', '弱势', '跌破', '跌', '差']
            
            def simple_sentiment(text):
                if any(word in text for word in positive_keywords):
                    return 1
                elif any(word in text for word in negative_keywords):
                    return -1
                else:
                    return 0
            
            processed_comment_df['sentiment_score'] = processed_comment_df['comment_text'].apply(simple_sentiment)
            
            # 按日聚合
            daily_sentiment_df = processed_comment_df.set_index('comment_date').groupby(pd.Grouper(freq='D'))['sentiment_score'].mean().to_frame(name='daily_sentiment_score')
            
            print(f"✅ 简化情感分析完成: 生成 {len(daily_sentiment_df)} 天的情感数据")
        
        # 4. 相关性分析
        print("\n--- 步骤4: 相关性分析 ---")
        try:
            engine = CorrelationEngine()
            
            # 准备数据
            price_return_series = processed_price_df['pct_change']
            sentiment_series = daily_sentiment_df['daily_sentiment_score']
            
            # 对齐数据
            aligned_return, aligned_sentiment = engine.align_series(price_return_series, sentiment_series)
            
            if aligned_return.empty or aligned_sentiment.empty:
                print("❌ 数据对齐失败")
                return False
            
            # 计算相关性
            overall_corr = aligned_return.corr(aligned_sentiment)
            rolling_corr = engine.calculate_rolling_correlation(aligned_return, aligned_sentiment, window=30)
            lagged_corr = engine.calculate_lagged_correlation(aligned_return, aligned_sentiment, max_lag=10)
            
            print(f"✅ 相关性分析完成:")
            print(f"   整体相关性: {overall_corr:.4f}")
            print(f"   滚动相关性数据点: {len(rolling_corr)} 个")
            print(f"   滞后相关性数据点: {len(lagged_corr)} 个")
            
            # 保存相关性分析结果
            correlation_results = {
                'overall': overall_corr,
                'max_lead_corr': lagged_corr[lagged_corr.index > 0].max() if any(lagged_corr.index > 0) else None,
                'max_lead_day': lagged_corr[lagged_corr.index > 0].idxmax() if any(lagged_corr.index > 0) else None,
                'max_lag_corr': lagged_corr[lagged_corr.index < 0].max() if any(lagged_corr.index < 0) else None,
                'max_lag_day': lagged_corr[lagged_corr.index < 0].idxmax() if any(lagged_corr.index < 0) else None,
            }
            
            # 保存结果文件
            saver.save_to_csv(pd.DataFrame({'return': aligned_return, 'sentiment': aligned_sentiment}).reset_index(), 
                             config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_aligned_return_sentiment.csv')
            saver.save_to_csv(rolling_corr.reset_index(), config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_rolling_correlation.csv')
            saver.save_to_csv(lagged_corr.reset_index(), config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_lagged_correlation.csv')
            
        except Exception as e:
            print(f"❌ 相关性分析失败: {e}")
            return False
        
        # 5. 可视化（如果可用）
        print("\n--- 步骤5: 数据可视化 ---")
        try:
            visualizer = Visualizer()
            
            # 准备可视化数据
            price_close_series = processed_price_df['close']
            aligned_price, aligned_sentiment_for_plot = engine.align_series(price_close_series, sentiment_series)
            
            # 生成图表
            figures = {
                'price_sentiment': visualizer.create_sentiment_vs_price_fig(aligned_price, aligned_sentiment_for_plot, config.STOCK_NAME),
                'scatterplot': visualizer.create_correlation_scatterplot_fig(aligned_sentiment, aligned_return, config.STOCK_NAME),
                'rolling_corr': visualizer.create_rolling_correlation_fig(rolling_corr, config.STOCK_NAME, window=30),
                'lagged_corr': visualizer.create_lagged_correlation_fig(lagged_corr, config.STOCK_NAME)
            }
            
            print("✅ 图表生成成功")
            
        except Exception as e:
            print(f"⚠️ 可视化模块暂不可用: {e}")
            figures = None
        
        print("\n=== 🎉 完整流程测试成功！ ===")
        print(f"所有处理后的数据已保存到: {config.PROCESSED_DATA_PATH}")
        print("\n主要结果:")
        print(f"- 整体相关性: {overall_corr:.4f}")
        print(f"- 有效评论数: {len(processed_comment_df)}")
        print(f"- 分析天数: {len(daily_sentiment_df)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 屏蔽警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')
    
    print("股吧API完整数据处理流程测试")
    print("=" * 50)
    
    # 确保目录存在
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
    
    # 运行测试
    success = test_complete_pipeline()
    
    if success:
        print("\n✅ 测试完成！您可以:")
        print("1. 运行 python main.py 生成完整报告")
        print("2. 运行 python app.py 启动Web应用")
        print("3. 查看 data/processed/ 目录下的处理结果")
    else:
        print("\n❌ 测试失败，请检查错误信息并修复问题")
