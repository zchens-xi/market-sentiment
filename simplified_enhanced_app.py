# -*- coding: utf-8 -*-
"""
简化版增强股票分析Web应用
基于已验证的完整流程，确保稳定运行
"""

import os
import sys
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 导入项目模块
import config
from data_ingestion import PriceDataFetcher, DataSaver
from data_preprocessing import DataPreprocessor
from sentiment_analysis import SentimentAnalyzer
from correlation_analysis import CorrelationEngine
from word_analyse import calculate_word_frequencies, calculate_sentiment_word_frequencies
# 移除未使用的导入: from visualization import Visualizer

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-only-change-me')

# 全局变量存储最新数据
latest_data = {
    'timestamp': None,
    'data': None,
    'is_updating': False,
    'error': None
}

class SimplifiedDataManager:
    """简化版数据管理器，使用已验证的完整流程逻辑"""
    
    def __init__(self):
        self.data_saver = DataSaver()
        self.preprocessor = DataPreprocessor()
        
    def run_complete_analysis(self, force_refresh=False):
        """运行完整的数据分析流程 - 基于test_complete_pipeline.py的逻辑"""
        try:
            print("=== 简化版完整数据分析流程 ===")
            
            # 1. 数据获取
            print("--- 步骤1: 数据获取 ---")
            raw_comment_path = os.path.join(config.RAW_DATA_PATH, f'{config.STOCK_CODE}_comment_raw.csv')
            raw_price_path = os.path.join(config.RAW_DATA_PATH, f'{config.STOCK_CODE}_price_raw.csv')
            
            if force_refresh or not os.path.exists(raw_comment_path):
                print("获取新数据...")
                # 获取价格数据
                price_fetcher = PriceDataFetcher(config.TUSHARE_TOKEN)
                price_data = price_fetcher.get_data(config.STOCK_CODE, config.START_DATE, config.END_DATE)
                
                # 获取评论数据
                comment_data = self.data_saver.fetch_comments_data(config.STOCK_CODE)
                print(f"✅ 数据获取成功: 价格数据 {len(price_data)} 条, 评论数据 {len(comment_data)} 条")
            else:
                print("使用现有数据...")
                
            # 2. 数据预处理
            print("--- 步骤2: 数据预处理 ---")
            processed_price_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_price_processed.csv')
            processed_comment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_processed.csv')
            
            if force_refresh or not os.path.exists(processed_price_path):
                self.preprocessor.preprocess_price_data(config.STOCK_CODE)
            if force_refresh or not os.path.exists(processed_comment_path):
                self.preprocessor.preprocess_comment_data(config.STOCK_CODE)
            
            # 读取预处理后的数据
            processed_price_df = pd.read_csv(processed_price_path)
            processed_comment_df = pd.read_csv(processed_comment_path)
            print(f"✅ 数据预处理成功: 价格数据 {len(processed_price_df)} 条, 评论数据 {len(processed_comment_df)} 条")
            
            # 3. 情感分析
            print("--- 步骤3: 情感分析 ---")
            daily_sentiment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_daily_sentiment.csv')
            
            if force_refresh or not os.path.exists(daily_sentiment_path):
                try:
                    # 尝试深度情感分析
                    print("正在加载情感分析模型...")
                    sentiment_analyzer = SentimentAnalyzer(model_path=config.SENTIMENT_MODEL_PATH)
                    processed_comment_df['sentiment_score'] = sentiment_analyzer.analyze(list(processed_comment_df['processed_text']))
                    daily_sentiment_df = sentiment_analyzer.aggregate_sentiment_daily(processed_comment_df)
                    print("✅ 深度情感分析完成")
                except Exception as e:
                    print(f"⚠️ 情感分析模块加载失败: {e}")
                    print("使用简化的关键词情感分析...")
                    
                    # 简化情感分析
                    def simple_sentiment(text):
                        positive_words = ['上涨', '涨', '好', '买入', '看好', '牛', '强', '多', '利好']
                        negative_words = ['下跌', '跌', '坏', '卖出', '看空', '熊', '弱', '空', '利空']
                        
                        pos_count = sum(1 for word in positive_words if word in str(text))
                        neg_count = sum(1 for word in negative_words if word in str(text))
                        
                        if pos_count > neg_count:
                            return 0.8
                        elif neg_count > pos_count:
                            return 0.2
                        else:
                            return 0.5
                    
                    processed_comment_df['sentiment_score'] = processed_comment_df['comment_text'].apply(simple_sentiment)
                    
                    # 按日期聚合情感数据
                    processed_comment_df['timestamp'] = pd.to_datetime(processed_comment_df['timestamp'])
                    processed_comment_df['trade_date'] = processed_comment_df['timestamp'].dt.date
                    
                    daily_sentiment_df = processed_comment_df.groupby('trade_date').agg({
                        'sentiment_score': 'mean'
                    }).reset_index()
                    daily_sentiment_df.columns = ['trade_date', 'daily_sentiment_score']
                    
                    print(f"✅ 简化情感分析完成: 生成 {len(daily_sentiment_df)} 天的情感数据")
                
                # 保存情感分析结果
                daily_sentiment_df.to_csv(daily_sentiment_path, index=False, encoding='utf-8-sig')
            else:
                daily_sentiment_df = pd.read_csv(daily_sentiment_path)
                print(f"使用现有情感分析结果: {len(daily_sentiment_df)} 天")
            
            # 4. 相关性分析
            print("--- 步骤4: 相关性分析 ---")
            overall_corr_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_overall_correlation.txt')
            
            if force_refresh or not os.path.exists(overall_corr_path):
                engine = CorrelationEngine()
                
                # 准备数据
                price_return_series = processed_price_df['pct_change']
                sentiment_series = daily_sentiment_df['daily_sentiment_score']
                
                # 对齐数据
                aligned_return, aligned_sentiment = engine.align_series(price_return_series, sentiment_series)
                
                if not aligned_return.empty and not aligned_sentiment.empty:
                    # 计算相关性
                    overall_corr = aligned_return.corr(aligned_sentiment)
                    rolling_corr = engine.calculate_rolling_correlation(aligned_return, aligned_sentiment, window=30)
                    lagged_corr = engine.calculate_lagged_correlation(aligned_return, aligned_sentiment, max_lag=10)
                    
                    print(f"✅ 相关性分析完成:")
                    print(f"   整体相关性: {overall_corr:.4f}")
                    print(f"   滚动相关性数据点: {len(rolling_corr)} 个")
                    print(f"   滞后相关性数据点: {len(lagged_corr)} 个")
                    
                    # 保存结果
                    with open(overall_corr_path, 'w', encoding='utf-8') as f:
                        f.write(str(overall_corr))
                    
                    # 保存对齐的数据 - 修复列名映射
                    aligned_df = pd.DataFrame({
                        'trade_date': aligned_return.index,
                        'return': aligned_return.values,
                        'sentiment': aligned_sentiment.values  # 修正列名为sentiment而不是sentiment_score
                    })
                    aligned_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_aligned_return_sentiment.csv')
                    aligned_df.to_csv(aligned_path, index=False, encoding='utf-8-sig')
                    
                    # 保存滚动相关性
                    rolling_df = pd.DataFrame({
                        'trade_date': rolling_corr.index,
                        'rolling_correlation': rolling_corr.values
                    })
                    rolling_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_rolling_correlation.csv')
                    rolling_df.to_csv(rolling_path, index=False, encoding='utf-8-sig')
                    
                    # 保存滞后相关性
                    lagged_df = pd.DataFrame({
                        'lag': lagged_corr.index,
                        'correlation': lagged_corr.values
                    })
                    lagged_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_lagged_correlation.csv')
                    lagged_df.to_csv(lagged_path, index=False, encoding='utf-8-sig')
                    
                else:
                    print("❌ 数据对齐失败")
                    # 创建默认的相关性文件
                    with open(overall_corr_path, 'w', encoding='utf-8') as f:
                        f.write('0.0')
            
            # 5. 生成分析结果
            analysis_data = self._generate_analysis_results()
            print("=== 🎉 简化版完整分析流程成功！ ===")
            return analysis_data
            
        except Exception as e:
            print(f"❌ 分析流程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': f'分析失败: {str(e)}'}
    
    def _generate_analysis_results(self):
        """生成分析结果"""
        try:
            # 读取相关性分析结果
            overall_corr_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_overall_correlation.txt')
            correlation_results = {'overall': 0.0}
            
            if os.path.exists(overall_corr_path):
                try:
                    with open(overall_corr_path, 'r', encoding='utf-8') as f:
                        correlation_results['overall'] = float(f.read().strip())
                except:
                    correlation_results['overall'] = 0.0
            
            # 读取滞后相关性数据
            lagged_corr_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_lagged_correlation.csv')
            if os.path.exists(lagged_corr_path):
                lagged_df = pd.read_csv(lagged_corr_path)
                if not lagged_df.empty and len(lagged_df.columns) >= 2:
                    # 检查列名并重命名
                    if 'correlation' not in lagged_df.columns:
                        lagged_df.columns = ['lag', 'correlation']
                    
                    # 移除空值
                    lagged_df = lagged_df.dropna()
                    
                    if not lagged_df.empty:
                        # 确保数据类型正确
                        lagged_df['correlation'] = pd.to_numeric(lagged_df['correlation'], errors='coerce')
                        lagged_df['lag'] = pd.to_numeric(lagged_df['lag'], errors='coerce')
                        lagged_df = lagged_df.dropna()
                        
                        if not lagged_df.empty:
                            max_corr_idx = lagged_df['correlation'].abs().idxmax()
                            max_corr_value = lagged_df.loc[max_corr_idx, 'correlation']
                            max_lag_value = lagged_df.loc[max_corr_idx, 'lag']
                            
                            # 安全的类型转换
                            try:
                                correlation_results['max_lead_corr'] = float(max_corr_value)
                                correlation_results['max_lead_day'] = int(max_lag_value)
                            except (ValueError, TypeError):
                                correlation_results.update({'max_lead_corr': 0.0, 'max_lead_day': 0})
                        else:
                            correlation_results.update({'max_lead_corr': 0.0, 'max_lead_day': 0})
                    else:
                        correlation_results.update({'max_lead_corr': 0.0, 'max_lead_day': 0})
                else:
                    correlation_results.update({'max_lead_corr': 0.0, 'max_lead_day': 0})
            else:
                correlation_results.update({'max_lead_corr': 0.0, 'max_lead_day': 0})
            
            # 生成图表
            figures = self._generate_interactive_charts()
            
            # 获取统计信息
            stats = self._get_comprehensive_statistics()
            
            return {
                'title': f'{config.STOCK_NAME}股票情感分析报告',
                'stock_name': config.STOCK_NAME,
                'stock_code': config.STOCK_CODE,
                'start_date': config.START_DATE,
                'end_date': config.END_DATE,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'corr_metrics': correlation_results,
                'figures': figures,
                'statistics': stats,
                'error': None
            }
            
        except Exception as e:
            print(f"生成分析结果时出错: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'生成结果失败: {str(e)}'}
    
    def _generate_interactive_charts(self):
        """生成交互式图表"""
        figures = {}
        
        try:
            print("开始生成交互式图表...")
            
            # 1. 价格与情感对比图
            aligned_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_aligned_return_sentiment.csv')
            print(f"检查文件: {aligned_path}")
            if os.path.exists(aligned_path):
                df = pd.read_csv(aligned_path)
                print(f"读取对齐数据: {len(df)} 行, 列名: {list(df.columns)}")
                
                # 检查并修正列名
                if 'index' in df.columns and 'trade_date' not in df.columns:
                    df = df.rename(columns={'index': 'trade_date'})
                    print("重命名列名: index -> trade_date")
                
                # 确保情感列名正确
                if 'sentiment_score' in df.columns and 'sentiment' not in df.columns:
                    df = df.rename(columns={'sentiment_score': 'sentiment'})
                    print("重命名列名: sentiment_score -> sentiment")
                
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    
                    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                    fig1.add_trace(
                        go.Scatter(x=df['trade_date'], y=df['return'], name='收益率', line=dict(color='#1f77b4')),
                        secondary_y=False
                    )
                    fig1.add_trace(
                        go.Scatter(x=df['trade_date'], y=df['sentiment'], name='情感得分', line=dict(color='#ff7f0e')),
                        secondary_y=True
                    )
                    fig1.update_xaxes(title_text="日期")
                    fig1.update_yaxes(title_text="收益率", secondary_y=False)
                    fig1.update_yaxes(title_text="情感得分", secondary_y=True)
                    fig1.update_layout(title="价格收益率与情感得分对比", height=400, template="plotly_white")
                    figures['price_sentiment'] = fig1.to_json()
                    print("✅ 价格与情感对比图生成成功")
                    
                    # 2. 相关性散点图
                    fig2 = px.scatter(df, x='sentiment', y='return', 
                                    title='情感得分与收益率相关性散点图',
                                    labels={'sentiment': '情感得分', 'return': '收益率'},
                                    template="plotly_white")
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig2.update_layout(height=400)
                    figures['scatterplot'] = fig2.to_json()
                    print("✅ 相关性散点图生成成功")
            else:
                print(f"❌ 对齐数据文件不存在: {aligned_path}")
            
            # 3. 滚动相关性图
            rolling_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_rolling_correlation.csv')
            print(f"检查滚动相关性文件: {rolling_path}")
            if os.path.exists(rolling_path):
                rolling_df = pd.read_csv(rolling_path)
                print(f"读取滚动相关性数据: {len(rolling_df)} 行, 列名: {list(rolling_df.columns)}")
                if not rolling_df.empty and len(rolling_df.columns) >= 2:
                    # 检查并修正列名
                    if rolling_df.columns[0] in ['index', 'Unnamed: 0']:
                        rolling_df.columns = ['trade_date', 'rolling_correlation']
                        print("重命名滚动相关性列名")
                    
                    # 移除空值并确保数据类型正确
                    rolling_df = rolling_df.dropna()
                    print(f"清理后滚动相关性数据: {len(rolling_df)} 行")
                    if not rolling_df.empty:
                        rolling_df['trade_date'] = pd.to_datetime(rolling_df['trade_date'])
                        rolling_df['rolling_correlation'] = pd.to_numeric(rolling_df['rolling_correlation'], errors='coerce')
                        rolling_df = rolling_df.dropna()
                        
                        if not rolling_df.empty:
                            fig3 = go.Figure()
                            fig3.add_trace(go.Scatter(x=rolling_df['trade_date'], y=rolling_df['rolling_correlation'], 
                                                    name='30天滚动相关性', line=dict(color='#2ca02c')))
                            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig3.update_layout(title="30天滚动相关性变化", xaxis_title="日期", yaxis_title="相关性", 
                                             height=400, template="plotly_white")
                            figures['rolling_corr'] = fig3.to_json()
                            print("✅ 滚动相关性图生成成功")
                        else:
                            print("❌ 滚动相关性数据为空（清理后）")
                else:
                    print("❌ 滚动相关性数据格式不正确")
            else:
                print(f"❌ 滚动相关性文件不存在: {rolling_path}")
            
            # 4. 滞后相关性图
            lagged_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_lagged_correlation.csv')
            print(f"检查滞后相关性文件: {lagged_path}")
            if os.path.exists(lagged_path):
                lagged_df = pd.read_csv(lagged_path)
                print(f"读取滞后相关性数据: {len(lagged_df)} 行, 列名: {list(lagged_df.columns)}")
                if not lagged_df.empty and len(lagged_df.columns) >= 2:
                    # 检查并修正列名
                    if lagged_df.columns[0] in ['index', 'Unnamed: 0']:
                        lagged_df.columns = ['lag', 'correlation']
                        print("重命名滞后相关性列名")
                    elif 'correlation' not in lagged_df.columns:
                        lagged_df.columns = ['lag', 'correlation']
                    
                    # 确保数据类型正确并移除空值
                    lagged_df['correlation'] = pd.to_numeric(lagged_df['correlation'], errors='coerce')
                    lagged_df['lag'] = pd.to_numeric(lagged_df['lag'], errors='coerce')
                    lagged_df = lagged_df.dropna()
                    print(f"清理后滞后相关性数据: {len(lagged_df)} 行")
                    
                    if not lagged_df.empty:
                        # 确保lag列是整数
                        lagged_df['lag'] = lagged_df['lag'].astype(int)
                        # 按lag排序
                        lagged_df = lagged_df.sort_values('lag')
                        
                        fig4 = go.Figure()
                        fig4.add_trace(go.Bar(x=lagged_df['lag'], y=lagged_df['correlation'], 
                                            name='滞后相关性', marker_color='#ff7f0e'))
                        fig4.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig4.update_layout(title="不同滞后天数的相关性", xaxis_title="滞后天数", yaxis_title="相关性", 
                                         height=400, template="plotly_white")
                        figures['lagged_corr'] = fig4.to_json()
                        print("✅ 滞后相关性图生成成功")
                    else:
                        print("❌ 滞后相关性数据为空（清理后）")
                else:
                    print("❌ 滞后相关性数据格式不正确")
            else:
                print(f"❌ 滞后相关性文件不存在: {lagged_path}")
            
            # 如果没有图表数据，提供占位符
            chart_names = ['price_sentiment', 'scatterplot', 'rolling_corr', 'lagged_corr']
            for name in chart_names:
                if name not in figures:
                    figures[name] = None  # 设置为None，让前端处理
                    print(f"⚠️ 图表 {name} 未生成，设置为None")
            
            print(f"图表生成完成，成功生成 {len([k for k, v in figures.items() if v is not None])} 个图表")
        
        except Exception as e:
            print(f"生成图表时出错: {e}")
            # 提供备用图表
            chart_names = ['price_sentiment', 'scatterplot', 'rolling_corr', 'lagged_corr']
            for name in chart_names:
                figures[name] = None  # 设置为None，让前端处理错误显示
        
        return figures
    
    def _get_comprehensive_statistics(self):
        """获取全面的统计信息"""
        stats = {}
        
        try:
            # 情感数据统计
            sentiment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_daily_sentiment.csv')
            if os.path.exists(sentiment_path):
                sentiment_df = pd.read_csv(sentiment_path)
                stats.update({
                    'total_days': len(sentiment_df),
                    'avg_sentiment': float(sentiment_df['daily_sentiment_score'].mean()),
                    'sentiment_std': float(sentiment_df['daily_sentiment_score'].std())
                })
            
            # 价格数据统计
            price_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_price_processed.csv')
            if os.path.exists(price_path):
                price_df = pd.read_csv(price_path)
                if 'pct_change' in price_df.columns:
                    price_df['pct_change'] = pd.to_numeric(price_df['pct_change'], errors='coerce')
                    stats.update({
                        'price_volatility': float(price_df['pct_change'].std()),
                        'max_return': float(price_df['pct_change'].max()),
                        'min_return': float(price_df['pct_change'].min()),
                        'avg_return': float(price_df['pct_change'].mean())
                    })
            
            # 评论数据统计
            comment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_processed.csv')
            if os.path.exists(comment_path):
                comment_df = pd.read_csv(comment_path)
                stats.update({
                    'total_comments': len(comment_df),
                    'avg_comment_length': float(comment_df['comment_text'].str.len().mean()) if 'comment_text' in comment_df.columns else 0
                })
                
        except Exception as e:
            print(f"获取统计信息时出错: {e}")
            # 提供默认值
            stats = {
                'total_days': 0,
                'avg_sentiment': 0.5,
                'sentiment_std': 0.0,
                'price_volatility': 0.0,
                'max_return': 0.0,
                'min_return': 0.0,
                'avg_return': 0.0,
                'total_comments': 0,
                'avg_comment_length': 0
            }
            
        return stats

# 创建数据管理器实例
data_manager = SimplifiedDataManager()

def update_data_background():
    """后台数据更新任务"""
    global latest_data
    
    while True:
        try:
            # 每30分钟更新一次数据
            if (latest_data['timestamp'] is None or 
                datetime.now() - latest_data['timestamp'] > timedelta(minutes=30)):
                
                if not latest_data['is_updating']:
                    latest_data['is_updating'] = True
                    print("开始后台数据更新...")
                    
                    new_data = data_manager.run_complete_analysis()
                    
                    latest_data['data'] = new_data
                    latest_data['timestamp'] = datetime.now()
                    latest_data['error'] = new_data.get('error')
                    latest_data['is_updating'] = False
                    
                    print("后台数据更新完成")
            
            time.sleep(600)  # 每10分钟检查一次
            
        except Exception as e:
            print(f"后台更新出错: {e}")
            latest_data['is_updating'] = False
            latest_data['error'] = str(e)
            time.sleep(300)  # 出错时等待5分钟再试

@app.route('/')
def index():
    """主页面"""
    return render_template('enhanced_index.html')

@app.route('/data')
def get_data():
    """获取分析数据API"""
    global latest_data
    
    # 如果没有缓存数据，立即获取
    if latest_data['data'] is None and not latest_data['is_updating']:
        latest_data['is_updating'] = True
        try:
            new_data = data_manager.run_complete_analysis()
            latest_data['data'] = new_data
            latest_data['timestamp'] = datetime.now()
            latest_data['error'] = new_data.get('error')
        finally:
            latest_data['is_updating'] = False
    
    return jsonify(latest_data['data'] or {'error': '数据暂时不可用'})

@app.route('/refresh')
def force_refresh():
    """强制刷新数据"""
    global latest_data
    
    if latest_data['is_updating']:
        return jsonify({'error': '数据正在更新中，请稍候'})
    
    latest_data['is_updating'] = True
    try:
        new_data = data_manager.run_complete_analysis(force_refresh=True)
        latest_data['data'] = new_data
        latest_data['timestamp'] = datetime.now()
        latest_data['error'] = new_data.get('error')
        return jsonify(new_data)
    except Exception as e:
        latest_data['error'] = str(e)
        return jsonify({'error': f'刷新失败: {str(e)}'})
    finally:
        latest_data['is_updating'] = False

@app.route('/status')
def get_status():
    """获取系统状态"""
    global latest_data
    
    return jsonify({
        'is_updating': latest_data['is_updating'],
        'last_update': latest_data['timestamp'].isoformat() if latest_data['timestamp'] else None,
        'has_data': latest_data['data'] is not None,
        'error': latest_data['error']
    })

@app.route('/quick_analysis')
def quick_analysis():
    """快速分析（仅使用现有数据）"""
    try:
        # 仅使用现有数据生成结果
        analysis_data = data_manager._generate_analysis_results()
        return jsonify(analysis_data)
    except Exception as e:
        return jsonify({'error': f'快速分析失败: {str(e)}'})

@app.route('/word_analysis')
def word_analysis():
    """词频分析API"""
    try:
        # 读取评论数据
        comment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_processed.csv')
        
        if not os.path.exists(comment_path):
            return jsonify({'error': '评论数据不存在，请先运行完整分析'})
        
        df = pd.read_csv(comment_path)
        
        if df.empty:
            return jsonify({'error': '评论数据为空'})
        
        # 基础词频分析
        word_freq = calculate_word_frequencies(df, 'processed_text' if 'processed_text' in df.columns else 'comment_text', 
                                             min_freq=2, top_n=30)
        
        result = {
            'total_comments': len(df),
            'word_frequencies': word_freq.to_dict('records'),
            'sentiment_analysis': None
        }
        
        # 如果有情感得分，进行情感词频分析
        if 'sentiment_score' in df.columns:
            try:
                pos_words, neg_words = calculate_sentiment_word_frequencies(
                    df, 'processed_text' if 'processed_text' in df.columns else 'comment_text', 
                    'sentiment_score', threshold=0.6
                )
                
                result['sentiment_analysis'] = {
                    'positive_words': pos_words.head(15).to_dict('records'),
                    'negative_words': neg_words.head(15).to_dict('records')
                }
            except Exception as e:
                print(f"情感词频分析失败: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'词频分析失败: {str(e)}'})

if __name__ == '__main__':
    print("启动简化版增强股票分析Web应用...")
    print("基于已验证的完整流程逻辑，确保稳定运行")
    print("\n主要功能:")
    print("- 股吧API数据采集")
    print("- 智能情感分析（深度学习+关键词降级）")
    print("- 交互式图表展示")
    print("- 相关性分析")
    print("\n访问地址:")
    print("- 主页: http://localhost:5000")
    print("- 快速分析: http://localhost:5000/quick_analysis")
    print("- 强制刷新: http://localhost:5000/refresh")
    print("- 状态检查: http://localhost:5000/status")
    
    # 启动后台数据更新线程
    background_thread = threading.Thread(target=update_data_background, daemon=True)
    background_thread.start()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
