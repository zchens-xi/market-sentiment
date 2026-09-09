# -*- coding: utf-8 -*-
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import config

class AdvancedVisualizer:
    """
    高级可视化类，专门用于可视化情感分析的高级结果
    """
    
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("高级可视化器初始化完成")
    
    def create_topic_sentiment_heatmap(self, topic_sentiment_dict):
        """
        创建主题情感热力图 - 使用单色渐变
        """
        # 准备数据
        all_dates = set()
        for topic_data in topic_sentiment_dict.values():
            all_dates.update(topic_data.index)
        
        all_dates = sorted(list(all_dates))
        topics = list(topic_sentiment_dict.keys())
        
        # 创建矩阵
        heatmap_data = []
        for topic in topics:
            topic_data = topic_sentiment_dict[topic]
            row = [topic_data.get(date, 0) for date in all_dates]
            heatmap_data.append(row)
        
        # 创建热力图 - 使用单色渐变
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[str(d) for d in all_dates],
            y=topics,
            colorscale=[
                [0.0, '#f0f0f0'],      # 最浅 - 接近白色（负面或无情感）
                [0.5, '#87ceeb'],      # 中等 - 浅蓝色（中性）
                [1.0, '#1e3a8a']       # 最深 - 深蓝色（强烈正面）
            ],
            colorbar=dict(
                title=dict(
                    text="情感强度",
                    side="right"
                ),
                tickvals=[-1, 0, 1],
                ticktext=["负面", "中性", "正面"]
            ),
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>日期: %{x}<br>情感分数: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 主题情感分析热力图",
            xaxis_title="日期",
            yaxis_title="主题",
            height=500,
            xaxis=dict(tickangle=45),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_sentiment_intensity_pie_chart(self, intensity_stats):
        """
        创建情感强度饼图
        """
        # 提取数据
        labels = intensity_stats.index.tolist()
        values = intensity_stats[('sentiment_score', 'count')].tolist()
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        )])
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 情感强度分布",
            height=400
        )
        
        return fig
    
    def create_intraday_sentiment_pattern(self, intraday_pattern):
        """
        创建日内情感模式柱状图
        """
        time_periods = intraday_pattern.index.tolist()
        avg_sentiment = intraday_pattern[('sentiment_score', 'mean')].tolist()
        comment_counts = intraday_pattern[('sentiment_score', 'count')].tolist()
        
        # 创建双轴图表
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f"{config.STOCK_NAME} 日内情感模式"]
        )
        
        # 情感分数柱状图
        fig.add_trace(
            go.Bar(name="平均情感分数", x=time_periods, y=avg_sentiment, 
                  marker_color='lightblue'),
            secondary_y=False,
        )
        
        # 评论数量折线图
        fig.add_trace(
            go.Scatter(name="评论数量", x=time_periods, y=comment_counts, 
                      mode='lines+markers', line=dict(color='red')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="时段")
        fig.update_yaxes(title_text="平均情感分数", secondary_y=False)
        fig.update_yaxes(title_text="评论数量", secondary_y=True)
        
        return fig
    
    def create_sentiment_momentum_dashboard(self, momentum_df):
        """
        创建情感动量仪表板
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['每日情感分数', '情感动量', '情感变化率', '情感波动率'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        dates = momentum_df.index
        
        # 每日情感分数
        fig.add_trace(
            go.Scatter(x=dates, y=momentum_df['daily_sentiment_score'], 
                      name="每日情感", line=dict(color='blue')),
            row=1, col=1
        )
        
        # 情感动量
        fig.add_trace(
            go.Scatter(x=dates, y=momentum_df['sentiment_momentum'], 
                      name="情感动量", line=dict(color='green')),
            row=1, col=2
        )
        
        # 情感变化率
        fig.add_trace(
            go.Scatter(x=dates, y=momentum_df['sentiment_change_rate'], 
                      name="变化率", line=dict(color='orange')),
            row=2, col=1
        )
        
        # 情感波动率
        fig.add_trace(
            go.Scatter(x=dates, y=momentum_df['sentiment_volatility'], 
                      name="波动率", line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 情感动量分析仪表板",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_extreme_sentiment_events_chart(self, extreme_events_df):
        """
        创建极端情感事件图表
        """
        fig = go.Figure()
        
        dates = extreme_events_df.index
        
        # 极端正面事件
        fig.add_trace(go.Bar(
            name="极端正面事件",
            x=dates,
            y=extreme_events_df['extreme_positive_count'],
            marker_color='green',
            opacity=0.7
        ))
        
        # 极端负面事件
        fig.add_trace(go.Bar(
            name="极端负面事件",
            x=dates,
            y=-extreme_events_df['extreme_negative_count'],  # 负值显示
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 极端情感事件分布",
            xaxis_title="日期",
            yaxis_title="事件数量",
            barmode='relative',
            height=400
        )
        
        return fig
    
    def create_sentiment_divergence_chart(self, divergence_df, divergence_events):
        """
        创建情感-价格背离分析图表
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'{config.STOCK_NAME} 情感与价格走势对比', '背离事件标记'],
            shared_xaxes=True
        )
        
        dates = divergence_df.index
        
        # 标准化数据用于对比
        norm_sentiment = (divergence_df['sentiment'] - divergence_df['sentiment'].mean()) / divergence_df['sentiment'].std()
        norm_returns = (divergence_df['returns'] - divergence_df['returns'].mean()) / divergence_df['returns'].std()
        
        # 情感和价格走势
        fig.add_trace(
            go.Scatter(x=dates, y=norm_sentiment, name="标准化情感", 
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=norm_returns, name="标准化收益率", 
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # 背离事件标记
        divergence_indicator = divergence_df['is_divergence'].astype(int)
        fig.add_trace(
            go.Scatter(x=dates, y=divergence_indicator, name="背离事件", 
                      mode='markers', marker=dict(color='orange', size=8)),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 情感-价格背离分析",
            height=600
        )
        
        return fig
    
    def create_echo_chamber_analysis_chart(self, echo_chamber_df):
        """
        创建回音室效应分析图表
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['文本相似度趋势', '回音室效应评分']
        )
        
        dates = echo_chamber_df['date']
        
        # 文本相似度
        fig.add_trace(
            go.Scatter(x=dates, y=echo_chamber_df['text_similarity'], 
                      name="文本相似度", line=dict(color='blue')),
            row=1, col=1
        )
        
        # 回音室效应评分
        fig.add_trace(
            go.Scatter(x=dates, y=echo_chamber_df['echo_chamber_score'], 
                      name="回音室评分", line=dict(color='red')),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 回音室效应分析",
            height=400
        )
        
        return fig
    
    def create_sentiment_signals_chart(self, signals_df):
        """
        创建情感交易信号图表
        """
        fig = go.Figure()
        
        dates = signals_df.index
        sentiment_scores = signals_df['daily_sentiment_score']
        
        # 基础情感线
        fig.add_trace(go.Scatter(
            x=dates, y=sentiment_scores,
            mode='lines',
            name='每日情感分数',
            line=dict(color='gray', width=1)
        ))
        
        # 不同信号的点
        signal_colors = {
            '强烈看多': 'darkgreen',
            '温和看多': 'lightgreen', 
            '强烈看空': 'darkred',
            '温和看空': 'lightcoral',
            '中性': 'gray'
        }
        
        for signal, color in signal_colors.items():
            signal_data = signals_df[signals_df['sentiment_signal'] == signal]
            if not signal_data.empty:
                fig.add_trace(go.Scatter(
                    x=signal_data.index,
                    y=signal_data['daily_sentiment_score'],
                    mode='markers',
                    name=signal,
                    marker=dict(color=color, size=8)
                ))
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 情感交易信号",
            xaxis_title="日期",
            yaxis_title="情感分数",
            height=500
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, advanced_results, temporal_results):
        """
        创建综合分析仪表板
        """
        # 创建一个大的子图布局
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '主题情感分析', '情感强度分布',
                '日内情感模式', '极端情感事件',
                '情感动量分析', '交易信号分布'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # 这里可以添加各种图表到子图中...
        # 由于布局复杂，建议分别生成单独的图表
        
        fig.update_layout(
            title=f"{config.STOCK_NAME} 高级情感分析综合仪表板",
            height=900,
            showlegend=True
        )
        
        return fig

def visualize_advanced_results(advanced_results, temporal_results):
    """
    可视化高级分析结果的主函数
    """
    visualizer = AdvancedVisualizer()
    figures = {}
    
    print("开始生成高级分析可视化图表...")
    
    # 1. 主题情感热力图
    if 'topic_sentiment' in advanced_results:
        figures['topic_heatmap'] = visualizer.create_topic_sentiment_heatmap(
            advanced_results['topic_sentiment']
        )
    
    # 2. 情感强度饼图
    if 'intensity_stats' in advanced_results:
        figures['intensity_pie'] = visualizer.create_sentiment_intensity_pie_chart(
            advanced_results['intensity_stats']
        )
    
    # 3. 日内情感模式
    if 'intraday_pattern' in temporal_results:
        figures['intraday_pattern'] = visualizer.create_intraday_sentiment_pattern(
            temporal_results['intraday_pattern']
        )
    
    # 4. 情感动量仪表板
    if 'momentum_analysis' in temporal_results:
        figures['momentum_dashboard'] = visualizer.create_sentiment_momentum_dashboard(
            temporal_results['momentum_analysis']
        )
    elif 'momentum_signals' in temporal_results:
        # 使用momentum_signals作为备选方案
        figures['momentum_dashboard'] = visualizer.create_sentiment_momentum_dashboard(
            temporal_results['momentum_signals']
        )
    
    # 5. 极端情感事件
    if 'extreme_events' in advanced_results:
        figures['extreme_events'] = visualizer.create_extreme_sentiment_events_chart(
            advanced_results['extreme_events']
        )
    
    # 6. 情感-价格背离分析
    if 'divergence_df' in advanced_results and 'divergence_events' in advanced_results:
        figures['divergence_analysis'] = visualizer.create_sentiment_divergence_chart(
            advanced_results['divergence_df'], 
            advanced_results['divergence_events']
        )
    
    # 7. 回音室效应分析
    if 'echo_chamber' in temporal_results:
        figures['echo_chamber'] = visualizer.create_echo_chamber_analysis_chart(
            temporal_results['echo_chamber']
        )
    
    # 8. 情感交易信号
    if 'momentum_signals' in temporal_results:
        figures['sentiment_signals'] = visualizer.create_sentiment_signals_chart(
            temporal_results['momentum_signals']
        )
    
    print(f"生成了 {len(figures)} 个高级分析可视化图表")
    return figures

# 保存和显示图表的辅助函数
def save_and_show_figures(figures, save_path="reports/figures/"):
    """
    保存并显示所有图表
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    for name, fig in figures.items():
        # 保存为HTML文件
        fig.write_html(f"{save_path}{name}.html")
        # 显示图表
        fig.show()
        print(f"图表 {name} 已保存到 {save_path}{name}.html")

# 在你的notebook中使用
def run_advanced_visualization(advanced_results, temporal_results):
    """
    运行高级可视化的主函数
    """
    # 生成所有可视化图表
    figures = visualize_advanced_results(advanced_results, temporal_results)
    
    # 保存并显示图表
    save_and_show_figures(figures)
    
    print("高级情感分析可视化完成！")
    return figures
