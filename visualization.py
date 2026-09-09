import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class Visualizer:
    """
    使用Plotly创建交互式图表。
    """

    def __init__(self):
        print("交互式可视化工具初始化成功。")

    def create_sentiment_vs_price_fig(self, price_series, sentiment_series, stock_name):
        print("正在创建价格与情感对比图...")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 添加价格曲线
        fig.add_trace(
            go.Scatter(x=price_series.index, y=price_series.values, name="收盘价", line=dict(color='royalblue')),
            secondary_y=False,
        )
        # 添加情感指数曲线
        fig.add_trace(
            go.Scatter(x=sentiment_series.index, y=sentiment_series.values, name="情感指数",
                       line=dict(color='firebrick')),
            secondary_y=True,
        )

        fig.update_layout(
            title_text=f"<b>{stock_name} 股价与公众情感指数时序图</b>",
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="日期")
        fig.update_yaxes(title_text="<b>收盘价 (元)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>每日情感指数</b>", secondary_y=True)

        return fig

    def create_correlation_scatterplot_fig(self, series1, series2, stock_name):
        print("正在创建相关性散点图...")
        df = pd.DataFrame({'sentiment': series1, 'return': series2})
        fig = px.scatter(
            df, x='sentiment', y='return',
            trendline="ols",  # 添加普通最小二乘回归线
            title=f"<b>{stock_name} 情感指数与日收益率相关性散点图</b>",
            labels={'sentiment': '每日情感指数', 'return': '日收益率 (%)'},
            template='plotly_white'
        )
        return fig

    def create_rolling_correlation_fig(self, rolling_corr_series, stock_name, window):
        print("正在创建滚动相关性图...")
        fig = px.line(
            rolling_corr_series,
            title=f'<b>{stock_name} {window}日滚动相关性</b>',
            labels={'index': '日期', 'value': '皮尔逊相关系数'},
            template='plotly_white'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        return fig

    def create_lagged_correlation_fig(self, lagged_corr_series, stock_name):
        print("正在创建滞后相关性图...")
        fig = px.bar(
            lagged_corr_series,
            title=f'<b>{stock_name} 情感与价格的滞后相关性分析</b>',
            labels={'index': '情感指数领先/滞后天数 (正数为领先)', 'value': '皮尔逊相关系数'},
            template='plotly_white'
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        return fig
