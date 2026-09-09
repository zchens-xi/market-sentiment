# 金融社交媒体情绪挖掘与股票市场关联分析

面向股票行情与投资者社区文本，构建从数据采集、文本清洗、情绪识别、日级情绪指标构建，到市场关联分析、时序建模与可视化展示的一体化分析流程。

## 项目功能

- **数据采集**：使用 Tushare 获取股票行情，并通过东方财富股吧接口采集投资者帖子/评论。
- **增量处理**：对价格与评论数据进行增量更新、去重、时间对齐与结构化处理。
- **情绪分析**：使用中文金融领域预训练模型进行情绪识别，并构建日级情绪指标。
- **关联分析**：分析投资者情绪与股票收益、价格变化之间的同步与滞后关系。
- **高级分析**：包含词频/词云、盘中情绪模式、极端情绪事件、情绪动量与分歧等分析。
- **预测建模**：包含 SARIMAX、LSTM、LSTNet 等时序预测实验与模型对比。
- **可视化应用**：基于 Flask + Plotly 提供交互式分析页面与报告展示。

## 项目结构

```text
.
├── data_ingestion.py              # 行情与评论数据采集
├── data_preprocessing.py          # 数据清洗与增量预处理
├── sentiment_analysis.py          # 情绪识别与日级聚合
├── feature_engineering.py         # 特征工程
├── correlation_analysis.py        # 情绪—市场关联分析
├── temporal_sentiment_analysis.py # 时序情绪分析
├── advanced_sentiment_analysis.py # 高级情绪分析
├── predictive_models.py           # SARIMAX/LSTM/LSTNet 等预测模型
├── word_analyse.py                # 词频分析
├── wordcloud_generator.py         # 词云生成
├── report_generator.py            # 报告生成
├── simplified_enhanced_app.py     # Flask 可视化应用
├── templates/                     # Web 页面模板
├── reports/                       # 文本分析报告
├── 数据分析.ipynb                 # 数据分析 Notebook
└── 报告.md                        # 项目说明/报告
```

## 安装

```bash
pip install -r requirements.txt
```

## 重要：凭据未上传

出于安全与隐私考虑，**本公开仓库不包含真实的 Tushare Token 和东方财富股吧 Cookie**。因此，直接克隆仓库后，涉及在线数据采集的功能在未配置凭据时无法正常访问对应接口。

需要在本地自行准备：

- `TUSHARE_TOKEN`：用于通过 Tushare API 获取股票行情数据。
- `GUBA_COOKIE`：用于访问东方财富股吧接口；Cookie 可能失效，需要从自己的浏览器会话中重新获取。

仓库中的代码已改为从环境变量读取这两个值，真实凭据不会写入源码。可参考 `.env.example` 查看需要配置的变量名称。

Linux / macOS：

```bash
export TUSHARE_TOKEN='your_token'
export GUBA_COOKIE='your_cookie'
export FLASK_SECRET_KEY='your_secret_key'
```

Windows PowerShell：

```powershell
$env:TUSHARE_TOKEN='your_token'
$env:GUBA_COOKIE='your_cookie'
$env:FLASK_SECRET_KEY='your_secret_key'
```

> `.env.example` 只提供变量名示例，不包含任何真实 Token、Cookie 或密码。

## 运行

可先运行完整流程测试：

```bash
python test_complete_pipeline.py
```

启动 Web 应用：

```bash
python simplified_enhanced_app.py
```

## 数据与模型文件说明

原始评论、处理后的大规模 CSV、调试文件、训练权重以及生成的 HTML/图片报告体积较大，且部分本地数据可能包含会话信息，因此未纳入公开仓库。仓库保留核心代码、Notebook、页面模板与文本报告，可在配置凭据并准备数据后重新运行分析流程。

## 主要技术栈

Python、Pandas、Tushare、Requests、Transformers、PyTorch、scikit-learn、Plotly、Flask、Jieba、WordCloud。
