# 金融社交媒体情绪挖掘与股票市场关联分析

面向雪球/东方财富股吧等投资者社区与股票行情数据，构建从数据采集、文本清洗、情绪识别、日级情绪指标构建，到市场关联分析、时序建模与可视化展示的一体化分析流程。

## 项目功能

- **数据采集**：使用 Tushare 获取股票行情，通过东方财富股吧接口采集投资者帖子/评论。
- **增量处理**：对价格与评论数据进行增量更新、去重、时间对齐与结构化处理。
- **情绪分析**：使用中文金融领域预训练模型进行情绪识别，并结合评论信息构建日级情绪指标。
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
├── reports/                       # 文本分析报告与统计结果
└── data/                          # 本地数据目录（大文件不纳入仓库）
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置

为避免泄露个人凭据，API Token 与 Cookie 不写入源码。运行前配置环境变量：

```bash
# Linux / macOS
export TUSHARE_TOKEN=your_token
export GUBA_COOKIE='your_cookie'
export FLASK_SECRET_KEY='your_secret_key'
```

Windows PowerShell：

```powershell
$env:TUSHARE_TOKEN='your_token'
$env:GUBA_COOKIE='your_cookie'
$env:FLASK_SECRET_KEY='your_secret_key'
```

`GUBA_COOKIE` 仅在目标接口当前需要 Cookie 时配置。

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

原始评论、处理后的大规模 CSV、调试文件、训练权重以及生成的 HTML/图片报告体积较大，且部分数据包含站点会话信息，因此未纳入公开仓库。仓库保留完整核心代码、页面模板、报告摘要与目录结构，可通过运行流程重新生成相关结果。

## 主要技术栈

Python、Pandas、Tushare、Requests、Transformers、PyTorch、scikit-learn、Plotly、Flask、Jieba、WordCloud。
