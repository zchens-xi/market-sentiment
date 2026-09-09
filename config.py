import os
from datetime import datetime, timedelta

# --- Tushare API 配置 ---
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN', '')  # 请通过环境变量配置

# --- 目标股票与时间范围 ---
STOCK_CODE = '000001.SZ'
STOCK_NAME = '平安银行' # 增加股票名称，用于报告标题
START_DATE = '20220722'
END_DATE = datetime.now().strftime('%Y%m%d')

# --- 数据与报告存储路径 ---
DATA_PATH = 'data'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
REPORT_PATH = 'reports' # 报告目录
HTML_REPORT_FILENAME = 'stock_analysis_report.html' # 报告文件名


# --- 爬虫相关配置 ---
# 移除了CHROME_DRIVER_PATH和雪球相关配置，现在使用股吧API
# XUEQIU_URL_TEMPLATE = "https://xueqiu.com/S/{stock_code}"  # 已弃用
# XUEQIU_COOKIES_PATH = os.path.join(DATA_PATH, 'xueqiu_cookies.json')  # 已弃用
MAX_COMMENTS_TO_SCRAPE = 100000  # 股吧API获取的评论数量
MAX_SCROLLS = 50  # 保留用于其他用途

# --- 模型路径 ---
SENTIMENT_MODEL_PATH = 'yiyanghkust/finbert-tone-chinese'

# --- 确保目录存在 ---
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)