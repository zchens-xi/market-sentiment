# -*- coding: utf-8 -*-
"""
词云图生成模块
从single_word列读取词汇数据并生成词云图
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_single_word_data(file_path, column_name='single_word'):
    """
    从文件加载single_word列数据

    参数:
    - file_path: 文件路径
    - column_name: 列名，默认'single_word'

    返回:
    - DataFrame: 包含词汇数据的DataFrame
    """
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ 成功读取文件: {file_path}")
        print(f"📊 文件形状: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")

        if column_name not in df.columns:
            print(f"❌ 未找到列 '{column_name}'")
            print(f"可用的列: {list(df.columns)}")
            return None

        # 检查数据
        valid_data = df[column_name].dropna()
        print(f"📈 有效数据行数: {len(valid_data)}")
        print(f"📝 数据示例: {valid_data.head().tolist()}")

        return df

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None

def process_words_from_column(df, column_name='single_word'):
    """
    从single_word列处理词汇数据

    参数:
    - df: DataFrame
    - column_name: 列名

    返回:
    - list: 处理后的词汇列表
    """
    if column_name not in df.columns:
        print(f"❌ 列 '{column_name}' 不存在")
        return []

    # 获取非空数据
    word_data = df[column_name].dropna().astype(str)

    all_words = []
    for text in word_data:
        # 按空格分词
        words = text.split()
        # 过滤处理
        for word in words:
            word = word.strip()
            # 过滤条件：长度大于1，不是纯数字或标点
            if len(word) > 1 and not re.match(r'^[0-9\.\-\+\%\s]+$', word):
                # 扩展的停用词列表 - 包含股票相关高频词
                stop_words = {
                    # 基础停用词
                    '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '个',
                    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看',
                    '好', '自己', '这', '那', '什么', '时候', '可以', '还是', '为了', '但是',
                    '因为', '如果', '虽然', '然后', '所以', '而且', '或者', '不过', '只是',
                    '已经', '可能', '应该', '比较', '非常', '特别', '一些', '这些', '那些',
                    '这样', '那样', '怎么', '为什么', '多少', '哪里', '什么时候', 'nbsp',
                    # 新增的高频无意义词汇
                    '这个', '一个', '大家', '中国', '继续', '大笑', '垃圾', '哈哈', '呵呵',
                    '真的', '还有', '其实', '本来', '原来', '后来', '终于', '立刻', '马上',
                    '刚才', '刚刚', '现在', '以前', '以后', '之前', '之后', '当时', '那时',
                    '这时', '此时', '平时', '有时', '经常', '总是', '从来', '一直', '一向',
                    '一般', '通常', '往往', '常常', '偶尔', '有时候', '什么的', '之类',
                    '等等', '什么', '东西', '事情', '问题', '方面', '情况', '地方', '时间',
                    '方式', '方法', '办法', '想法', '看法', '说法', '做法', '用法',
                    # 股票相关高频词
                    '股票', '今天', '明天', '昨天', '现在', '以后', '之前', '之后', '感觉',
                    '觉得', '认为', '估计', '应该', '肯定', '可能', '或许', '大概', '差不多',
                    '银行', '平安', '分红', '就是', '亿元', '市场', '买入', '公司', '万元',
                    '股价', '涨停', '跌停', '涨幅', '跌幅', '开盘', '收盘', '成交量', '换手率',
                    '主力', '机构', '散户', '资金', '流入', '流出', '净流入', '净流出',
                    '利好', '利空', '消息', '公告', '年报', '季报', '业绩', '盈利',
                    '投资', '投资者', '持股', '持有', '建仓', '减仓', '加仓', '清仓',
                    '看多', '看空', '看好', '看跌', '看涨', '止损', '止盈', '套牢',
                    '解套', '抄底', '逃顶', '追高', '杀跌', '反弹', '调整', '回调',
                    '突破', '支撑', '压力', '阻力', '趋势', '走势', '形态', '技术',
                    '基本面', '消息面', '政策', '央行', '降准', '降息', '加息',
                    '板块', '概念', '题材', '热点', '龙头', '妖股', '黑马', '白马',
                    '蓝筹', '小盘', '大盘', '指数', '上证', '深证', '创业板', '科创板',
                    '北向', '南向', '外资', '内资', '国资', '民营', '混改',
                    '重组', '并购', '分拆', '退市', '停牌', '复牌', '除权', '除息',
                    '配股', '增发', '回购', '质押', '解质', '减持', '增持',
                    '研报', '评级', '目标价', '估值', '市盈率', '市净率', '市销率',
                    '毛利率', '净利率', 'ROE', 'ROA', 'PB', 'PE', 'PEG',
                    '现金流', '负债率', '资产', '负债', '股本', '流通股', '总股本',
                    '股东', '股份', '控股', '参股', '子公司', '母公司', '关联',
                    '行业', '产业', '制造', '科技', '医药', '金融', '地产', '消费',
                    '新能源', '人工智能', '5G', '芯片', '半导体', '新材料',
                    '环保', '军工', '航空', '铁路', '港口', '物流', '电商',
                    '游戏', '影视', '教育', '旅游', '餐饮', '零售', '保险',
                    '券商', '信托', '基金', '私募', '公募', '理财', '债券',
                    '期货', '期权', '外汇', '黄金', '原油', '商品', '大宗',
                    '宏观', '微观', '经济', '通胀', '通缩', 'GDP', 'CPI', 'PPI',
                    '货币', '财政', '税收', '补贴', '扶持', '监管', '合规',
                    '风险', '收益', '波动', '稳定', '增长', '下滑', '复苏', '萧条'
                }
                if word not in stop_words:
                    all_words.append(word)

    print(f"📝 处理后的词汇总数: {len(all_words)}")
    print(f"📝 去重后的词汇数: {len(set(all_words))}")

    return all_words

def generate_wordcloud(words_list, save_path='reports/figures/wordcloud.png',
                      width=1200, height=800, max_words=200, background_color='white'):
    """
    生成词云图

    参数:
    - words_list: 词汇列表
    - save_path: 保存路径
    - width: 图片宽度
    - height: 图片高度
    - max_words: 最大词汇数
    - background_color: 背景颜色

    返回:
    - WordCloud对象
    """
    if not words_list:
        print("❌ 词汇列表为空，无法生成词云图")
        return None

    # 统计词频
    word_freq = Counter(words_list)
    print(f"📊 词频统计完成，共 {len(word_freq)} 个不同词汇")

    # 显示前10个高频词
    top_words = word_freq.most_common(10)
    print("🔥 前10个高频词:")
    for word, freq in top_words:
        print(f"   {word}: {freq}")

    # 创建词云对象
    try:
        # 尝试使用中文字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]

        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break

        if font_path:
            print(f"✅ 使用字体: {font_path}")
        else:
            print("⚠️ 未找到中文字体，使用默认字体")

        wordcloud = WordCloud(
            font_path=font_path,
            width=width,
            height=height,
            max_words=max_words,
            background_color=background_color,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)

        print("✅ 词云图生成成功")

        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存词云图
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('词汇词云图', fontsize=20, pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 词云图已保存到: {save_path}")

        return wordcloud

    except Exception as e:
        print(f"❌ 生成词云图失败: {e}")
        return None

def generate_word_frequency_chart(words_list, save_path='reports/figures/word_frequency.png', top_n=20):
    """
    生成词频柱状图

    参数:
    - words_list: 词汇列表
    - save_path: 保存路径
    - top_n: 显示前N个高频词
    """
    if not words_list:
        print("❌ 词汇列表为空，无法生成词频图")
        return

    # 统计词频
    word_freq = Counter(words_list)
    top_words = word_freq.most_common(top_n)

    # 创建DataFrame
    freq_df = pd.DataFrame(top_words, columns=['词汇', '频次'])

    # 绘制柱状图
    plt.figure(figsize=(12, 8))
    sns.barplot(data=freq_df, x='频次', y='词汇', palette='viridis')
    plt.title(f'前{top_n}个高频词汇', fontsize=16)
    plt.xlabel('频次', fontsize=12)
    plt.ylabel('词汇', fontsize=12)

    # 在柱子上显示数值
    for i, v in enumerate(freq_df['频次']):
        plt.text(v + 0.1, i, str(v), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 词频图已保存到: {save_path}")

def auto_find_single_word_file():
    """
    自动查找包含single_word列的文件
    """
    search_paths = [
        'data/processed/',
        'data/raw/',
        './'
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(search_path, file)
                    try:
                        # 只读取前几行检查列名
                        df_sample = pd.read_csv(file_path, nrows=5)
                        if 'single_word' in df_sample.columns:
                            print(f"🔍 找到包含single_word列的文件: {file_path}")
                            return file_path
                    except:
                        continue

    print("❌ 未找到包含single_word列的文件")
    return None

def run_wordcloud_analysis(file_path=None, column_name='single_word'):
    """
    运行完整的词云分析

    参数:
    - file_path: 文件路径，如果为None则自动查找
    - column_name: 列名
    """
    print("🎨 开始词云分析...")
    print("="*60)

    # 自动查找文件
    if file_path is None:
        file_path = auto_find_single_word_file()
        if file_path is None:
            print("❌ 请指定包含single_word列的文件路径")
            return

    # 加载数据
    df = load_single_word_data(file_path, column_name)
    if df is None:
        return

    # 处理词汇
    words_list = process_words_from_column(df, column_name)
    if not words_list:
        print("❌ 无法提取有效词汇")
        return

    # 生成词云图
    print("\n🎨 生成词云图...")
    wordcloud = generate_wordcloud(words_list)

    # 生成词频图
    print("\n📊 生成词频图...")
    generate_word_frequency_chart(words_list)

    # 保存词频统计
    word_freq = Counter(words_list)
    freq_df = pd.DataFrame(word_freq.most_common(100), columns=['词汇', '频次'])
    freq_path = 'reports/word_frequency_stats.csv'
    freq_df.to_csv(freq_path, index=False, encoding='utf-8-sig')
    print(f"✅ 词频统计已保存到: {freq_path}")

    print("\n🎉 词云分析完成！")
    print("="*60)

    return wordcloud, words_list

if __name__ == "__main__":
    # 运行词云分析
    # 方式1: 自动查找文件
    run_wordcloud_analysis()

    # 方式2: 指定文件路径
    # run_wordcloud_analysis('data/processed/your_file.csv')
