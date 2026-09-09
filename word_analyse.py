import pandas as pd
import numpy as np
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

def calculate_word_frequencies(df, text_column='processed_text', min_freq=2, top_n=50):
    """
    计算文本列中每个词的频率，并返回一个DataFrame。
    
    参数:
        df: DataFrame，包含文本数据
        text_column: str，文本列名，默认'processed_text'
        min_freq: int，最小词频过滤，默认2
        top_n: int，返回前N个高频词，默认50
    
    返回:
        DataFrame，包含词语和频率
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame中必须包含 '{text_column}' 列。")
    
    # 合并所有文本
    all_text = ' '.join(df[text_column].dropna().astype(str))
    
    # 分词
    words = jieba.cut(all_text)
    
    # 过滤停用词和无效词
    stop_words = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '什么', '时候', '可以', '还是', '为了', '但是', '因为', '如果', '虽然', '然后', '所以', '而且', '或者', '不过', '只是', '已经', '可能', '应该', '比较', '非常', '特别', '一些', '这些', '那些', '这样', '那样', '怎么', '为什么', '多少', '哪里', '什么时候'}
    
    valid_words = [word.strip() for word in words 
                   if len(word.strip()) > 1 and word.strip() not in stop_words 
                   and not re.match(r'^[0-9\.\-\+\%]+$', word.strip())]
    
    # 计算词频
    word_freq = Counter(valid_words)
    
    # 过滤低频词
    filtered_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    # 转换为DataFrame并排序
    freq_df = pd.DataFrame(list(filtered_freq.items()), columns=['word', 'frequency'])
    freq_df = freq_df.sort_values('frequency', ascending=False).head(top_n)
    freq_df = freq_df.reset_index(drop=True)
    
    return freq_df

def calculate_sentiment_word_frequencies(df, text_column='processed_text', sentiment_column='sentiment_score', threshold=0.6):
    """
    分别计算正面和负面情感文本的词频
    
    参数:
        df: DataFrame，包含文本和情感数据
        text_column: str，文本列名
        sentiment_column: str，情感得分列名
        threshold: float，情感分类阈值
    
    返回:
        tuple: (positive_words_df, negative_words_df)
    """
    if text_column not in df.columns or sentiment_column not in df.columns:
        raise ValueError(f"DataFrame中必须包含 '{text_column}' 和 '{sentiment_column}' 列。")
    
    # 分离正面和负面情感文本
    positive_df = df[df[sentiment_column] > threshold]
    negative_df = df[df[sentiment_column] < (1 - threshold)]
    
    # 分别计算词频
    positive_words = calculate_word_frequencies(positive_df, text_column, min_freq=1, top_n=30)
    negative_words = calculate_word_frequencies(negative_df, text_column, min_freq=1, top_n=30)
    
    return positive_words, negative_words

def generate_wordcloud(df, text_column='processed_text', output_path=None, width=800, height=400):
    """
    生成词云图
    
    参数:
        df: DataFrame，包含文本数据
        text_column: str，文本列名
        output_path: str，保存路径，如果为None则不保存
        width: int，图片宽度
        height: int，图片高度
    
    返回:
        WordCloud对象
    """
    if text_column not in df.columns:
        raise ValueError(f"DataFrame中必须包含 '{text_column}' 列。")
    
    # 合并所有文本
    all_text = ' '.join(df[text_column].dropna().astype(str))
    
    # 设置字体路径（Windows系统）
    font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体
    
    # 生成词云
    try:
        wordcloud = WordCloud(
            font_path=font_path,
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
    except:
        # 如果字体文件不存在，使用默认设置
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
    
    # 保存图片
    if output_path:
        plt.figure(figsize=(width/100, height/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"词云图已保存到: {output_path}")
    
    return wordcloud

def plot_word_frequencies(freq_df, title="词频分析", top_n=20, output_path=None):
    """
    绘制词频柱状图
    
    参数:
        freq_df: DataFrame，词频数据
        title: str，图表标题
        top_n: int，显示前N个词
        output_path: str，保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 取前N个词
    top_words = freq_df.head(top_n)
    
    # 创建柱状图
    sns.barplot(data=top_words, x='frequency', y='word', palette='viridis')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('频率', fontsize=12)
    plt.ylabel('词语', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"词频图已保存到: {output_path}")
    
    plt.show()

def analyze_keyword_trends(df, keywords, text_column='processed_text', date_column='comment_date'):
    """
    分析关键词在时间序列中的趋势
    
    参数:
        df: DataFrame，包含文本和日期数据
        keywords: list，要分析的关键词列表
        text_column: str，文本列名
        date_column: str，日期列名
    
    返回:
        DataFrame，关键词趋势数据
    """
    if text_column not in df.columns or date_column not in df.columns:
        raise ValueError(f"DataFrame中必须包含 '{text_column}' 和 '{date_column}' 列。")
    
    # 确保日期列是datetime类型
    df[date_column] = pd.to_datetime(df[date_column])
    
    # 为每个关键词创建计数
    keyword_trends = []
    
    for keyword in keywords:
        # 按日期分组，计算每天包含该关键词的评论数
        daily_counts = df.groupby(df[date_column].dt.date).apply(
            lambda x: x[text_column].str.contains(keyword, na=False).sum()
        ).reset_index()
        daily_counts.columns = ['date', f'{keyword}_count']
        daily_counts['keyword'] = keyword
        keyword_trends.append(daily_counts)
    
    # 合并所有关键词的趋势数据
    if keyword_trends:
        trend_df = pd.concat(keyword_trends, ignore_index=True)
        return trend_df
    else:
        return pd.DataFrame()

def get_stock_related_keywords():
    """
    获取股票相关的常用关键词列表
    
    返回:
        dict: 分类的关键词字典
    """
    return {
        'positive': ['涨', '上涨', '买入', '看好', '牛市', '利好', '强势', '突破', '收益'],
        'negative': ['跌', '下跌', '卖出', '看空', '熊市', '利空', '弱势', '破位', '亏损'],
        'neutral': ['观望', '震荡', '整理', '横盘', '分析', '研究', '关注', '持有'],
        'technical': ['支撑', '压力', '突破', '回调', '反弹', '趋势', '均线', '成交量']
    }

# 使用示例和测试函数
def example_usage():
    """
    词频分析模块的使用示例
    """
    import os
    import config
    
    try:
        # 读取处理后的评论数据
        comment_path = os.path.join(config.PROCESSED_DATA_PATH, f'{config.STOCK_CODE}_comments_processed.csv')
        
        if os.path.exists(comment_path):
            df = pd.read_csv(comment_path)
            print(f"读取评论数据: {len(df)} 条")
            
            # 1. 基础词频分析
            print("\n=== 基础词频分析 ===")
            word_freq = calculate_word_frequencies(df, 'processed_text', min_freq=3, top_n=20)
            print("前20个高频词:")
            print(word_freq)
            
            # 2. 情感词频分析（如果有情感得分）
            if 'sentiment_score' in df.columns:
                print("\n=== 情感词频分析 ===")
                pos_words, neg_words = calculate_sentiment_word_frequencies(df, 'processed_text', 'sentiment_score')
                print("正面情感高频词:")
                print(pos_words.head(10))
                print("\n负面情感高频词:")
                print(neg_words.head(10))
            
            # 3. 关键词趋势分析（如果有日期）
            if 'comment_date' in df.columns:
                print("\n=== 关键词趋势分析 ===")
                stock_keywords = get_stock_related_keywords()
                trend_keywords = stock_keywords['positive'][:3] + stock_keywords['negative'][:3]
                
                trend_df = analyze_keyword_trends(df, trend_keywords, 'processed_text', 'comment_date')
                print("关键词趋势数据:")
                print(trend_df.head())
            
            # 4. 生成词频图表
            print("\n=== 生成词频图表 ===")
            output_dir = os.path.join('reports', 'figures')
            os.makedirs(output_dir, exist_ok=True)
            
            # 绘制词频图
            plot_word_frequencies(word_freq, f"{config.STOCK_NAME}词频分析", 
                                top_n=15, output_path=os.path.join(output_dir, 'word_frequencies.png'))
            
            # 生成词云（可选，需要wordcloud包）
            try:
                wordcloud = generate_wordcloud(df, 'processed_text', 
                                             output_path=os.path.join(output_dir, 'wordcloud.png'))
                print("词云图生成成功")
            except Exception as e:
                print(f"词云图生成失败: {e}")
            
            return word_freq
            
        else:
            print(f"评论数据文件不存在: {comment_path}")
            return None
            
    except Exception as e:
        print(f"词频分析示例执行失败: {e}")
        return None

if __name__ == '__main__':
    print("=== 词频分析模块测试 ===")
    result = example_usage()
    
    if result is not None:
        print("\n✅ 词频分析模块运行成功！")
    else:
        print("\n❌ 词频分析模块运行失败！")


