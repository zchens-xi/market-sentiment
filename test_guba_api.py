#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试股吧API评论获取功能
"""

import os
import sys
import pandas as pd
import config
from data_ingestion import CommentScraper, DataSaver

def test_guba_api():
    """测试股吧API功能"""
    print("=== 股吧API评论获取测试 ===")
    
    # 测试参数
    test_stock_code = config.STOCK_CODE  # 使用配置中的股票代码
    test_limit = 50  # 测试只获取50条数据
    
    print(f"测试股票代码: {test_stock_code}")
    print(f"计划获取数量: {test_limit} 条")
    
    try:
        # 创建评论爬虫实例
        scraper = CommentScraper()
        
        # 获取数据
        print("\n开始获取股吧评论数据...")
        comment_df = scraper.get_data(test_stock_code)
        
        if comment_df is not None and not comment_df.empty:
            print(f"\n✅ 测试成功！获取到 {len(comment_df)} 条评论")
            
            # 显示数据预览
            print("\n--- 数据预览 (前5条) ---")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', 100)
            
            print(comment_df.head())
            
            # 数据统计
            print(f"\n--- 数据统计 ---")
            print(f"总评论数: {len(comment_df)}")
            print(f"数据列: {list(comment_df.columns)}")
            print(f"时间范围: {comment_df['timestamp'].min()} 到 {comment_df['timestamp'].max()}")
            
            # 保存测试数据
            saver = DataSaver()
            test_file = f"test_{test_stock_code}_guba_comments.csv"
            saver.save_to_csv(comment_df, config.RAW_DATA_PATH, test_file)
            print(f"\n测试数据已保存到: {os.path.join(config.RAW_DATA_PATH, test_file)}")
            
        else:
            print("\n❌ 测试失败：未能获取到任何评论数据")
            print("可能的原因:")
            print("1. 网络连接问题")
            print("2. 股吧API接口变更")
            print("3. 股票代码格式问题")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        print("\n请检查:")
        print("1. 网络连接是否正常")
        print("2. 依赖包是否安装完整")
        print("3. 股票代码是否正确")

def test_data_integration():
    """测试数据整合功能"""
    print("\n=== 数据整合测试 ===")
    
    try:
        # 测试DataSaver的完整流程
        saver = DataSaver()
        
        print("测试股票价格数据获取...")
        price_df = saver.fetch_stock_data(config.STOCK_CODE, config.START_DATE, config.END_DATE)
        
        print("测试评论数据获取...")
        comment_df = saver.fetch_comments_data(config.STOCK_CODE)
        
        if price_df is not None and not price_df.empty:
            print(f"✅ 股票价格数据: {len(price_df)} 条")
        else:
            print("❌ 股票价格数据获取失败")
            
        if comment_df is not None and not comment_df.empty:
            print(f"✅ 评论数据: {len(comment_df)} 条")
        else:
            print("❌ 评论数据获取失败")
            
    except Exception as e:
        print(f"❌ 数据整合测试失败: {e}")

if __name__ == "__main__":
    # 屏蔽警告
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')
    
    print("股吧API集成测试")
    print("=" * 40)
    
    # 检查基本环境
    print(f"当前工作目录: {os.getcwd()}")
    print(f"配置的股票代码: {config.STOCK_CODE}")
    print(f"数据保存路径: {config.RAW_DATA_PATH}")
    
    # 确保目录存在
    os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
    
    # 运行测试
    test_guba_api()
    test_data_integration()
    
    print("\n" + "=" * 40)
    print("测试完成！")
