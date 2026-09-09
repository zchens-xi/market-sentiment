#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增量爬虫功能
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import config
from data_ingestion import PriceDataFetcher, CommentScraper

def test_incremental_price_fetcher():
    """测试增量价格数据获取"""
    print("=" * 50)
    print("测试增量价格数据获取功能")
    print("=" * 50)
    
    # 创建测试用的时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # 最近30天
    
    fetcher = PriceDataFetcher(token=config.TUSHARE_TOKEN)
    
    # 第一次运行 - 应该获取所有数据
    print("\n第一次运行 - 获取初始数据...")
    df1 = fetcher.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    print(f"第一次获取数据量: {len(df1) if df1 is not None else 0}")
    
    # 第二次运行 - 应该只获取增量数据
    print("\n第二次运行 - 测试增量获取...")
    df2 = fetcher.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d')
    )
    print(f"第二次获取数据量: {len(df2) if df2 is not None else 0}")
    
    # 测试更小的时间范围
    print("\n测试更小的时间范围...")
    small_end = end_date - timedelta(days=5)
    df3 = fetcher.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=small_end.strftime('%Y%m%d')
    )
    print(f"小范围获取数据量: {len(df3) if df3 is not None else 0}")

def test_incremental_comment_scraper():
    """测试增量评论爬取"""
    print("\n" + "=" * 50)
    print("测试增量评论爬取功能")
    print("=" * 50)
    
    # 创建测试用的时间范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # 最近7天
    
    scraper = CommentScraper()
    
    # 第一次运行 - 应该爬取数据
    print("\n第一次运行 - 爬取初始数据...")
    df1 = scraper.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date,
        end_date=end_date
    )
    print(f"第一次爬取数据量: {len(df1) if df1 is not None else 0}")
    
    # 第二次运行 - 应该只爬取增量数据
    print("\n第二次运行 - 测试增量爬取...")
    df2 = scraper.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date,
        end_date=end_date
    )
    print(f"第二次爬取数据量: {len(df2) if df2 is not None else 0}")
    
    # 测试更小的时间范围
    print("\n测试更小的时间范围...")
    small_end = end_date - timedelta(days=3)
    df3 = scraper.get_data(
        stock_code=config.STOCK_CODE,
        start_date=start_date,
        end_date=small_end
    )
    print(f"小范围爬取数据量: {len(df3) if df3 is not None else 0}")

def test_data_persistence():
    """测试数据持久化"""
    print("\n" + "=" * 50)
    print("测试数据持久化功能")
    print("=" * 50)
    
    # 检查文件是否存在
    price_file = os.path.join(config.RAW_DATA_PATH, f"{config.STOCK_CODE}_price_raw.csv")
    comment_file = os.path.join(config.RAW_DATA_PATH, f"{config.STOCK_CODE}_comment_raw.csv")
    
    print(f"价格数据文件存在: {os.path.exists(price_file)}")
    if os.path.exists(price_file):
        df = pd.read_csv(price_file)
        print(f"价格数据文件大小: {len(df)} 条记录")
    
    print(f"评论数据文件存在: {os.path.exists(comment_file)}")
    if os.path.exists(comment_file):
        df = pd.read_csv(comment_file)
        print(f"评论数据文件大小: {len(df)} 条记录")

def main():
    """主测试函数"""
    print("开始测试增量爬虫功能...")
    
    try:
        # 测试价格数据增量获取
        test_incremental_price_fetcher()
        
        # 测试评论数据增量爬取
        test_incremental_comment_scraper()
        
        # 测试数据持久化
        test_data_persistence()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 