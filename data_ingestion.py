# -*- coding: utf-8 -*-
import time
import pandas as pd
import tushare as ts
import json
import os
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import config
from util import DataSaver

class DataSource(ABC):
    @abstractmethod
    def get_data(self, stock_code, start_date=None, end_date=None) -> pd.DataFrame | None:
        """返回数据，类型可以是DataFrame或None"""
        pass


class PriceDataFetcher(DataSource):
    def __init__(self, token):
        self.pro = ts.pro_api(token)
        print("价格数据获取器初始化成功。")

    def get_data(self, stock_code=config.STOCK_CODE, start_date=config.START_DATE, end_date=config.END_DATE,
                 save_path=config.RAW_DATA_PATH, save_filename=f"{config.STOCK_CODE}_price_raw.csv"):
        """
        实现增量价格数据获取功能
        """
        import pandas as pd
        from datetime import datetime
        
        # 1. 读取历史数据
        file_path = os.path.join(save_path, save_filename)
        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)
            old_df['trade_date'] = pd.to_datetime(old_df['trade_date'])
            min_date, max_date = old_df['trade_date'].min(), old_df['trade_date'].max()
            print(f"已有价格数据区间: {min_date} ~ {max_date}")
        else:
            old_df = pd.DataFrame()
            min_date, max_date = None, None

        # 2. 解析时间范围参数
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y%m%d')
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        else:
            end_dt = end_date

        print(f"请求价格数据区间: {start_dt} ~ {end_dt}")

        # 3. 判断是否需要增量获取
        need_fetch = False
        fetch_reason = ""
        
        if old_df.empty:
            need_fetch = True
            fetch_reason = "无历史价格数据，需要首次获取"
        else:
            # 检查是否需要获取更新的数据
            if max_date < end_dt:
                need_fetch = True
                fetch_reason = f"历史数据最新日期为 {max_date}，需要获取 {max_date} 到 {end_dt} 的新数据"
            elif min_date > start_dt:
                need_fetch = True
                fetch_reason = f"历史数据最早日期为 {min_date}，需要获取 {start_dt} 到 {min_date} 的历史数据"
            else:
                # 检查是否有数据缺失
                date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
                existing_dates = old_df['trade_date'].dt.date.unique()
                missing_dates = [d.date() for d in date_range if d.date() not in existing_dates]
                
                if missing_dates:
                    need_fetch = True
                    fetch_reason = f"发现缺失日期: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                else:
                    need_fetch = False
                    fetch_reason = "请求区间已被历史数据完全覆盖"

        # 4. 若无增量，直接返回已有数据
        if not need_fetch:
            print(f"无需获取价格数据: {fetch_reason}")
            # 返回指定时间范围内的数据
            if not old_df.empty:
                filtered_df = old_df[
                    (old_df['trade_date'] >= start_dt) & 
                    (old_df['trade_date'] <= end_dt)
                ]
                print(f"返回历史价格数据 {len(filtered_df)} 条")
                return filtered_df
            else:
                return old_df

        # 5. 需要增量获取
        print(f"开始增量获取价格数据: {fetch_reason}")
        
        # 计算需要获取的时间范围
        if old_df.empty:
            fetch_start = start_date
            fetch_end = end_date
        else:
            # 获取最新数据之后的日期
            if max_date < end_dt:
                fetch_start = max_date.strftime('%Y%m%d')
                fetch_end = end_date
            else:
                fetch_start = start_date
                fetch_end = end_date

        print(f"增量获取时间范围: {fetch_start} ~ {fetch_end}")

        # 6. 获取新数据
        for i in range(3):
            try:
                df = self.pro.daily(ts_code=stock_code, start_date=fetch_start, end_date=fetch_end)
                if not df.empty:
                    print("价格数据获取成功。")
                    break
            except Exception as e:
                print(f"第 {i + 1} 次获取价格数据失败，1秒后重试... 错误: {e}")
                time.sleep(1)
        else:
            print("错误：多次尝试后仍无法获取价格数据。")
            # 返回指定时间范围内的历史数据
            if not old_df.empty:
                filtered_df = old_df[
                    (old_df['trade_date'] >= start_dt) & 
                    (old_df['trade_date'] <= end_dt)
                ]
                return filtered_df
            else:
                return None

        # 7. 合并数据
        if not old_df.empty:
            # 转换新数据的日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            merged_df = pd.concat([old_df, df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        else:
            merged_df = df
            merged_df['trade_date'] = pd.to_datetime(merged_df['trade_date'])

        # 8. 保存数据
        if save_path and save_filename:
            saver = DataSaver(status=True)
            saver.save_to_csv(merged_df, save_path, save_filename)
            print(f"价格数据已增量保存到: {os.path.join(save_path, save_filename)}")

        new_count = len(df)
        print(f"本次新增 {new_count} 条价格数据，合计 {len(merged_df)} 条")
        
        # 返回指定时间范围内的数据
        final_df = merged_df[
            (merged_df['trade_date'] >= start_dt) & 
            (merged_df['trade_date'] <= end_dt)
        ]
        print(f"返回指定时间范围价格数据 {len(final_df)} 条")
        return final_df


class CommentScraper(DataSource):
    def __init__(self):
        """
        股吧API评论获取器，不再需要Selenium WebDriver
        """
        self.headers = {
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Cookie': os.getenv('GUBA_COOKIE', ''),
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0',
        }
        print("股吧API评论获取器初始化成功。")

    def get_data(self, stock_code=config.STOCK_CODE, start_date=None, end_date=None, save_path=config.RAW_DATA_PATH,
                 save_filename=f"{config.STOCK_CODE}_comment_raw.csv"):
        """
        实现真正的增量爬虫功能，基于时间范围判断是否需要爬取新数据
        """
        import pandas as pd
        from datetime import datetime

        # 1. 读取历史数据
        file_path = os.path.join(save_path, save_filename)
        if os.path.exists(file_path):
            old_df = pd.read_csv(file_path)
            old_df['timestamp'] = pd.to_datetime(old_df['timestamp'])
            min_time, max_time = old_df['timestamp'].min(), old_df['timestamp'].max()
            print(f"已有数据区间: {min_time} ~ {max_time}")
        else:
            old_df = pd.DataFrame()
            min_time, max_time = None, None

        # 2. 解析时间范围参数
        if start_date:
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y%m%d')
            else:
                start_dt = start_date
        else:
            start_dt = datetime.strptime(config.START_DATE, '%Y%m%d')
            
        if end_date:
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y%m%d')
            else:
                end_dt = end_date
        else:
            end_dt = datetime.now()

        print(f"请求数据区间: {start_dt} ~ {end_dt}")

        # 3. 判断是否需要增量爬取
        need_fetch = False
        fetch_reason = ""
        
        if old_df.empty:
            need_fetch = True
            fetch_reason = "无历史数据，需要首次爬取"
        else:
            # 检查是否需要获取更新的数据
            if max_time < end_dt:
                need_fetch = True
                fetch_reason = f"历史数据最新时间为 {max_time}，需要获取 {max_time} 到 {end_dt} 的新数据"
            elif min_time > start_dt:
                need_fetch = True
                fetch_reason = f"历史数据最早时间为 {min_time}，需要获取 {start_dt} 到 {min_time} 的历史数据"
            else:
                # 检查是否有数据缺失
                date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
                existing_dates = old_df['timestamp'].dt.date.unique()
                missing_dates = [d.date() for d in date_range if d.date() not in existing_dates]
                
                if missing_dates:
                    need_fetch = True
                    fetch_reason = f"发现缺失日期: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}"
                else:
                    need_fetch = False
                    fetch_reason = "请求区间已被历史数据完全覆盖"

        # 4. 若无增量，直接返回已有数据
        if not need_fetch:
            print(f"无需爬取: {fetch_reason}")
            # 返回指定时间范围内的数据
            if not old_df.empty:
                filtered_df = old_df[
                    (old_df['timestamp'] >= start_dt) & 
                    (old_df['timestamp'] <= end_dt)
                ]
                print(f"返回历史数据 {len(filtered_df)} 条")
                return filtered_df
            else:
                return old_df

        # 5. 需要增量爬取
        print(f"开始增量爬取: {fetch_reason}")
        
        # 计算需要爬取的时间范围
        if old_df.empty:
            fetch_start = start_dt
            fetch_end = end_dt
        else:
            # 获取最新数据之后的时间
            fetch_start = max_time if max_time < end_dt else start_dt
            fetch_end = end_dt

        print(f"增量爬取时间范围: {fetch_start} ~ {fetch_end}")

        # 6. 爬取数据（带时间范围限制）
        comments_df = self._get_guba_posts_with_time_limit(
            code=stock_code.split('.')[0],
            start_time=fetch_start,
            end_time=fetch_end,
            limit=config.MAX_COMMENTS_TO_SCRAPE if hasattr(config, 'MAX_COMMENTS_TO_SCRAPE') else 1000
        )

        if comments_df.empty:
            print("增量爬取未获取到新数据")
            # 返回指定时间范围内的历史数据
            if not old_df.empty:
                filtered_df = old_df[
                    (old_df['timestamp'] >= start_dt) & 
                    (old_df['timestamp'] <= end_dt)
                ]
                return filtered_df
            else:
                return old_df

        # 7. 整理新数据
        result_df = pd.DataFrame({
            'comment_text': comments_df['comment_text'],
            'timestamp': comments_df['post_publish_time'],
            'stock_code': stock_code
        })

        # 8. 合并去重
        if not old_df.empty:
            merged_df = pd.concat([old_df, result_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['comment_text', 'timestamp']).sort_values(
                'timestamp').reset_index(drop=True)
        else:
            merged_df = result_df.sort_values('timestamp').reset_index(drop=True)

        # 9. 保存前转换为字符串格式
        save_df = merged_df.copy()
        save_df['timestamp'] = save_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if save_path and save_filename:
            saver = DataSaver()
            saver.save_to_csv(save_df, save_path, save_filename)
            print(f"评论数据已增量保存到: {os.path.join(save_path, save_filename)}")

        new_count = len(result_df)
        print(f"本次新增 {new_count} 条评论，合计 {len(merged_df)} 条")
        
        # 返回指定时间范围内的数据
        final_df = merged_df[
            (merged_df['timestamp'] >= start_dt) & 
            (merged_df['timestamp'] <= end_dt)
        ]
        print(f"返回指定时间范围数据 {len(final_df)} 条")
        return final_df

    def _fetch_page_standalone(self, code, page, page_size, sort_type):
        """
        获取单个页面的帖子数据，优化为独立函数便于并行处理
        """
        base_url = "https://guba.eastmoney.com/api/getData"
        path = "webarticlelist/api/article/WebArticleList"
        api_url = f"{base_url}?code={code}&path={path}"

        params_str = f"code={code}&type=0&sorttype={sort_type}&p={page}&ps={page_size}"
        payload = {
            'param': params_str, 'plat': 'Web', 'path': path, 'env': '2',
            'origin': '', 'version': '2022', 'product': 'Guba'
        }

        try:
            # 为每个页面设置独立的Referer
            headers = self.headers.copy()
            headers['Referer'] = f'https://guba.eastmoney.com/list,{code}.html'

            response = requests.post(api_url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            # 增强数据格式检查
            if isinstance(data, dict) and 're' in data and 'count' in data:
                return page, data
            else:
                return page, None

        except Exception as e:
            print(f"获取第 {page} 页失败: {e}")
            return page, None

    def _get_guba_posts(self, code, limit=1000, max_workers=15):
        """
        从东方财富股吧获取指定代码的帖子列表，使用高效并行处理
        参照guba.py的优化策略
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime
        import requests
        
        all_posts = []
        page_size = 100
        sort_type = '0'  # 按发布时间排序

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            current_page = 1
            stop_fetching = False

            while not stop_fetching and len(all_posts) < limit:
                # 批量提交任务，提高并行效率
                batch_size = max_workers * 2
                futures = {
                    executor.submit(self._fetch_page_standalone, code, page, page_size, sort_type): page
                    for page in range(current_page, current_page + batch_size)
                }

                print(f"提交批量任务: 正在获取第 {current_page} 到 {current_page + batch_size - 1} 页...")

                # 收集批量结果
                batch_results = []
                for future in as_completed(futures):
                    page, data = future.result()
                    if data:
                        batch_results.append((page, data))

                # 按页码排序确保数据顺序
                batch_results.sort(key=lambda x: x[0])

                if not batch_results:
                    print("所有任务均失败或无数据，停止获取。")
                    break

                # 处理批量结果
                for page, data in batch_results:
                    posts = data.get('re', [])
                    if not posts:
                        print(f"第 {page} 页无帖子内容，可能已达末页。")
                        stop_fetching = True
                        break

                    for post in posts:
                        post_time_str = post.get('post_publish_time', '')
                        try:
                            post_dt = datetime.fromisoformat(post_time_str)
                        except (ValueError, TypeError):
                            continue
                            
                        title = post.get('post_title', '') or ''
                        content = post.get('post_content', '') or ''
                        comment_text = f"{title}\n{content}".strip()
                        
                        all_posts.append({
                            'comment_text': comment_text,
                            'post_like_count': post.get('post_like_count', 0),
                            'post_publish_time': post_dt
                        })
                        
                        if len(all_posts) >= limit:
                            print(f"已达到指定的 {limit} 条帖子数量上限。")
                            stop_fetching = True
                            break
                    
                    if stop_fetching:
                        break

                current_page += batch_size
                # 短暂延迟避免请求过于频繁
                time.sleep(0.1)

        if not all_posts:
            print("未能获取到任何帖子数据。")
            return pd.DataFrame()

        print(f"总共获取 {len(all_posts)} 条帖子，正在整理成DataFrame...")
        df = pd.DataFrame(all_posts)
        df = df.sort_values(by='post_publish_time', ascending=False).reset_index(drop=True)

        return df[['comment_text', 'post_like_count', 'post_publish_time']]

    def _get_guba_posts_with_time_limit(self, code, start_time, end_time, limit=1000):
        """
        从东方财富股吧获取指定代码的帖子列表，带时间范围限制和智能提前停止
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from datetime import datetime
        import requests
        
        all_posts = []
        page_size = 100
        sort_type = '0'  # 按发布时间排序
        consecutive_old_posts = 0  # 连续超出时间范围的帖子计数
        max_consecutive_old = 50   # 连续50个帖子超出时间范围就停止

        print(f"开始获取时间范围 {start_time} ~ {end_time} 的帖子...")

        with ThreadPoolExecutor(max_workers=15) as executor:
            current_page = 1
            stop_fetching = False

            while not stop_fetching and len(all_posts) < limit:
                # 批量提交任务，提高并行效率
                batch_size = 15
                futures = {
                    executor.submit(self._fetch_page_standalone, code, page, page_size, sort_type): page
                    for page in range(current_page, current_page + batch_size)
                }

                print(f"提交批量任务: 正在获取第 {current_page} 到 {current_page + batch_size - 1} 页...")

                # 收集批量结果
                batch_results = []
                for future in as_completed(futures):
                    page, data = future.result()
                    if data:
                        batch_results.append((page, data))

                # 按页码排序确保数据顺序
                batch_results.sort(key=lambda x: x[0])

                if not batch_results:
                    print("所有任务均失败或无数据，停止获取。")
                    break

                # 处理批量结果
                for page, data in batch_results:
                    posts = data.get('re', [])
                    if not posts:
                        print(f"第 {page} 页无帖子内容，可能已达末页。")
                        stop_fetching = True
                        break

                    page_has_valid_posts = False
                    for post in posts:
                        post_time_str = post.get('post_publish_time', '')
                        try:
                            post_dt = datetime.fromisoformat(post_time_str)
                        except (ValueError, TypeError):
                            continue
                            
                        if start_time <= post_dt <= end_time:
                            title = post.get('post_title', '') or ''
                            content = post.get('post_content', '') or ''
                            comment_text = f"{title}\n{content}".strip()
                            
                            all_posts.append({
                                'comment_text': comment_text,
                                'post_like_count': post.get('post_like_count', 0),
                                'post_publish_time': post_dt
                            })
                            
                            page_has_valid_posts = True
                            consecutive_old_posts = 0  # 重置计数器
                            
                            if len(all_posts) >= limit:
                                print(f"已达到指定的 {limit} 条帖子数量上限。")
                                stop_fetching = True
                                break
                        else:
                            # 帖子时间超出范围
                            consecutive_old_posts += 1
                            if consecutive_old_posts >= max_consecutive_old:
                                print(f"连续 {max_consecutive_old} 个帖子超出时间范围，提前停止爬取。")
                                stop_fetching = True
                                break
                    
                    if not page_has_valid_posts:
                        print(f"第 {page} 页无有效时间范围内的帖子")
                    
                    if stop_fetching:
                        break

                current_page += batch_size
                # 短暂延迟避免请求过于频繁
                time.sleep(0.1)

        if not all_posts:
            print("未能获取到任何帖子数据。")
            return pd.DataFrame()

        print(f"总共获取 {len(all_posts)} 条帖子，正在整理成DataFrame...")
        df = pd.DataFrame(all_posts)
        df = df.sort_values(by='post_publish_time', ascending=False).reset_index(drop=True)

        return df[['comment_text', 'post_like_count', 'post_publish_time']]
