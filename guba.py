import requests
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


# --- “工人”函数 (修正版) ---
def fetch_page_standalone(code, page, page_size, sort_type, headers):
    """
    独立地、自给自足地获取单个页面的帖子数据。
    每个线程调用此函数时，都会使用完整的、包含有效Cookie的请求头。
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
        # 使用 requests.post 而不是 session.post，每次都是一个新请求
        response = requests.post(api_url, headers=headers, data=payload, timeout=15)
        response.raise_for_status()
        data = response.json()

        # 增加对返回数据格式的健壮性检查
        if isinstance(data, dict) and 're' in data and 'count' in data:
            return page, data
        else:
            # 如果格式不对，就返回None，让主线程知道这个页面失败了
            return page, None

    except requests.exceptions.RequestException as e:
        # print(f"错误: 获取第 {page} 页失败 - {e}") # 可以取消注释来查看详细错误
        return page, None
    except Exception as e:
        # print(f"错误: 处理第 {page} 页时发生未知错误 - {e}")
        return page, None


# --- “工头”函数 (修正版) ---
def get_guba_posts_multithreaded(
        code: str,
        start_date: str,
        end_date: str,
        limit: int,
        sort_by_publish_time: bool = True,
        max_workers: int = 10
) -> pd.DataFrame:
    """
    从东方财富股吧多线程获取指定代码的帖子列表。
    """
    if not any([start_date, limit]):
        print("错误：必须提供 'start_date' 或 'limit' 参数之一。")
        return pd.DataFrame()

    # ===================== 【核心修正】 =====================
    # 1. 直接定义一个包含有效Cookie的headers字典。
    # 这个Cookie是整个脚本成功的关键，如果失效，需要从浏览器中重新获取。
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Cookie': os.getenv('GUBA_COOKIE', ''),
        'Referer': f'https://guba.eastmoney.com/list,{code}.html',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0',
    }
    # 2. 不再使用 requests.Session 和两步获取法。
    # ====================================================

    # ... 日期处理部分不变 ...
    start_dt, end_dt = None, None
    if start_date:
        try:
            start_dt = datetime.strptime(start_date, '%y%m%d')
        except ValueError:
            print(f"错误：开始日期 '{start_date}' 格式不正确。"); return pd.DataFrame()
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, '%y%m%d').replace(hour=23, minute=59, second=59)
        except ValueError:
            print(f"错误：结束日期 '{end_date}' 格式不正确。"); return pd.DataFrame()

    all_posts = []
    page_size = 100
    sort_type = '0' if sort_by_publish_time else '1'

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        current_page = 1
        stop_fetching = False

        while not stop_fetching:
            batch_size = max_workers * 2
            # 3. 将完整的 headers 字典传递给每个线程任务
            futures = {
                executor.submit(fetch_page_standalone, code, page, page_size, sort_type, headers): page
                for page in range(current_page, current_page + batch_size)
            }

            print(f"提交任务: 正在获取第 {current_page} 到 {current_page + batch_size - 1} 页...")

            batch_results = []
            for future in as_completed(futures):
                page, data = future.result()
                if data:
                    batch_results.append((page, data))

            batch_results.sort(key=lambda x: x[0])

            if not batch_results:
                print("所有任务均失败或无数据，可能已达末页，停止获取。")
                break

            for page, data in batch_results:
                posts = data.get('re', [])
                if not posts:
                    print(f"第 {page} 页无帖子内容，可能已达末页。")
                    stop_fetching = True;
                    break

                for post in posts:
                    # ... 后续数据处理逻辑完全保持不变 ...
                    post_time_str = post.get('post_publish_time', '')
                    try:
                        post_dt = datetime.fromisoformat(post_time_str)
                    except (ValueError, TypeError):
                        continue
                    if start_dt and post_dt < start_dt:
                        print("已获取到指定时间范围的帖子，停止抓取。");
                        stop_fetching = True;
                        break
                    title = post.get('post_title', '') or ''
                    content = post.get('post_content', '') or ''
                    comment_text = f"{title}\n{content}".strip()
                    all_posts.append({
                        'comment_text': comment_text,
                        'post_like_count': post.get('post_like_count', 0),
                        'post_publish_time': post_dt
                    })
                    if limit and len(all_posts) >= limit:
                        print(f"已达到指定的 {limit} 条帖子数量上限。");
                        stop_fetching = True;
                        break
                if stop_fetching: break

            current_page += batch_size
            time.sleep(0.1)

    if not all_posts: print("未能获取到任何帖子数据。"); return pd.DataFrame()

    print(f"总共获取 {len(all_posts)} 条帖子，正在整理成DataFrame...")
    df = pd.DataFrame(all_posts)

    if end_dt: df = df[df['post_publish_time'] <= end_dt]
    if limit: df = df.head(limit)

    df = df.sort_values(by='post_publish_time', ascending=False).reset_index(drop=True)

    return df[['comment_text', 'post_like_count', 'post_publish_time']]


if __name__ == '__main__':
    code_to_fetch = '000001'
    output_dir = './data/raw'
    output_file = f'{code_to_fetch}_comment_raw.csv'
    output_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- 目标: 获取 {code_to_fetch} 的 100,000 条帖子 ---")
    start_time = time.time()

    df_limit = get_guba_posts_multithreaded(
        code=code_to_fetch,
        limit=100000,
        max_workers=15
    )

    end_time = time.time()
    print(f"--- 数据获取完成，耗时: {end_time - start_time:.2f} 秒 ---")

    if not df_limit.empty:
        print(f"成功获取 {len(df_limit)} 条帖子。")
        print("正在保存到CSV文件...")
        try:
            df_limit.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"文件已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存文件失败: {e}")
        print("\n数据预览:")
        print(df_limit.head())