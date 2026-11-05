# -*- coding: utf-8 -*-
"""
yf_amazon 数据集：数据清洗 + 跨表一致性校验
============================================
输入（位于 BASE 目录）：
    categories.csv（已解压） / links.zip / products.zip / ratings.zip
输出（位于 BASE/cleaned 目录）：
    cleaned_categories.csv
    cleaned_products.csv
    cleaned_links.csv
    cleaned_ratings.csv
    users.csv              （由 ratings 派生的用户表，MovieLens 风格）

v2 新增的一致性处理（相比 v1）：
    1) 用户一致性：数据集中没有 users 表，因此以“原始 ratings 中出现过的 userId”
       定义用户全集；评分中引用不存在用户、非法（负数）userId 的行会被删除；
       并从清洗后的 ratings 派生 users.csv，与 ratings 做双向校验。
    2) links ↔ products 双向校验：
       - 删除 productId 在 products 中不存在的孤儿 links；
       - 报告 products 中缺少 links 的商品数量（默认保留，可选删除）；
       - 检查 amazonId 是否出现“一个编号对应多个商品”的冲突。
    3) 类目一致性：products.catIds 中引用的类目 ID 必须在 categories.catId 中存在
       （默认只报告，可选删除）。
    4) 重新连续编号：清洗删除行后 ID 会留下空洞，默认把所有 ID（productId / userId /
       catId）重新编号为 0 开始的连续整数，并联动更新 ratings / links / products /
       users，保证跨表引用一致。可用 RENUMBER_IDS 关闭。
"""

import os
import re
import pandas as pd

# ========== 1. 路径配置 ==========
BASE = r'D:\PythonProject1\yf_amazon\raw'
OUT  = os.path.join(BASE, 'cleaned')
os.makedirs(OUT, exist_ok=True)

# ========== 2. 处理开关 ==========
RENUMBER_IDS            = True    # 清洗后重新连续编号（默认开启）
DROP_PRODUCTS_NO_LINK   = False   # 是否删除“没有 links 记录”的商品（默认保留并报告）
DROP_PRODUCTS_BAD_CATID = False   # 是否删除“catIds 引用了不存在类目”的商品（默认保留并报告）


# ========== 3. 工具函数 ==========
def full2half(text):
    """全角字符 → 半角字符（全角空格 U+3000、全角 ASCII 区 FF01~FF5E、全角￥）"""
    if not isinstance(text, str):
        return text
    chars = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            chars.append(' ')
        elif 0xFF01 <= code <= 0xFF5E:
            chars.append(chr(code - 0xFEE0))
        elif code == 0xFFE5:
            chars.append('¥')
        else:
            chars.append(ch)
    return ''.join(chars)


def clean_text(s):
    """清洗文本列：全角转半角 + 去首尾空白 + 压缩连续空白"""
    if not isinstance(s, str):
        return s
    s = full2half(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s


def clean_cat_ids(s):
    """清洗 catIds：全角转半角、去掉逗号周围空格，如 '802, 143，996' → '802,143,996'"""
    if not isinstance(s, str):
        return s
    s = full2half(s).strip()
    s = re.sub(r'\s*,\s*', ',', s)
    return s


def read_csv(name):
    """按 name.csv → name.zip 的顺序读取，兼容“已解压 / 未解压”两种状态"""
    for ext in ('.csv', '.zip'):
        p = os.path.join(BASE, name + ext)
        if os.path.exists(p):
            print(f'[读取] {os.path.basename(p)} ...', flush=True)
            return pd.read_csv(p, encoding='utf-8')
    raise FileNotFoundError(f'在 {BASE} 下找不到 {name}.csv 或 {name}.zip')


def catids_valid(s, valid_cats):
    """校验 catIds 字符串里的每个类目 ID 是否都在 categories 表中。
    返回 True=全部合法 / False=存在非法引用 / None=无法校验（空值）。"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s == '':
        return None
    parts = [int(x) for x in s.split(',') if x != '']
    return all(p in valid_cats for p in parts)


def renumber(df, col):
    """把 df[col] 重新编号为 0..n-1（按原 ID 升序）。
    返回 (新 DataFrame, 旧ID → 新ID 映射 dict)。"""
    old_ids = sorted(df[col].unique())
    mapping = {old: new for new, old in enumerate(old_ids)}
    out = df.copy()
    out[col] = out[col].map(mapping)
    return out, mapping


def remap_cat_ids(s, mapping):
    """把 catIds 字符串里的类目 ID 按 mapping 重新映射。"""
    if not isinstance(s, str) or s == '':
        return s
    parts = []
    for x in s.split(','):
        x = x.strip()
        if x == '':
            continue
        parts.append(str(mapping.get(int(x), int(x))))
    return ','.join(parts)


# ========== 4. 商品表 products.csv ==========
products = read_csv('products')
n0 = len(products)
products = products.drop_duplicates(subset=['productId'], keep='first')   # 商品 ID 去重
products['name']   = products['name'].map(clean_text)                     # 商品名清洗
products['catIds'] = products['catIds'].map(clean_cat_ids)                # 类目 ID 串清洗
products = products.dropna(subset=['name'])                               # 商品名为空的删除
products = products[products['name'] != '']
products = products.sort_values('productId').reset_index(drop=True)

# ========== 5. 类目表 categories.csv ==========
categories = read_csv('categories')
categories = categories.drop_duplicates(subset=['catId'], keep='first')   # 类目 ID 去重
categories['category'] = categories['category'].map(clean_text)           # 类目名清洗
categories = categories.dropna(subset=['category'])
categories = categories[categories['category'] != '']
categories = categories.sort_values('catId').reset_index(drop=True)

# ========== 6. 关联表 links.csv ==========
links = read_csv('links')
links = links.drop_duplicates(subset=['productId'], keep='first')         # 商品 ID 去重
links['amazonId'] = links['amazonId'].map(clean_text)                     # 亚马逊编号清洗
links = links.dropna(subset=['amazonId'])
links = links[links['amazonId'] != '']

# ========== 7. 评分/评论表 ratings.csv（核心，最大） ==========
ratings = read_csv('ratings')
print(f'[ratings] 原始行数：{len(ratings)}', flush=True)

# 7.0 关键：数据集中没有 users 表，用户全集 = “原始 ratings 中出现过的 userId”
users_orig = set(ratings['userId'].dropna().unique())

# 7.1 去重：同一用户对同一商品只保留“时间最新”的一条
ratings = ratings.sort_values(['userId', 'timestamp'])
ratings = ratings.drop_duplicates(subset=['userId', 'productId'], keep='last')

# 7.2 缺失值处理：userId / productId / rating 为空的行直接删除
ratings = ratings.dropna(subset=['userId', 'productId', 'rating'])

# 7.3 评分范围校验：rating 必须是 [1, 5] 之间的整数
ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]

# 7.4 类型转换（缺失值删除后，可以安全地转成整数）
ratings['userId']    = ratings['userId'].astype(int)
ratings['productId'] = ratings['productId'].astype(int)
ratings['rating']    = ratings['rating'].astype(int)
ratings['timestamp'] = ratings['timestamp'].astype(int)

# 7.5 文本列清洗：NaN 补成空字符串，再全角转半角
ratings['title']   = (ratings['title'].fillna('').astype(str).map(clean_text))
ratings['comment'] = ratings['comment'].fillna('').astype(str).map(clean_text)

# 7.6 用户一致性过滤（v2 新增）：
#     引用“不存在用户”的评分删除（以原始 ratings 全集为准）；非法（负数）userId 删除
users_orig_int = {int(x) for x in users_orig}
ratings = ratings[ratings['userId'].isin(users_orig_int)]
ratings = ratings[ratings['userId'] >= 0]

# 7.7 商品一致性过滤：删除引用了“不存在商品”的评分
valid_pids = set(products['productId'])
ratings = ratings[ratings['productId'].isin(valid_pids)]

ratings = ratings.sort_values(['userId', 'productId']).reset_index(drop=True)
print(f'[ratings] 清洗+一致性后行数：{len(ratings)}', flush=True)

# ========== 8. 跨表一致性校验 ==========
# 8.1 派生用户表 users.csv（数据没有用户表，用清洗后的 ratings 定义用户实体）
user_stats = ratings.groupby('userId', as_index=False).agg(ratingCount=('rating', 'size'))
users = user_stats.rename(columns={'rating': 'ratingCount'})[['userId', 'ratingCount']]

# 8.2 links ↔ products 双向校验（v2 新增）
links_before = len(links)
links = links[links['productId'].isin(valid_pids)]          # 孤儿 links 删除
links_orphan = links_before - len(links)

products_no_link = products[~products['productId'].isin(set(links['productId']))]
if DROP_PRODUCTS_NO_LINK:
    products = products[products['productId'].isin(set(links['productId']))]
    # 级联：删商品后，评分里引用这些商品的也要删
    ratings = ratings[ratings['productId'].isin(set(products['productId']))]

# 8.2.1 amazonId 冲突检查：同一个 amazonId 是否对应多个商品
dup_mask = links.duplicated('amazonId', keep=False)
amazon_multi = int((links[dup_mask].groupby('amazonId')['productId'].nunique() > 1).sum())

# 8.3 类目一致性（v2 新增）：products.catIds 引用的类目必须在 categories 中存在
valid_cats = set(categories['catId'])
catid_check = products['catIds'].map(lambda s: catids_valid(s, valid_cats))
products_bad_catid  = int((catid_check == False).sum())
products_no_catid   = int(products['catIds'].fillna('').astype(str).str.strip().eq('').sum())
if DROP_PRODUCTS_BAD_CATID:
    products = products[catid_check.fillna(True)]
    ratings = ratings[ratings['productId'].isin(set(products['productId']))]

# ========== 9. 重新连续编号（v2 新增，默认开启） ==========
# 清洗删除了行，ID 会留下空洞；重新编号恢复“从 0 开始连续编号”的约定，
# 并联动更新所有引用它的表，保证跨表引用一致。
if RENUMBER_IDS:
    # 9.1 先重编号类目（products.catIds 引用它）
    categories, cat_map = renumber(categories, 'catId')
    products['catIds'] = products['catIds'].map(lambda s: remap_cat_ids(s, cat_map))
    # 9.2 重编号商品（ratings / links 引用它）
    products, pid_map = renumber(products, 'productId')
    ratings['productId'] = ratings['productId'].map(pid_map)
    links['productId']   = links['productId'].map(pid_map)
    # 9.3 重编号用户（ratings 引用它）
    users, uid_map = renumber(users, 'userId')
    ratings['userId'] = ratings['userId'].map(uid_map)
    ratings = ratings.sort_values(['userId', 'productId']).reset_index(drop=True)

# ========== 10. 保存清洗后的全部表（统一 UTF-8） ==========
products.to_csv(os.path.join(OUT, 'cleaned_products.csv'),    index=False, encoding='utf-8')
categories.to_csv(os.path.join(OUT, 'cleaned_categories.csv'), index=False, encoding='utf-8')
links.to_csv(os.path.join(OUT, 'cleaned_links.csv'),          index=False, encoding='utf-8')
ratings.to_csv(os.path.join(OUT, 'cleaned_ratings.csv'),      index=False, encoding='utf-8')
users.to_csv(os.path.join(OUT, 'users.csv'),                  index=False, encoding='utf-8')
print('[保存] 清洗后的 5 张表已写入 cleaned/ 目录', flush=True)

# ========== 11. 结果汇总 ==========
print('\n' + '=' * 58)
print('清洗结果汇总')
print('=' * 58)
print(f'products  : {n0:>8} -> {len(products):>8} 行')
print(f'categories: {categories.shape[0]:>8} 行')
print(f'links     : {links_before:>8} -> {len(links):>8} 行')
print(f'users     : {len(users_orig):>8} -> {len(users):>8} 行（派生自 ratings）')
print(f'ratings   : {ratings.shape[0]:>8} 行')
print()
print('跨表一致性校验（v2）')
print('-' * 58)
print(f'用户：原始 {len(users_orig)} 人，清洗后 {len(users)} 人，'
      f'消失 {len(users_orig) - len(users)} 人（其评分全部被清洗）')
print(f'用户双向校验：ratings 引用的用户都在 users 表中：'
      f'{ratings["userId"].isin(set(users["userId"])).all()}')
print(f'孤儿 links（productId 不存在于 products）：删除 {links_orphan} 条')
print(f'products 中缺少 links 的商品：{len(products_no_link)} 个'
      f'（覆盖率 {len(links) / len(products) * 100:.2f}%）')
print(f'amazonId 一对多冲突：{amazon_multi} 个')
print(f'catIds 引用不存在类目的商品：{products_bad_catid} 个')
print(f'catIds 为空/缺失的商品：{products_no_catid} 个')
print(f'ID 重新连续编号：{"开启" if RENUMBER_IDS else "关闭"}')
print()
print(f'输出目录：{OUT}')
print(ratings['comment'].isna().sum())