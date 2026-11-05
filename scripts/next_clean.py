# -*- coding: utf-8 -*-
"""
yf_amazon 文本分类数据准备：构建二分类标签 → 8:2 划分 → 文本预处理
====================================================================
输入：
    cleaned_ratings.csv   （清洗脚本 yf_amazon_clean_split.py 的输出）
    停用词文件（你自己准备的，修改下方 STOPWORDS_PATH）
输出（保存在 OUT 目录）：
    train_prep.csv / test_prep.csv
    列：userId, productId, rating, label, text（合并清洗后）, tokens（分词去停用词，空格分隔）

流程顺序：
    1) 构建二分类标签：rating>=4 → 1（好评），rating<=2 → 0（差评），rating==3 → 删除
    2) 8:2 分层划分训练/测试集（stratify 保证好评:差评比例两边一致）
    3) 对训练集和测试集【分别】做：合并 title+comment 与清洗 → jieba 分词 → 去停用词

"""

import os
import re
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split

# ========== 1. 路径配置（改这里） ==========
BASE            = 'D:\PythonProject1\yf_amazon\cleaned'  # 清洗后数据所在目录
STOPWORDS_PATH  = '/scripts/stopwords\hit_stopwords.txt'  # ← 改成你自己的停用词文件路径
OUT             = BASE
RANDOM_STATE    = 42                              # 固定随机种子，保证结果可复现


# ========== 2. 加载停用词 ==========
def load_stopwords(path):
    """读取停用词文件：支持“一行一个词”，也支持逗号/空格/顿号分隔。
    编码自动尝试 UTF-8（带 BOM 也兼容），失败再试 GBK。"""
    words = set()
    for enc in ('utf-8-sig', 'gbk'):
        try:
            with open(path, encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    for w in re.split(r'[,，、;；\s]+', line):
                        if w:
                            words.add(w)
            print(f'[停用词] 已加载 {len(words)} 个（编码 {enc}）', flush=True)
            return words
        except UnicodeDecodeError:
            continue
    raise ValueError('停用词文件编码无法识别，请确认是 UTF-8 或 GBK')


stopwords = load_stopwords(STOPWORDS_PATH)


# ========== 3. 读取评分数据 + 构建二分类标签 ==========
print('[读取] cleaned_ratings.csv ...', flush=True)
ratings = pd.read_csv(os.path.join(BASE, 'cleaned_ratings.csv'),
                      usecols=['userId', 'productId', 'rating', 'title', 'comment'])

# 3.1 删除中性评分（rating==3）
ratings = ratings[ratings['rating'] != 3]

# 3.2 打标签：rating>=4 → 1（好评），rating<=2 → 0（差评）
ratings['label'] = (ratings['rating'] >= 4).astype(int)

n_pos = int((ratings['label'] == 1).sum())
n_neg = int((ratings['label'] == 0).sum())
print(f'[标签] 好评(label=1)：{n_pos} 条，差评(label=0)：{n_neg} 条，'
      f'比例 {n_pos / n_neg:.2f} : 1', flush=True)


# ========== 4. 8:2 划分（分层，保证两集类别比例一致） ==========
train, test = train_test_split(ratings, test_size=0.2,
                               random_state=RANDOM_STATE,
                               stratify=ratings['label'])
print(f'[划分] 训练集 {len(train)} 条（{len(train) / len(ratings) * 100:.1f}%），'
      f'测试集 {len(test)} 条（{len(test) / len(ratings) * 100:.1f}%）', flush=True)


# ========== 5. 文本合并与清洗 ==========
def clean_text(s):
    """单个文本的清洗：去网址/邮箱 → 英文小写 → 去标点符号 → 去数字 → 压缩空白"""
    if not isinstance(s, str):
        return ''
    s = s.strip()
    s = re.sub(r'https?://\S+|www\.\S+', '', s)      # 去网址
    s = re.sub(r'\S+@\S+', '', s)                    # 去邮箱
    s = s.lower()                                    # 英文转小写
    s = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', s)      # 去标点/符号（保留中文、英文、数字、下划线）
    s = re.sub(r'\d+', '', s)                        # 去数字（对情感判断基本无用；觉得有用可删这行）
    s = re.sub(r'\s+', ' ', s)                       # 压缩连续空白
    return s.strip()


def merge_text(title, comment):
    """合并标题和评论正文：空值补空串，中间用空格连接"""
    title   = title   if isinstance(title, str) else ''
    comment = comment if isinstance(comment, str) else ''
    return (title + ' ' + comment).strip()


def process_split(df, name):
    """对单个数据集执行：合并 → 清洗 → 分词 → 去停用词"""
    df = df.copy()
    # 5.1 合并 title + comment
    df['text'] = [merge_text(t, c) for t, c in zip(df['title'], df['comment'])]
    # 5.2 清洗
    df['text'] = df['text'].map(clean_text)
    # 5.3 清洗后为空（比如标题评论都空）的样本删除
    df = df[df['text'] != '']
    # 5.4 分词 + 去停用词（tokens 列用空格连接，方便直接喂给模型）
    df['tokens'] = df['text'].map(
        lambda s: ' '.join(w for w in jieba.lcut(s) if w.strip() and w not in stopwords)
    )
    # 5.5 只保留需要的列
    df = df[df['tokens'] != '']
    df = df[['userId', 'productId', 'rating', 'label', 'text', 'tokens']]
    df = df.reset_index(drop=True)
    out_path = os.path.join(OUT, f'{name}_prep.csv')
    df.to_csv(out_path, index=False, encoding='utf-8')
    avg_tokens = df['tokens'].str.split().str.len().mean()
    print(f'[完成] {name}：{len(df)} 条 → 已保存 {os.path.basename(out_path)}'
          f'（平均每个文本 {avg_tokens:.0f} 个词）', flush=True)
    return df


# ========== 6. 对训练集和测试集分别处理 ==========
train_prep = process_split(train, 'train')
test_prep  = process_split(test,  'test')


# ========== 7. 结果汇总 ==========
print('\n' + '=' * 55)
print('结果汇总')
print('=' * 55)
for name, df in (('train', train_prep), ('test', test_prep)):
    print(f'{name}: 共 {len(df)} 条 | 好评 {int((df["label"] == 1).sum())} 条 '
          f'| 差评 {int((df["label"] == 0).sum())} 条 '
          f'| 好评占比 {(df["label"] == 1).mean() * 100:.1f}%')
print(f'\n输出目录：{OUT}')


# ========== 备选：按用户划分 8:2（不按行随机，防止同一用户的数据跨集） ==========
# 需要时用下面代码替换第 4 节：
#
# shuffled = ratings.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
# counts = shuffled.groupby('userId')['userId'].transform('size').to_numpy()
# rank   = shuffled.groupby('userId').cumcount().to_numpy()
# train_n = np.rint(counts * 0.8).astype(int)
# train_n = np.where(counts == 1, 1, np.clip(train_n, 1, counts - 1))
# train = shuffled[rank < train_n].reset_index(drop=True)
# test  = shuffled[rank >= train_n].reset_index(drop=True)
