# -*- coding: utf-8 -*-
"""
TextCNN 情感分析推理程序（面向对象）
====================================================================
功能：输入一段中文评论文字，判断情感正/负向（好评/差评）并输出置信度。

输入：
    textcnn_best.pt   —— 训练脚本保存的模型权重
    train_prep.csv    —— 仅首次运行需要：用于重建词表（自动缓存为 vocab.json）
    stopwords.txt     —— 你自己的停用词文件（必须与训练时是同一个）
输出：
    控制台打印：评论 → 预测类别（好评/差评）+ 置信度

关键点：
    1) 推理的清洗、分词、去停用词必须与训练时完全一致，否则词表对不上。
       本程序已按 yf_amazon_label_preprocess.py 的规则 1:1 复刻。
    2) 训练只保存了模型权重，没有保存词表；本程序首次运行会从
       train_prep.csv 用与训练完全相同的逻辑重建词表并缓存为 vocab.json，
       之后再运行直接加载缓存（几秒）。

用法：
    D:\\PythonProject1\\.venv\\Scripts\\python.exe predict_sentiment.py
    D:\\PythonProject1\\.venv\\Scripts\\python.exe predict_sentiment.py "快递太慢了，客服也不理人"

依赖：torch、jieba、pandas、numpy（都在你训练用的 venv 里）
"""

import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import jieba

# ========== 1. 配置（改成你本地的实际路径） ==========
# 模型权重：你上次训练实际保存的位置
MODEL_PATH     = 'D:/PythonProject1/output/textcnn_best.pt'
# 词表缓存：自动生成，放在模型旁边即可
VOCAB_PATH     = 'D:/PythonProject1/yf_amazon/cleaned/vocab.json'
# 训练集：词表重建用（只读 tokens 列；若你的数据在 D:/shuju/yf_amazon/cleaned 就改这里）
TRAIN_CSV      = 'D:/PythonProject1/yf_amazon/cleaned/train_prep.csv'
# ← 你自己的停用词文件（与训练时同一个）
STOPWORDS_PATH = 'D:/PythonProject1/scripts/stopwords/hit_stopwords.txt'

# 模型超参（必须与训练脚本一致；改训练参数时这里要同步改）
EMBED_DIM    = 100
NUM_FILTERS  = 100
FILTER_SIZES = [2, 3, 4]
DROPOUT      = 0.5
MAX_LEN      = 64      # 与训练一致：超过截断
MIN_FREQ     = 2       # 与训练一致：重建词表时的最低词频

PAD_IDX, UNK_IDX = 0, 1   # 词表固定约定


# ========== 2. 模型类（与训练脚本的 TextCNN 完全一致） ==========
class TextCNN(nn.Module):
    """TextCNN（Yoon Kim 2014）：词嵌入 → 多宽度卷积 → 全局最大池化 → 全连接 → 二分类"""

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes,
                 num_classes=2, dropout=DROPOUT, pad_idx=PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x)              # (B, L, D)
        emb = emb.transpose(1, 2)            # (B, D, L)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))        # (B, C, L)
            pooled.append(torch.max_pool1d(c, c.size(2)).squeeze(2))
        out = torch.cat(pooled, dim=1)       # (B, C*核数)
        out = self.dropout(out)
        return self.fc(out)


# ========== 3. 文本预处理类（复刻训练脚本的清洗/分词/去停用词） ==========
class TextPreprocessor:
    """把一条评论转成与训练完全一致的词序列"""

    def __init__(self, stopwords_path):
        self.stopwords = self._load_stopwords(stopwords_path)

    @staticmethod
    def _load_stopwords(path):
        """与训练脚本相同：UTF-8(BOM)/GBK 自动识别，一行一个词或逗号/空格分隔"""
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
                print(f'[TextPreprocessor] 停用词 {len(words)} 个（编码 {enc}）', flush=True)
                return words
            except UnicodeDecodeError:
                continue
        raise ValueError('停用词文件编码无法识别，请确认是 UTF-8 或 GBK')

    @staticmethod
    def clean_text(s):
        """与训练脚本 clean_text 完全一致：去网址/邮箱 → 小写 → 去标点 → 去数字 → 压空白"""
        if not isinstance(s, str):
            return ''
        s = s.strip()
        s = re.sub(r'https?://\S+|www\.\S+', '', s)
        s = re.sub(r'\S+@\S+', '', s)
        s = s.lower()
        s = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', s)
        s = re.sub(r'\d+', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    def tokenize(self, text):
        """清洗 + jieba 分词 + 去停用词，返回词列表（与训练 tokens 列逻辑一致）"""
        cleaned = self.clean_text(text)
        return [w for w in jieba.lcut(cleaned) if w.strip() and w not in self.stopwords]


# ========== 4. 分类器封装（面向对象入口） ==========
class SentimentClassifier:
    """加载模型 + 词表 + 预处理，输入评论输出情感判断"""

    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH,
                 train_csv=TRAIN_CSV, stopwords_path=STOPWORDS_PATH, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = TextPreprocessor(stopwords_path)
        self.vocab = self._load_or_build_vocab(vocab_path, train_csv)
        self.model = self._load_model(model_path)
        if self.device.type == 'cuda':
            print(f'[SentimentClassifier] 推理设备 GPU: {torch.cuda.get_device_name(0)}', flush=True)
        else:
            print('[SentimentClassifier] 推理设备 CPU', flush=True)

    # ---------- 词表：优先加载缓存，没有就从训练集重建 ----------
    def _load_or_build_vocab(self, vocab_path, train_csv):
        if vocab_path and os.path.exists(vocab_path):
            vocab = json.load(open(vocab_path, encoding='utf-8'))
            vocab = {k: int(v) for k, v in vocab.items()}
            print(f'[词表] 加载缓存 {vocab_path}（{len(vocab)} 个词）', flush=True)
            return vocab

        print(f'[词表] 无缓存，从 {train_csv} 重建（只需一次，约 1 分钟）...', flush=True)
        df = pd.read_csv(train_csv, usecols=['tokens'])
        # 与训练 load_data 相同：排除 NaN 和空串（CSV 空字段读回是 NaN）
        df = df[df['tokens'].notna() & df['tokens'].str.strip().ne('')]
        counter = Counter()
        for toks in df['tokens']:
            counter.update(toks.split())
        words = [w for w, c in counter.items() if c >= MIN_FREQ]
        vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
        vocab.update({w: i + 2 for i, w in enumerate(words)})

        if vocab_path:
            with open(vocab_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False)
            print(f'[词表] 已缓存到 {vocab_path}（{len(vocab)} 个词）', flush=True)
        return vocab

    # ---------- 模型加载 ----------
    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'找不到模型文件：{model_path}\n'
                f'请先运行训练脚本生成 textcnn_best.pt，并确认 MODEL_PATH 正确')
        model = TextCNN(len(self.vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES,
                        dropout=DROPOUT, pad_idx=PAD_IDX)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device).eval()
        print(f'[模型] 已加载 {model_path}（词表 {len(self.vocab)}）', flush=True)
        return model

    # ---------- 推理 ----------
    @torch.no_grad()
    def predict_proba(self, text):
        """返回 [差评概率, 好评概率]"""
        toks = self.preprocessor.tokenize(text)
        if not toks:
            # 清洗后没有任何有效词（比如全是数字/停用词），无法判断
            return [0.5, 0.5]
        ids = [self.vocab.get(w, UNK_IDX) for w in toks][:MAX_LEN]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    def predict(self, text):
        """返回 (类别 0/1, 类别名, 置信度)"""
        proba = self.predict_proba(text)
        label = int(np.argmax(proba))
        name = '好评（正）' if label == 1 else '差评（负）'
        return label, name, float(proba[label])


# ========== 5. 命令行 / 交互式入口 ==========
def judge_and_print(clf, text):
    label, name, conf = clf.predict(text)
    print(f'评论：{text}')
    print(f'→ 情感：{name}（{"正向" if label == 1 else "负向"}）置信度 {conf * 100:.2f}%')


def main():
    # 方式一：命令行参数 python predict_sentiment.py "评论内容"
    if len(sys.argv) > 1:
        judge_and_print(SentimentClassifier(), ' '.join(sys.argv[1:]))
        return

    # 方式二：交互式循环输入
    clf = SentimentClassifier()
    print()
    print('=' * 52)
    print('TextCNN 情感分析 | 输入一条评论回车判断 | 输入 q/exit 退出')
    print('=' * 52)
    while True:
        try:
            text = input('\n请输入评论：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n已退出')
            break
        if not text:
            continue
        if text.lower() in ('q', 'quit', 'exit', '退出'):
            print('已退出')
            break
        judge_and_print(clf, text)


if __name__ == '__main__':
    main()
