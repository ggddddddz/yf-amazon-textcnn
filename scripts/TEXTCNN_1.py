# -*- coding: utf-8 -*-
"""
yf_amazon 文本情感分类：TextCNN（GPU 加速 + tqdm 进度条）
====================================================================
输入：
    train_prep.csv / test_prep.csv
    （yf_amazon_label_preprocess.py 的输出：列含 tokens、label）
输出：
    textcnn_best.pt  （测试集上指定指标最好的模型权重）
    控制台打印每个 epoch：训练损失、准确率(主)、负类F1、ΔF1、规则基线指标、混淆矩阵

评价指标（按你的要求）：
    准确率（主）
    负类 F1（差评类 label=0，重点关注）
    规则修正 ΔF1 = 模型负类F1 − 规则基线负类F1
    规则基线 = 基于“差评关键词”的简单规则分类器
              （关键词从训练集统计得到，无需外部词典；见 build_neg_keywords）

GPU 提速手段（v3）：
    1) 多进程数据加载：DataLoader(num_workers=4)，CPU 并行取 batch，
       不再让 GPU 等 CPU —— 这是之前慢的主要原因（原 num_workers=0 单进程）
    2) pin_memory=True：batch 放进锁页内存，加速 CPU→GPU 拷贝
    3) 混合精度 AMP（autocast + GradScaler）：RTX 40 系显卡 FP16 约快 1.5~2 倍
    4) BATCH_SIZE 提到 512、MAX_LEN 降到 64（绝大多数评论远短于此）
    5) cudnn.benchmark 已开启
    注：Windows 下必须用 if __name__ == '__main__' 保护主流程，
        多进程数据加载才不报错（否则每个 worker 会把整个脚本重跑一遍）。

依赖安装（运行前）：
    pip install torch tqdm scikit-learn
    # 确认 GPU 可用：python -c "import torch; print(torch.cuda.is_available())"  → True

用法：
    python yf_amazon_textcnn.py
    调试期建议把 MAX_SAMPLES 设成 50000 先跑通，再改回 None 全量训练。
"""
import json
import os
import sys
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ========== 1. 配置 ==========
TRAIN_CSV    = r'D:\PythonProject1\yf_amazon\cleaned\train_prep.csv'
TEST_CSV     = r'D:\PythonProject1\yf_amazon\cleaned\test_prep.csv'
MODEL_SAVE   = r'D:\PythonProject1\output\textcnn_best.pt'
# 训练历史自动保存位置（每轮 loss/acc/负类F1/混淆矩阵），可视化脚本 visualize_report.py 读取
HISTORY_SAVE = os.path.join(os.path.dirname(MODEL_SAVE), 'train_history.json')
# 每次运行导出训练日志：log/train_时间戳.log（控制台输出与文件同步双写）

LOG_DIR  = os.path.join(os.path.dirname(MODEL_SAVE), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")

EMBED_DIM    = 100      # 词向量维度
NUM_FILTERS  = 100      # 每种卷积核的输出通道数
FILTER_SIZES = [2, 3, 4]  # 卷积核宽度（滑动 n-gram）
DROPOUT      = 0.5
BATCH_SIZE   = 512      # 加大 batch 提高 GPU 利用率（4060 8G 显存可跑 512）
EPOCHS       = 5
LR           = 1e-3
MAX_LEN      = 64       # 文本最长截断词数：绝大多数评论远短于此，调低显著提速
MIN_FREQ     = 2        # 词频 < 2 的词不进词表（视为 <UNK>）
MAX_SAMPLES  = None     # 调试用：如 50000 表示只取前 5 万条；None=全量
RANDOM_SEED  = 42

# —— 提速相关 ——
NUM_WORKERS = 4         # 数据加载进程数（CPU 并行取 batch，Windows 建议 2~8）
USE_AMP     = True      # 混合精度（仅 GPU 生效，RTX 40 系收益明显）

BEST_METRIC   = 'neg_f1'    # 保存最佳模型的依据：'neg_f1'（负类F1）或 'accuracy'
# —— 规则基线参数（差评关键词规则，无需外部词典）——
RULE_MIN_DOC  = 50        # 差评关键词：至少出现在多少条差评中
RULE_MIN_RATIO = 0.5      # 差评关键词：该词出现在差评中的最低占比
RULE_TOP_K    = 200       # 差评关键词：最多取多少个
RULE_HIT      = 1         # 规则：命中 >= 1 个差评关键词即判为差评

# 词表固定约定：<PAD>=0（补齐位）、<UNK>=1（未登录词），构建词表时保持此约定
PAD_IDX, UNK_IDX = 0, 1


# ========== 2. 固定随机种子（保证可复现） ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========== 3. 加载数据 ==========
def load_data(path, max_samples=None):
    df = pd.read_csv(path, usecols=['tokens', 'label'])
    # 关键：先排除 NaN（CSV 空字段读回后是 NaN），再排除纯空字符串。
    # 只写 df['tokens'].str.strip().ne('') 会漏掉 NaN（NaN != '' 为 True），导致 split 报错
    df = df[df['tokens'].notna() & df['tokens'].str.strip().ne('')]
    if max_samples is not None:
        df = df.head(max_samples)
    return df


# ========== 4. Dataset 与 batch 组装 ==========
class TextDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.ids = [
            [vocab.get(w, UNK_IDX) for w in toks.split()][:max_len]
            for toks in df['tokens']
        ]
        self.labels = df['label'].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long), \
               torch.tensor(self.labels[i], dtype=torch.long)


def collate_fn(batch):
    """把长度不一的序列补齐成同长度（按批内最长），不足补 <PAD>"""
    xs, ys = zip(*batch)
    xs = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    return xs, torch.stack(ys)


# ========== 5. TextCNN 模型 ==========
class TextCNN(nn.Module):
    """经典 TextCNN（Yoon Kim, 2014）：
    词嵌入 → 多宽度一维卷积(2/3/4-gram) → 全局最大池化 → 拼接 → 全连接 → 二分类
    """

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
        emb = self.embedding(x)              # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)            # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))        # (batch, num_filters, seq_len)
            p = torch.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ========== 6. 规则基线：差评关键词规则（从训练集统计，无需外部词典） ==========
def build_neg_keywords(train_df, min_doc=RULE_MIN_DOC, min_ratio=RULE_MIN_RATIO, top_k=RULE_TOP_K):
    """挑出“差评倾向词”：在差评中出现次数够多、且差评占比高的词"""
    neg_doc, pos_doc = Counter(), Counter()
    for toks, lab in zip(train_df['tokens'], train_df['label']):
        words = set(toks.split())
        (neg_doc if lab == 0 else pos_doc).update(words)
    cand = []
    for w, nc in neg_doc.items():
        if nc < min_doc:
            continue
        pc = pos_doc.get(w, 0)
        ratio = nc / (nc + pc)
        if ratio >= min_ratio:
            cand.append((ratio, nc, w))
    cand.sort(reverse=True)
    return [w for _, _, w in cand[:top_k]]


def rule_predict(tokens_str, neg_keywords, hit=RULE_HIT):
    """简单规则：命中差评关键词 >= hit 个 → 差评(0)，否则好评(1)"""
    words = set(tokens_str.split())
    return int(sum(w in neg_keywords for w in words) < hit)


# ========== 7. 训练一个 epoch（tqdm 进度条 + AMP 混合精度） ==========
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch, epochs):
    model.train()
    total_loss, n_batch = 0.0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [train]', unit='batch', leave=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):      # 混合精度前向
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()                       # GradScaler 防梯度下溢
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batch += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / n_batch


# ========== 8. 测试集评估：准确率(主) + 负类F1（tqdm 进度条） ==========
def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, trues = [], []
    pbar = tqdm(loader, desc='[eval]', unit='batch', leave=True)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
    acc    = accuracy_score(trues, preds)
    f1_neg = f1_score(trues, preds, pos_label=0)   # 负类（差评，label=0）的 F1
    cm     = confusion_matrix(trues, preds)
    return acc, f1_neg, cm

# ========== 8.5 日志导出：控制台与文件双写 ==========
class Tee:
    """把 print 的输出同时写到控制台和日志文件"""

    def __init__(self, stream, log_f):
        self.stream = stream
        self.log_f = log_f

    def write(self, data):
        self.stream.write(data)
        self.log_f.write(data)
        self.log_f.flush()

    def flush(self):
        self.stream.flush()
        self.log_f.flush()


# ========== 9. 主流程 ==========
def main():
    set_seed(RANDOM_SEED)
    # 日志导出：本次运行的所有输出（设备/数据/每轮指标/混淆矩阵/最终结果）
    # 同步写入 logs/train_时间戳.log；控制台照常显示
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    _log_f = open(LOG_FILE, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, _log_f)
    sys.stderr = Tee(sys.stderr, _log_f)
    print(f'[日志] 本次训练日志文件：{LOG_FILE}', flush=True)

    # 设备：优先本地 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'[设备] GPU: {torch.cuda.get_device_name(0)}', flush=True)
        torch.backends.cudnn.benchmark = True   # 输入尺寸变化不大时加速
    else:
        print('[设备] CPU（无 CUDA，会很慢，请安装 CUDA 版 PyTorch）', flush=True)
    use_amp = USE_AMP and device.type == 'cuda'

    # 加载数据 + 构建词表（只用训练集，测试集不参与，避免信息泄漏）
    train_df = load_data(TRAIN_CSV, MAX_SAMPLES)
    test_df  = load_data(TEST_CSV,  MAX_SAMPLES)
    print(f'[数据] 训练集 {len(train_df)} 条，测试集 {len(test_df)} 条', flush=True)

    counter = Counter()
    for toks in train_df['tokens']:
        counter.update(toks.split())
    words = [w for w, c in counter.items() if c >= MIN_FREQ]
    vocab = {'<PAD>': PAD_IDX, '<UNK>': UNK_IDX}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    print(f'[词表] 大小 {len(vocab)}（含 <PAD>、<UNK>；训练集去重词 {len(counter)} 个）', flush=True)

    # DataLoader：多进程 + 锁页内存（提速关键）
    train_ds = TextDataset(train_df, vocab, MAX_LEN)
    test_ds  = TextDataset(test_df,  vocab, MAX_LEN)
    # num_workers>0：CPU 多进程并行取 batch，GPU 不用等 CPU（Windows 需 __main__ 保护，已加）
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)

    # 模型
    model = TextCNN(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[模型] 参数量 {n_params / 1e6:.2f}M', flush=True)

    # 加权损失（处理好评:差评 ≈ 10:1 的不均衡）
    counts = train_df['label'].value_counts().sort_index().to_numpy()
    class_weights = len(train_df) / (counts * len(counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f'[不均衡] 差评 {counts[0]} / 好评 {counts[1]}，类别权重 {class_weights.tolist()}', flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 规则基线（测试集上算一次，不随 epoch 变化）
    neg_keywords = build_neg_keywords(train_df)
    print(f'[规则基线] 差评关键词 {len(neg_keywords)} 个（示例：{" ".join(neg_keywords[:10])}）', flush=True)
    rule_trues  = test_df['label'].astype(int).tolist()
    rule_preds  = [rule_predict(t, neg_keywords) for t in test_df['tokens']]
    rule_acc    = accuracy_score(rule_trues, rule_preds)
    rule_f1_neg = f1_score(rule_trues, rule_preds, pos_label=0)
    print(f'[规则基线] 测试集 准确率 {rule_acc:.4f} | 负类F1 {rule_f1_neg:.4f}', flush=True)

    # 训练 + 评估
    history = []          # 每轮指标存档，训练结束写入 HISTORY_SAVE 供可视化
    best_score = -1.0
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                   device, use_amp, epoch, EPOCHS)
        acc, f1_neg, cm = evaluate(model, test_loader, device, use_amp)
        delta_f1 = f1_neg - rule_f1_neg          # 规则修正 ΔF1
        print(f'  → 训练损失 {avg_loss:.4f} | 准确率(主) {acc:.4f} | 负类F1 {f1_neg:.4f} | ΔF1 {delta_f1:+.4f}', flush=True)
        print(f'    规则基线：准确率 {rule_acc:.4f} | 负类F1 {rule_f1_neg:.4f}', flush=True)
        print(f'    混淆矩阵（行=真实 差评/好评，列=预测）：\n{cm}', flush=True)

        score = f1_neg if BEST_METRIC == 'neg_f1' else acc
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f'  → 已保存最佳模型到 {MODEL_SAVE}（{BEST_METRIC}={score:.4f}）', flush=True)

        history.append({'epoch': epoch,
                        'train_loss': round(float(avg_loss), 6),
                        'acc': round(float(acc), 6),
                        'neg_f1': round(float(f1_neg), 6),
                        'delta_f1': round(float(delta_f1), 6),
                        'cm': cm.tolist()})

    # 保存训练历史：loss/acc 曲线 + 混淆矩阵 + 规则基线（可视化脚本读取）
    with open(HISTORY_SAVE, 'w', encoding='utf-8') as f:
        json.dump({'history': history,
                   'rule_baseline': {'acc': round(float(rule_acc), 6),
                                     'neg_f1': round(float(rule_f1_neg), 6)},
                   'best_metric': BEST_METRIC,
                   'best_score': round(float(best_score), 6),
                   'train_size': int(len(train_df)),
                   'test_size': int(len(test_df))},
                  f, ensure_ascii=False, indent=2)
    print(f'[历史] 训练曲线/混淆矩阵已保存到 {HISTORY_SAVE}', flush=True)

    print(f'\n[完成] 最佳{BEST_METRIC} = {best_score:.4f}，模型文件：{MODEL_SAVE}')
    print(f'最终指标：准确率(主) {acc:.4f} | 负类F1 {f1_neg:.4f} | '
          f'规则基线负类F1 {rule_f1_neg:.4f} | ΔF1 {delta_f1:+.4f}')


if __name__ == '__main__':
    main()
