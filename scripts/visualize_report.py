# -*- coding: utf-8 -*-
"""
TextCNN 训练结果可视化脚本（5 张图，面向对象）
====================================================================
输出 5 张 PNG：
    1) 训练 loss 曲线        —— 每个 epoch 的平均训练损失
    2) 验证 acc 曲线         —— 测试集 准确率(主)/负类F1/ΔF1（随 epoch 变化）
    3) 词频柱状图            —— 训练集 Top-N 高频词
    4) 混淆矩阵热力图        —— 测试集（末轮评估结果）
    5) 正负样本分布饼图      —— 好评 vs 差评

数据来源：
    - 曲线 & 混淆矩阵：train_history.json
      （由训练脚本 textcnn.py 自动保存；需用更新后的 textcnn.py 重跑一次训练生成）
    - 词频 & 样本分布：train_prep.csv / test_prep.csv
依赖：
    pip install matplotlib    （其余 torch/pandas/jieba 训练时已装）
用法：
    D:\\PythonProject1\\.venv\\Scripts\\python.exe visualize_report.py
    把下方 4 个路径改成你本地的实际路径后运行。
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')            # 后台模式：不弹窗口，直接存图
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']   # 中文字体
    plt.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    raise RuntimeError(
        '未安装 matplotlib，请先运行：\n'
        '    D:\\PythonProject1\\.venv\\Scripts\\pip install matplotlib') from e


# ========== 1. 配置（改这里） ==========
OUTPUT_DIR = 'D:\PythonProject1\output\log'      # ← 你自己填输出目录，如 'D:/project/charts'；留空 = 当前目录
TRAIN_CSV  = r'D:/PythonProject1/yf_amazon/cleaned/train_prep.csv'
TEST_CSV   = r'D:/PythonProject1/yf_amazon/cleaned/test_prep.csv'
# ← 与训练脚本保存 train_history.json 的位置一致（默认在模型同目录）
HISTORY    = r'D:\PythonProject1\output\train_history.json'
TOP_WORDS  = 30      # 词频柱状图展示前 N 个高频词
DPI        = 150     # 图片清晰度


# ========== 2. 可视化器（每个图表一个方法） ==========
class ReportVisualizer:
    """读取训练历史 + 预处理数据，输出 5 张 PNG 到 OUTPUT_DIR"""

    def __init__(self, output_dir=OUTPUT_DIR, train_csv=TRAIN_CSV,
                 test_csv=TEST_CSV, history_path=HISTORY, top_words=TOP_WORDS):
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.top_words = top_words
        self.history = self._load_history(history_path)
        print(f'[可视化] 输出目录：{self.output_dir}', flush=True)

    def _load_history(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'找不到训练历史：{path}\n'
                f'请用更新后的 textcnn.py（已加 [历史] 保存功能）重跑一次训练，'
                f'生成 train_history.json 后再运行本脚本')
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    # ---------- 1) 训练 loss 曲线 ----------
    def plot_loss_curve(self):
        hist = self.history['history']
        epochs = [h['epoch'] for h in hist]
        losses = [h['train_loss'] for h in hist]
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, losses, 'o-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('训练损失')
        plt.title(f'训练 Loss 曲线（{len(epochs)} 个 epoch，末轮 {losses[-1]:.4f}）')
        plt.grid(alpha=0.3)
        plt.xticks(epochs)
        out = os.path.join(self.output_dir, '1_训练loss曲线.png')
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  已生成 {out}', flush=True)

    # ---------- 2) 验证 acc 曲线（附 负类F1 / ΔF1） ----------
    def plot_acc_curve(self):
        hist = self.history['history']
        epochs = [h['epoch'] for h in hist]
        accs = [h['acc'] for h in hist]
        f1s = [h['neg_f1'] for h in hist]
        deltas = [h['delta_f1'] for h in hist]
        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, accs, 'o-', linewidth=2, label='准确率(主)')
        plt.plot(epochs, f1s, 's--', linewidth=1.8, label='负类F1')
        plt.plot(epochs, deltas, '^:', linewidth=1.6, label='ΔF1')
        plt.xlabel('Epoch')
        plt.ylabel('指标')
        plt.title('验证曲线（测试集）')
        plt.grid(alpha=0.3)
        plt.xticks(epochs)
        plt.legend()
        out = os.path.join(self.output_dir, '2_验证acc曲线.png')
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  已生成 {out}', flush=True)

    # ---------- 3) 词频柱状图（训练集 Top-N） ----------
    def plot_word_freq(self):
        print(f'  统计训练集词频（读 {self.train_csv}，首次较慢）...', flush=True)
        df = pd.read_csv(self.train_csv, usecols=['tokens'])
        df = df[df['tokens'].notna() & df['tokens'].str.strip().ne('')]
        counter = Counter()
        for toks in df['tokens']:
            counter.update(toks.split())
        top = counter.most_common(self.top_words)
        words = [w for w, _ in top][::-1]        # 倒序，让最高的在顶部
        freqs = [c for _, c in top][::-1]

        plt.figure(figsize=(8, max(4.5, len(words) * 0.35)))
        plt.barh(words, freqs)
        plt.xlabel('出现次数')
        plt.ylabel('词')
        plt.title(f'训练集词频 Top{self.top_words}')
        for i, v in enumerate(freqs):
            plt.text(v, i, f' {v:,}', va='center', fontsize=8)
        out = os.path.join(self.output_dir, '3_词频柱状图.png')
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  已生成 {out}', flush=True)

    # ---------- 4) 混淆矩阵热力图（测试集，末轮） ----------
    def plot_confusion_matrix(self):
        cm = np.array(self.history['history'][-1]['cm'])   # 末轮评估结果
        total = cm.sum()
        labels = ['差评', '好评']
        plt.figure(figsize=(6.5, 5.5))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar(shrink=0.85, label='样本数')
        plt.xticks([0, 1], [f'预测\n{labels[0]}', f'预测\n{labels[1]}'])
        plt.yticks([0, 1], [f'真实\n{labels[0]}', f'真实\n{labels[1]}'])
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                color = 'white' if v > cm.max() * 0.5 else 'black'
                plt.text(j, i, f'{v:,}\n{v / total * 100:.2f}%',
                         ha='center', va='center', color=color, fontsize=11)
        plt.title(f'混淆矩阵（测试集 {total:,} 条，末轮）')
        out = os.path.join(self.output_dir, '4_混淆矩阵.png')
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  已生成 {out}', flush=True)

    # ---------- 5) 正负样本分布饼图 ----------
    def plot_label_pie(self):
        n_train_pos = int((pd.read_csv(self.train_csv, usecols=['label'])['label'] == 1).sum())
        n_train_neg = int((pd.read_csv(self.train_csv, usecols=['label'])['label'] == 0).sum())
        n_test_pos = int((pd.read_csv(self.test_csv, usecols=['label'])['label'] == 1).sum())
        n_test_neg = int((pd.read_csv(self.test_csv, usecols=['label'])['label'] == 0).sum())

        pos = n_train_pos + n_test_pos
        neg = n_train_neg + n_test_neg
        plt.figure(figsize=(6.5, 5.5))
        plt.pie([neg, pos], labels=[f'差评（负）\n{neg:,}', f'好评（正）\n{pos:,}'],
                autopct='%1.1f%%', startangle=90,
                colors=['#c62828', '#2e7d32'], explode=(0, 0.03))
        plt.title(f'正负样本分布（训练+测试，共 {pos + neg:,} 条）')
        out = os.path.join(self.output_dir, '5_正负样本分布饼图.png')
        plt.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f'  已生成 {out}', flush=True)

    # ---------- 一键生成全部 ----------
    def generate_all(self):
        self.plot_loss_curve()
        self.plot_acc_curve()
        self.plot_word_freq()
        self.plot_confusion_matrix()
        self.plot_label_pie()


# ========== 3. 入口 ==========
if __name__ == '__main__':
    ReportVisualizer().generate_all()
    print('\n[完成] 5 张图已全部生成，输出目录见上方。')
