# 基于 TextCNN 的亚马逊电商评论情感二分类系统

面向对象设计的 PyTorch 情感分析项目：输入一条电商评论文本，自动判断其为**好评（label=1）**还是**差评（label=0）**。
数据为中文亚马逊电商评论（yf_amazon），在 RTX 4060 Laptop 上完成训练与推理。

---

## 一、项目简介

针对电商评论数据类别极不平衡（好评:差评 ≈ 10:1）的问题，采用 TextCNN 模型进行二分类情感分析，
以**负类 F1（差评召回）**为核心优化目标，并引入规则基线做 ΔF1 修正评估。

- 标签定义：rating ≥ 4 → 好评（1）；rating ≤ 2 → 差评（0）；rating = 3 → 删除（中性，不参与训练）
- 划分方式：按标签分层 8:2 划分训练集 / 测试集
- 预处理：jieba 分词 → 加载哈工大停用词表去停用词 → 按 MIN_FREQ=2 构建词表 → padding/truncation 到 MAX_LEN=64

## 二、环境依赖

| 依赖 | 说明 |
|---|---|
| Python 3.11 | 虚拟环境 `D:\PythonProject1\.venv` |
| PyTorch 2.x（CUDA） | GPU 训练与推理 |
| jieba | 中文分词 |
| pandas / scikit-learn | 数据处理、指标计算 |
| matplotlib | 报告图绘制 |
| tqdm | 进度条 |

## 三、项目结构

```
D:\PythonProject1\
├── scripts\
│   ├── TEXTCNN_1.py          # 训练主脚本（含日志双写、训练历史导出）
│   ├── demo.py               # 面向对象推理程序（输入评论 → 输出情感极性）
│   ├── visualize_report.py   # 5 张报告图生成脚本
│   ├── data_clean1.py        # 原始数据清洗
│   ├── next_clean.py         # 多表一致性校验（productId / userId / links）
│   └── stopwords\hit_stopwords.txt   # 哈工大停用词表
├── yf_amazon\cleaned\
│   ├── train_prep.csv        # 训练集（tokens、label 列）
│   ├── test_prep.csv         # 测试集
│   └── vocab.json            # 词表缓存（自动生成）
└── output\
    ├── textcnn_best.pt       # 最佳模型权重（按负类F1保存）
    ├── train_history.json    # 每轮 loss/acc/neg_f1/混淆矩阵
    └── log\train_时间戳.log   # 完整训练日志（控制台与文件双写）
```

## 四、模型结构

TextCNN（Kim, 2014）：

```
输入 token id (seq_len=64)
  → Embedding(vocab_size, embed_dim=100, padding_idx=0)
  → Conv1d 卷积核宽度 = {2, 3, 4}，每种 100 个通道
  → 各卷积层全局最大池化 (1-Max Pooling)
  → 拼接 → Dropout(0.5)
  → Linear(300 → 2) 二分类输出
```

- 损失函数：带类别权重的 CrossEntropyLoss（缓解好评/差评 10:1 失衡）
- 优化器：Adam，lr = 1e-3，共 5 个 epoch
- 训练加速：AMP 混合精度 + pin_memory + DataLoader(num_workers=4) + persistent_workers，BATCH_SIZE=512

## 五、评价指标

| 指标 | 含义 |
|---|---|
| **准确率（主）** | 整体分类正确率 |
| **负类 F1** | 差评（label=0）的 F1——类别不平衡场景下的核心指标 |
| **ΔF1** | 模型负类 F1 − 规则基线负类 F1。规则基线 = 训练集统计的 Top200 差评关键词简单分类器 |

## 六、实验结果（测试集约 126 万条）

| 模型 | 准确率 | 负类 F1 |
|---|---|---|
| **TextCNN（本项目）** | **0.9356** | **0.7096** |
| 规则基线（Top200 差评关键词） | 0.9278 | 0.3555 |
| **ΔF1** | — | **+0.3541** |

最终混淆矩阵（行=真实 差评/好评，列=预测）：

```
[[  99182    9802]    ← 差评：召回率约 91%
 [  71394 1080779]]
```

## 七、快速开始

```powershell
# 1. 训练（确保 train_prep.csv / test_prep.csv 已就绪）
D:\PythonProject1\.venv\Scripts\python.exe D:\PythonProject1\scripts\TEXTCNN_1.py

# 2. 生成 5 张报告图（loss 曲线 / acc 曲线 / 词频柱状图 / 混淆矩阵 / 正负样本饼图）
D:\PythonProject1\.venv\Scripts\python.exe D:\PythonProject1\scripts\visualize_report.py

# 3. 单条评论推理（交互式输入评论，输出好评/差评）
D:\PythonProject1\.venv\Scripts\python.exe D:\PythonProject1\scripts\demo.py
```

> 注：首次运行 `demo.py` 时会自动从 `train_prep.csv` 重建词表并缓存为 `vocab.json`（约 1 分钟），之后秒出结果。

## 八、关键设计说明

1. **词表未随模型保存**：推理脚本首次运行时自动从训练集重建词表，故 `demo.py` 中的 `TRAIN_CSV` 必须与训练时一致。
2. **路径集中在脚本顶部配置区**：`MODEL_PATH / VOCAB_PATH / TRAIN_CSV / STOPWORDS_PATH` 修改一处即可。
3. **Windows 多进程注意**：训练脚本必须有 `if __name__ == '__main__'` 保护，否则 `num_workers>0` 会重复启动进程。
4. **空值兜底**：预处理阶段删除空 tokens 行，训练侧另有 `notna()` 兜底防止 NaN 进入分词。

## 九、产物清单

- 模型权重：`output/textcnn_best.pt`（约 101 MB）
- 训练曲线数据：`output/train_history.json`
- 训练日志：`output/log/train_时间戳.log`
- 5 张报告图：运行 `visualize_report.py` 后生成（输出目录见脚本顶部 `OUTPUT_DIR`）
