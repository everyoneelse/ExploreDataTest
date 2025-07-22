# GraphRAG PopQA 评测系统

这是一个使用GraphRAG (Graph Retrieval-Augmented Generation) 对PopQA数据集进行问答评测的完整系统。

## 系统概述

该系统结合了知识图谱和检索增强生成技术，通过以下方式回答问题：
1. 构建基于PopQA数据的知识图谱
2. 使用语义检索找到相关文档
3. 从知识图谱中提取相关三元组
4. 结合文档和图谱信息生成答案

## 文件结构

```
├── requirements.txt          # 项目依赖
├── data_loader.py           # PopQA数据加载器
├── graph_rag.py            # GraphRAG核心实现
├── evaluator.py            # 评测指标计算
├── main_evaluation.py      # 主评测脚本
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

注意：如果在安装过程中遇到问题，建议使用conda环境：

```bash
conda create -n graphrag python=3.8
conda activate graphrag
pip install -r requirements.txt
```

## 快速开始

### 1. 使用样本数据进行测试

```bash
python main_evaluation.py --use-sample --sample-size 50
```

### 2. 从Hugging Face加载完整数据集

```bash
python main_evaluation.py --max-samples 1000
```

### 3. 使用本地数据文件

```bash
python main_evaluation.py --data-file path/to/your/popqa.json --max-samples 500
```

## 详细使用说明

### 命令行参数

#### 数据相关参数
- `--data-file`: 本地PopQA数据文件路径（JSON或JSONL格式）
- `--use-sample`: 使用内置样本数据进行测试
- `--sample-size`: 样本数据大小（默认100）
- `--max-samples`: 最大处理样本数量限制

#### 模型相关参数
- `--model-name`: 句子编码器模型名称（默认：sentence-transformers/all-MiniLM-L6-v2）
- `--top-k`: 检索相关文档数量（默认5）

#### 知识图谱相关参数
- `--knowledge-graph-path`: 预训练知识图谱文件路径
- `--save-knowledge-graph`: 保存知识图谱的路径

#### 输出相关参数
- `--output-dir`: 结果输出目录（默认：evaluation_results）
- `--save-intermediate`: 保存中间结果

#### 其他参数
- `--log-level`: 日志级别（DEBUG/INFO/WARNING/ERROR）

### 使用示例

#### 基础评测
```bash
python main_evaluation.py \
    --use-sample \
    --sample-size 100 \
    --output-dir results_basic
```

#### 完整评测
```bash
python main_evaluation.py \
    --max-samples 2000 \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --top-k 10 \
    --save-knowledge-graph knowledge_graph.pkl \
    --output-dir results_full \
    --save-intermediate
```

#### 使用预训练知识图谱
```bash
python main_evaluation.py \
    --knowledge-graph-path knowledge_graph.pkl \
    --max-samples 1000 \
    --output-dir results_pretrained
```

## 评测指标

系统使用多种评测指标来全面评估性能：

### 匹配指标
- **精确匹配 (Exact Match)**: 预测答案与标准答案完全匹配
- **包含匹配 (Contains Match)**: 预测答案包含标准答案或被包含
- **模糊匹配 (Fuzzy Match)**: 基于编辑距离的相似度匹配（阈值0.8）

### 文本相似度指标
- **ROUGE-1/2/L**: 基于n-gram重叠的文本相似度
- **BLEU**: 机器翻译质量评估指标
- **BERTScore**: 基于BERT的语义相似度评估

## 输出文件

评测完成后，系统会在输出目录中生成以下文件：

```
evaluation_results/
├── evaluation_report.txt      # 文本格式评测报告
├── detailed_results.json      # 详细评测结果
├── predictions.json          # 所有预测结果
├── config.json              # 评测配置信息
├── overall_performance.png   # 整体性能可视化
└── category_performance.png  # 分类别性能可视化
```

## 系统架构

### 1. 数据加载器 (data_loader.py)
- 支持从Hugging Face、本地文件或样本数据加载PopQA
- 数据标准化和预处理

### 2. GraphRAG核心 (graph_rag.py)
- **知识图谱构建**: 从PopQA数据构建实体-关系图
- **语义检索**: 使用FAISS进行高效相似度检索
- **图遍历**: 提取查询相关的子图信息
- **答案生成**: 结合检索和图谱信息生成答案

### 3. 评测器 (evaluator.py)
- 多种评测指标计算
- 分类别性能分析
- 可视化报告生成

### 4. 主评测脚本 (main_evaluation.py)
- 整合所有组件
- 命令行接口
- 日志和错误处理

## 性能优化建议

### 1. 模型选择
- 快速测试: `sentence-transformers/all-MiniLM-L6-v2`
- 高精度: `sentence-transformers/all-mpnet-base-v2`
- 多语言: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

### 2. 参数调优
- 增加`top-k`值可能提高召回率但增加计算成本
- 较大的`sample-size`用于初步测试
- 使用`--save-knowledge-graph`避免重复构建图谱

### 3. 硬件要求
- 推荐GPU加速（自动检测CUDA）
- 至少8GB内存用于大规模评测
- SSD存储提高I/O性能

## 扩展功能

### 自定义数据格式
如果您有自定义的问答数据，请确保JSON格式如下：
```json
[
  {
    "id": 0,
    "question": "问题文本",
    "answer": "答案文本",
    "subject": "主体实体",
    "relation": "关系",
    "object": "客体实体"
  }
]
```

### 添加新的评测指标
在`evaluator.py`中的`PopQAEvaluator`类中添加新方法：
```python
def your_custom_metric(self, predicted: str, ground_truth: str) -> float:
    # 实现您的评测逻辑
    return score
```

## 常见问题

### Q: 如何处理大规模数据集？
A: 使用`--max-samples`限制样本数量，使用`--save-intermediate`保存中间结果，考虑分批处理。

### Q: 评测速度很慢怎么办？
A: 使用更小的模型，减少`top-k`值，使用GPU加速，预先构建知识图谱。

### Q: 如何提高评测精度？
A: 使用更大的预训练模型，增加检索文档数量，优化实体提取算法。

### Q: 支持中文数据吗？
A: 支持，使用多语言模型如`paraphrase-multilingual-mpnet-base-v2`。

## 贡献和反馈

欢迎提交Issue和Pull Request来改进这个系统。

## 许可证

MIT License