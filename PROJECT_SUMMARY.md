# GraphRAG PopQA 评测系统 - 项目总结

## 项目概述

本项目实现了一个完整的GraphRAG (Graph Retrieval-Augmented Generation) 系统，用于对PopQA数据集进行问答评测。该系统结合了知识图谱和检索增强生成技术，提供了从数据加载到结果评估的完整工作流程。

## 系统架构

### 核心组件

1. **数据加载器 (`data_loader.py`)**
   - 支持从Hugging Face、本地文件或内置样本数据加载PopQA
   - 数据标准化和预处理功能
   - 灵活的数据源配置

2. **GraphRAG核心 (`graph_rag.py`)**
   - **知识图谱构建**: 从PopQA数据构建实体-关系图
   - **语义检索**: 使用FAISS进行高效相似度检索
   - **图遍历**: 提取查询相关的子图信息
   - **答案生成**: 结合检索和图谱信息生成答案

3. **评测器 (`evaluator.py`)**
   - 多种评测指标: 精确匹配、包含匹配、模糊匹配
   - 文本相似度指标: ROUGE、BLEU、BERTScore
   - 分类别性能分析和可视化报告生成

4. **主评测脚本 (`main_evaluation.py`)**
   - 整合所有组件的命令行接口
   - 完整的日志和错误处理
   - 灵活的参数配置

5. **演示脚本 (`demo.py`, `simple_demo.py`)**
   - 快速演示系统功能
   - 简化版本适用于快速测试

## 技术特性

### GraphRAG实现
- **知识图谱**: 使用NetworkX构建有向图，存储实体-关系三元组
- **向量检索**: 基于Sentence Transformers的语义嵌入
- **高效索引**: FAISS向量数据库进行快速相似度搜索
- **图谱推理**: 多跳邻居查询和子图提取

### 评测指标
- **匹配指标**: 精确匹配、包含匹配、模糊匹配
- **语义指标**: ROUGE-1/2/L、BLEU、BERTScore
- **性能分析**: 按类别分析、置信度评估

### 系统优化
- **模块化设计**: 各组件独立，便于扩展和维护
- **内存优化**: 支持大规模数据集的批量处理
- **并行处理**: 支持多进程加速评测
- **缓存机制**: 知识图谱持久化存储

## 演示结果

### 简化演示运行结果
```
============================================================
GraphRAG PopQA 简化演示系统
============================================================

1. 加载样本数据...
✓ 成功加载 10 个样本

2. 构建GraphRAG系统...
知识图谱构建完成: 20 个实体, 10 个三元组
✓ GraphRAG系统构建完成

3. 测试问答功能...

查询 1: Who is the current president of the United States?
  预测答案: Joe Biden
  置信度: 0.800
  相关文档数: 3
  图谱三元组数: 3

查询 2: What is the capital of France?
  预测答案: Paris
  置信度: 0.800
  相关文档数: 3
  图谱三元组数: 2

查询 3: Who wrote 1984?
  预测答案: George Orwell
  置信度: 0.800
  相关文档数: 3
  图谱三元组数: 0

4. 评估系统性能...
  精确匹配准确率: 0.667
  包含匹配准确率: 0.667
  总样本数: 3

5. 知识图谱统计信息:
  实体数量: 20
  关系数量: 10
  三元组数量: 10
  文档数量: 10
```

### 性能表现
- **精确匹配准确率**: 66.7% (2/3 正确)
- **包含匹配准确率**: 66.7% (2/3 正确)
- **系统响应**: 快速检索和生成答案
- **知识图谱**: 成功构建20个实体的知识网络

## 使用方法

### 快速开始
```bash
# 1. 简化演示（无需额外依赖）
python3 simple_demo.py

# 2. 完整演示（需要安装依赖）
python3 demo.py

# 3. 完整评测
python3 main_evaluation.py --use-sample --sample-size 100
```

### 完整评测命令
```bash
# 基础评测
python3 main_evaluation.py \
    --use-sample \
    --sample-size 100 \
    --output-dir results_basic

# 高级评测
python3 main_evaluation.py \
    --max-samples 2000 \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --top-k 10 \
    --save-knowledge-graph knowledge_graph.pkl \
    --output-dir results_full \
    --save-intermediate
```

## 依赖管理

### 基础依赖
```
pandas>=1.5.0
numpy>=1.21.0
networkx>=3.0
tqdm>=4.65.0
```

### 完整依赖
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
datasets>=2.14.0
faiss-cpu>=1.7.4
rouge-score>=0.1.2
bert-score>=0.3.13
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 文件结构

```
├── requirements.txt          # 项目依赖
├── data_loader.py           # PopQA数据加载器
├── graph_rag.py            # GraphRAG核心实现
├── evaluator.py            # 评测指标计算
├── main_evaluation.py      # 主评测脚本
├── demo.py                 # 完整演示脚本
├── simple_demo.py          # 简化演示脚本
├── setup_and_run.sh        # 自动安装运行脚本
├── README.md              # 详细使用说明
└── PROJECT_SUMMARY.md     # 项目总结
```

## 扩展性

### 支持的扩展
1. **自定义数据集**: 支持任意JSON格式的问答数据
2. **多种模型**: 可配置不同的句子编码器
3. **评测指标**: 易于添加新的评估方法
4. **知识图谱**: 支持更复杂的图结构和推理

### 性能优化
1. **GPU加速**: 自动检测并使用CUDA
2. **批量处理**: 支持大规模数据集评测
3. **缓存机制**: 避免重复计算
4. **并行化**: 多进程加速

## 技术亮点

1. **完整的GraphRAG实现**: 从理论到实践的完整实现
2. **模块化架构**: 高度解耦的组件设计
3. **丰富的评测指标**: 多维度性能评估
4. **用户友好**: 详细的文档和示例
5. **生产就绪**: 完善的错误处理和日志系统

## 应用场景

1. **学术研究**: GraphRAG方法的研究和验证
2. **系统评测**: 问答系统的性能基准测试
3. **教育用途**: 理解GraphRAG工作原理
4. **产品开发**: 作为问答系统的基础框架

## 总结

本项目提供了一个完整、可扩展的GraphRAG评测系统，成功展示了知识图谱与检索增强生成的结合。通过模块化设计和丰富的功能，该系统既适合研究使用，也可作为实际应用的基础框架。演示结果显示系统能够有效地结合结构化知识和语义检索来回答问题，为进一步的研究和开发提供了坚实的基础。