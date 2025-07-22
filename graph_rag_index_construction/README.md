# Graph-based RAG Index Construction Methods

本项目实现了《In-depth Analysis of Graph-based RAG in a Unified Framework》论文中提到的三大类Index Construction方法，提供了完整的图索引构建解决方案。

## 📖 项目概述

Graph-based RAG通过构建知识图谱来增强检索增强生成（RAG）系统的性能。本项目实现了三大类索引构建方法：

### 1. Node Index (节点索引) 🔗
- **Entity-based Node Index**: 基于命名实体识别构建节点索引，适用于知识图谱构建
- **Document-based Node Index**: 将文档作为节点，基于相似度建立连接，适用于文档检索
- **Concept-based Node Index**: 基于概念抽取构建索引，适用于主题建模和概念导航
- **Hierarchical Node Index**: 分层节点索引，支持多层次信息组织

### 2. Relationship Index (关系索引) 🔄
- **Semantic Relationship Index**: 语义关系索引，识别文档间的语义关系
- **Citation Relationship Index**: 引用关系索引，分析文档引用网络
- **Co-occurrence Relationship Index**: 共现关系索引，基于实体共现模式
- **Temporal Relationship Index**: 时序关系索引，捕获时间序列关系

### 3. Community Index (社区索引) 🏘️
- **Leiden Community Index**: 基于Leiden算法的高质量社区检测
- **Louvain Community Index**: 基于Louvain算法的快速社区发现
- **Hierarchical Community Index**: 分层社区结构检测
- **Dynamic Community Index**: 动态社区演化分析

## 📁 项目结构

```
graph_rag_index_construction/
├── node_index/                    # 节点索引实现
│   ├── entity_node_index.py       # 实体节点索引
│   ├── document_node_index.py     # 文档节点索引
│   ├── concept_node_index.py      # 概念节点索引
│   └── hierarchical_node_index.py # 分层节点索引
├── relationship_index/            # 关系索引实现
│   ├── semantic_relationship_index.py      # 语义关系索引
│   ├── citation_relationship_index.py      # 引用关系索引
│   ├── cooccurrence_relationship_index.py  # 共现关系索引
│   └── temporal_relationship_index.py      # 时序关系索引
├── community_index/               # 社区索引实现
│   ├── leiden_community_index.py     # Leiden社区索引
│   ├── louvain_community_index.py    # Louvain社区索引
│   ├── hierarchical_community_index.py # 分层社区索引
│   └── dynamic_community_index.py    # 动态社区索引
├── utils/                         # 工具函数
│   ├── graph_utils.py            # 图操作工具
│   ├── evaluation_metrics.py     # 评估指标
│   └── visualization.py          # 可视化工具
├── examples/                      # 示例代码
│   ├── demo_all_methods.py       # 完整演示
│   └── simple_demo.py            # 简化演示
├── tests/                        # 测试代码
├── requirements.txt              # 依赖包
└── README.md                    # 项目说明
```

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install numpy pandas networkx scikit-learn sentence-transformers

# 可选依赖（用于特定功能）
pip install spacy nltk python-louvain leidenalg igraph

# 或者安装所有依赖
pip install -r requirements.txt
```

### 基本使用示例

```python
# 1. 实体节点索引
from node_index import EntityNodeIndex

entity_index = EntityNodeIndex()
documents = [
    {'id': 'doc1', 'text': 'Machine learning is a subset of AI...'},
    {'id': 'doc2', 'text': 'Deep learning uses neural networks...'},
]
entity_index.build_index(documents)
results = entity_index.find_similar_entities('neural networks')

# 2. 语义关系索引
from relationship_index import SemanticRelationshipIndex

semantic_index = SemanticRelationshipIndex()
semantic_index.build_index(documents)
relations = semantic_index.get_related_documents('doc1', relation_type='similarity')

# 3. 社区索引
from community_index import LouvainCommunityIndex

community_index = LouvainCommunityIndex()
community_index.build_index(documents)
communities = community_index.query_communities('machine learning')
```

### 运行演示

```bash
# 运行简化演示（不需要外部依赖）
python3 examples/simple_demo.py

# 运行完整演示（需要安装依赖）
python3 examples/demo_all_methods.py
```

## 🔧 核心特性

### 统一API设计
所有索引方法都遵循统一的接口：
- `build_index(documents)`: 构建索引
- `query()` / `find()` / `get()`: 查询接口
- `save_index()` / `load_index()`: 持久化
- `get_graph_statistics()`: 统计信息

### 灵活配置
每种方法都支持丰富的参数配置：
```python
# 实体索引配置
entity_index = EntityNodeIndex(
    embedding_model="all-MiniLM-L6-v2",
    similarity_threshold=0.7,
    spacy_model="en_core_web_sm"
)

# 社区索引配置
community_index = LouvainCommunityIndex(
    similarity_threshold=0.6,
    resolution=1.0,
    randomize=42
)
```

### 多种图操作工具
```python
from utils import GraphUtils

# 图合并
merged_graph = GraphUtils.merge_graphs([graph1, graph2])

# 中心性分析
centrality = GraphUtils.calculate_centrality_measures(graph)

# 社区发现
communities = GraphUtils.find_communities_networkx(graph)
```

## 📊 应用场景

| 索引类型 | 适用场景 | 优势 |
|---------|---------|------|
| Entity-based | 知识图谱构建、实体链接 | 精确的实体识别和关联 |
| Document-based | 文档检索、相似推荐 | 高效的文档相似度计算 |
| Concept-based | 主题建模、概念导航 | 语义概念的自动抽取 |
| Hierarchical | 多层次信息组织 | 支持不同粒度的检索 |
| Semantic | 语义搜索、关系推理 | 丰富的语义关系类型 |
| Community | 社区发现、群体分析 | 高效的社区结构检测 |

## 🔬 技术特点

### 先进算法
- **Sentence Transformers**: 高质量文本嵌入
- **SpaCy NLP**: 专业的自然语言处理
- **NetworkX**: 强大的图分析工具
- **Leiden/Louvain**: 最先进的社区检测算法

### 性能优化
- 并行计算支持
- 内存高效的数据结构
- 增量索引更新
- 批处理优化

### 扩展性设计
- 模块化架构
- 插件式索引方法
- 自定义相似度函数
- 多种持久化格式

## 📈 评估指标

项目提供了完整的评估框架：
- **图结构指标**: 密度、聚类系数、连通性
- **社区质量**: 模块度、轮廓系数
- **检索性能**: 准确率、召回率、F1分数
- **效率指标**: 构建时间、查询延迟

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 如何贡献
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/your-repo/graph-rag-index-construction.git
cd graph-rag-index-construction

# 安装开发依赖
pip install -r requirements.txt

# 运行测试
python -m pytest tests/
```

## 📝 方法详解

### Node Index Methods

#### 1. Entity-based Node Index
- **核心思想**: 将命名实体作为图的节点，基于实体共现和语义相似度建立边
- **实现特点**: 使用SpaCy进行实体识别，Sentence Transformers计算实体嵌入
- **适用场景**: 构建知识图谱、实体链接、问答系统

#### 2. Document-based Node Index  
- **核心思想**: 将文档作为节点，基于文档内容相似度构建图
- **实现特点**: 支持文档分块、多层次相似度计算、跨文档块连接
- **适用场景**: 文档检索、推荐系统、内容发现

#### 3. Concept-based Node Index
- **核心思想**: 提取文档中的关键概念，构建概念关系图
- **实现特点**: TF-IDF关键词提取、概念共现分析、PMI相关性计算
- **适用场景**: 主题建模、概念导航、知识发现

#### 4. Hierarchical Node Index
- **核心思想**: 通过聚类算法构建多层次的节点组织结构
- **实现特点**: 支持多种聚类算法、自适应层次深度、层内连接优化
- **适用场景**: 大规模文档组织、多粒度检索、层次化浏览

### Relationship Index Methods

#### 1. Semantic Relationship Index
- **核心思想**: 识别和建模文档间的语义关系类型
- **实现特点**: 支持10种关系类型、模式匹配、置信度评估
- **关系类型**: similarity, causation, comparison, contradiction, elaboration等

#### 2. Citation Relationship Index
- **核心思想**: 基于文档间的引用关系构建有向图
- **实现特点**: 引用解析、影响力传播、权威度计算
- **适用场景**: 学术文献分析、影响力评估、引用推荐

### Community Index Methods

#### 1. Louvain Community Index
- **核心思想**: 使用Louvain算法进行快速社区检测
- **实现特点**: 模块度优化、多分辨率支持、社区摘要生成
- **优势**: 计算效率高、适合大规模图、社区质量好

#### 2. Leiden Community Index
- **核心思想**: 使用Leiden算法进行高质量社区检测
- **实现特点**: 避免分辨率限制、保证连通性、优化局部移动
- **优势**: 社区质量更高、避免不良连接的社区

## 🔍 使用建议

### 选择合适的索引方法

1. **文档数量较少(<1000)**: 推荐Document-based + Semantic Relationship
2. **实体丰富的文档**: 推荐Entity-based + Co-occurrence Relationship  
3. **大规模文档集合**: 推荐Hierarchical + Louvain Community
4. **学术文献**: 推荐Citation Relationship + Leiden Community
5. **多主题文档**: 推荐Concept-based + Hierarchical Community

### 参数调优建议

- **similarity_threshold**: 0.6-0.8，值越高连接越少但质量更高
- **embedding_model**: 推荐使用all-MiniLM-L6-v2或更大的模型
- **community_resolution**: 1.0为默认值，<1.0得到更大社区，>1.0得到更小社区

## 📞 联系我们

如有问题或建议，请：
- 提交Issue到GitHub仓库
- 参与项目讨论

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！