# GraphRAG PopQA 评测系统 - 快速开始指南

## 🚀 立即开始

### 方法一：简化演示（推荐新手）
无需安装复杂依赖，直接运行：

```bash
# 1. 创建虚拟环境
python3 -m venv graphrag_env
source graphrag_env/bin/activate

# 2. 安装基础依赖
pip install pandas numpy tqdm networkx

# 3. 运行简化演示
python3 simple_demo.py
```

### 方法二：自动安装（一键运行）
```bash
# 运行自动安装脚本
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### 方法三：手动完整安装
```bash
# 1. 创建虚拟环境
python3 -m venv graphrag_env
source graphrag_env/bin/activate

# 2. 安装所有依赖
pip install -r requirements.txt

# 3. 运行完整演示
python3 demo.py

# 4. 运行评测
python3 main_evaluation.py --use-sample --sample-size 50
```

## 📊 预期结果

运行成功后，您将看到类似以下的输出：

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

4. 评估系统性能...
  精确匹配准确率: 0.667
  包含匹配准确率: 0.667
  总样本数: 3
```

## 🔧 常见问题

### Q: 提示缺少某个模块怎么办？
A: 确保激活了虚拟环境，然后安装缺少的包：
```bash
source graphrag_env/bin/activate
pip install 缺少的包名
```

### Q: 如何使用自己的数据？
A: 准备JSON格式的数据文件，然后：
```bash
python3 main_evaluation.py --data-file your_data.json
```

### Q: 如何提高评测性能？
A: 使用更好的模型和更多样本：
```bash
python3 main_evaluation.py \
    --model-name sentence-transformers/all-mpnet-base-v2 \
    --max-samples 1000 \
    --top-k 10
```

## 📁 输出文件

运行后会生成以下文件：
- `simple_demo_results.json` - 演示结果
- `evaluation_results/` - 完整评测结果目录
- `knowledge_graph.pkl` - 保存的知识图谱（如果指定）

## 🎯 下一步

1. 查看 `README.md` 了解详细使用方法
2. 查看 `PROJECT_SUMMARY.md` 了解系统架构
3. 修改代码以适应您的具体需求
4. 添加新的评测指标或数据源

## 💡 技术支持

如遇到问题，请检查：
1. Python版本 >= 3.7
2. 虚拟环境是否正确激活
3. 依赖包是否正确安装
4. 系统内存是否充足（建议 >= 4GB）

祝您使用愉快！🎉