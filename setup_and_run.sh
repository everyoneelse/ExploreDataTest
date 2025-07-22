#!/bin/bash

# GraphRAG PopQA 评测系统 - 安装和运行脚本

echo "=================================================="
echo "GraphRAG PopQA 评测系统 - 自动安装和运行"
echo "=================================================="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1)
if [[ $? -ne 0 ]]; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.7+"
    exit 1
fi
echo "✓ $python_version"

# 检查pip
echo "检查pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ 错误: 未找到pip3，请先安装pip"
    exit 1
fi
echo "✓ pip3 已安装"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境？(y/N): " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "创建虚拟环境..."
    python3 -m venv graphrag_env
    source graphrag_env/bin/activate
    echo "✓ 虚拟环境已激活"
fi

# 安装依赖
echo "安装Python依赖包..."
pip3 install --upgrade pip

# 分步安装以避免依赖冲突
echo "安装基础包..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "安装其他依赖..."
pip3 install transformers sentence-transformers networkx numpy pandas scikit-learn
pip3 install datasets tqdm matplotlib seaborn rouge-score bert-score nltk spacy
pip3 install requests faiss-cpu

if [[ $? -eq 0 ]]; then
    echo "✓ 所有依赖包安装成功"
else
    echo "❌ 依赖包安装失败，尝试使用requirements.txt..."
    pip3 install -r requirements.txt
    if [[ $? -ne 0 ]]; then
        echo "❌ 安装失败，请手动安装依赖"
        exit 1
    fi
fi

# 下载必要的NLTK数据
echo "下载NLTK数据..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null

# 检查安装
echo "验证安装..."
python3 -c "
try:
    import torch, transformers, sentence_transformers, networkx
    import numpy, pandas, sklearn, datasets, nltk
    import rouge_score, bert_score, matplotlib, faiss
    print('✓ 所有包导入成功')
except ImportError as e:
    print(f'❌ 导入错误: {e}')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    echo "❌ 包验证失败"
    exit 1
fi

echo ""
echo "=================================================="
echo "安装完成！开始运行演示..."
echo "=================================================="

# 运行演示
echo "运行GraphRAG演示..."
python3 demo.py

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=================================================="
    echo "演示成功完成！"
    echo "=================================================="
    echo ""
    echo "接下来你可以："
    echo "1. 运行基础评测: python3 main_evaluation.py --use-sample --sample-size 50"
    echo "2. 运行完整评测: python3 main_evaluation.py --max-samples 1000"
    echo "3. 查看帮助信息: python3 main_evaluation.py --help"
    echo ""
    echo "详细使用说明请参考 README.md"
else
    echo "❌ 演示运行失败"
    exit 1
fi