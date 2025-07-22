#!/usr/bin/env python3
"""
GraphRAG PopQA 评测主程序
使用GraphRAG对PopQA数据集进行问答评测
"""

import argparse
import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging

from data_loader import PopQADataLoader
from graph_rag import GraphRAG
from evaluator import PopQAEvaluator


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_data(args, logger) -> List[Dict[str, Any]]:
    """加载PopQA数据"""
    logger.info("开始加载PopQA数据...")
    
    loader = PopQADataLoader()
    
    if args.data_file:
        # 从本地文件加载
        logger.info(f"从文件加载数据: {args.data_file}")
        data = loader.load_from_file(args.data_file)
    elif args.use_sample:
        # 使用样本数据
        logger.info(f"使用样本数据: {args.sample_size} 个样本")
        data = loader.get_sample_data(args.sample_size)
    else:
        # 从Hugging Face加载
        logger.info("从Hugging Face加载数据...")
        data = loader.load_from_huggingface()
        
        if not data:
            logger.warning("从Hugging Face加载失败，使用样本数据")
            data = loader.get_sample_data(args.sample_size)
    
    if args.max_samples and len(data) > args.max_samples:
        logger.info(f"限制样本数量为: {args.max_samples}")
        data = data[:args.max_samples]
    
    logger.info(f"成功加载 {len(data)} 个样本")
    return data


def build_or_load_graph_rag(args, data: List[Dict[str, Any]], logger) -> GraphRAG:
    """构建或加载GraphRAG"""
    logger.info("初始化GraphRAG...")
    
    # 检查是否有预训练的知识图谱
    kg_path = args.knowledge_graph_path
    if kg_path and os.path.exists(kg_path):
        logger.info(f"从文件加载知识图谱: {kg_path}")
        graph_rag = GraphRAG(model_name=args.model_name, kg_path=kg_path)
    else:
        logger.info("构建新的知识图谱...")
        graph_rag = GraphRAG(model_name=args.model_name)
        graph_rag.build_knowledge_graph(data)
        
        # 保存知识图谱
        if args.save_knowledge_graph:
            save_path = args.save_knowledge_graph
            logger.info(f"保存知识图谱到: {save_path}")
            graph_rag.save_knowledge_graph(save_path)
    
    return graph_rag


def run_evaluation(graph_rag: GraphRAG, test_data: List[Dict[str, Any]], 
                  args, logger) -> List[Dict[str, Any]]:
    """运行评测"""
    logger.info("开始运行评测...")
    
    predictions = []
    
    for i, item in enumerate(tqdm(test_data, desc="评测进行中")):
        try:
            question = item['question']
            
            # 使用GraphRAG生成答案
            result = graph_rag.generate_answer(question, top_k=args.top_k)
            
            prediction = {
                'question_id': item['id'],
                'question': question,
                'predicted_answer': result['predicted_answer'],
                'confidence': result['confidence'],
                'context': result['context'],
                'num_relevant_docs': len(result['relevant_documents']),
                'num_graph_triples': len(result['graph_context'])
            }
            
            predictions.append(prediction)
            
            # 保存中间结果（每100个样本保存一次）
            if args.save_intermediate and (i + 1) % 100 == 0:
                intermediate_path = f"intermediate_predictions_{i+1}.json"
                with open(intermediate_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存中间结果: {intermediate_path}")
                
        except Exception as e:
            logger.error(f"处理问题 {item['id']} 时出错: {e}")
            # 添加空预测以保持数据对齐
            predictions.append({
                'question_id': item['id'],
                'question': item['question'],
                'predicted_answer': '',
                'confidence': 0.0,
                'context': '',
                'num_relevant_docs': 0,
                'num_graph_triples': 0
            })
    
    logger.info(f"评测完成，共处理 {len(predictions)} 个问题")
    return predictions


def evaluate_results(predictions: List[Dict[str, Any]], 
                    ground_truths: List[Dict[str, Any]], 
                    args, logger) -> Dict[str, Any]:
    """评估结果"""
    logger.info("开始评估结果...")
    
    evaluator = PopQAEvaluator()
    
    # 运行评估
    evaluation_results = evaluator.evaluate_batch(predictions, ground_truths)
    
    # 生成报告
    report_path = evaluator.generate_report(evaluation_results, args.output_dir)
    
    logger.info("评估完成")
    return evaluation_results


def save_results(predictions: List[Dict[str, Any]], 
                evaluation_results: Dict[str, Any],
                args, logger):
    """保存结果"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存预测结果
    predictions_path = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"预测结果已保存到: {predictions_path}")
    
    # 保存评估结果
    evaluation_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存到: {evaluation_path}")
    
    # 保存配置信息
    config = {
        'model_name': args.model_name,
        'max_samples': args.max_samples,
        'top_k': args.top_k,
        'use_sample': args.use_sample,
        'sample_size': args.sample_size,
        'data_file': args.data_file,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"配置信息已保存到: {config_path}")


def print_summary(evaluation_results: Dict[str, Any], logger):
    """打印评估摘要"""
    overall = evaluation_results['overall_metrics']
    
    print("\n" + "="*60)
    print("GraphRAG PopQA 评测结果摘要")
    print("="*60)
    print(f"总样本数: {evaluation_results['total_samples']}")
    print(f"精确匹配准确率: {overall['exact_match_accuracy']:.4f}")
    print(f"包含匹配准确率: {overall['contains_match_accuracy']:.4f}")
    print(f"模糊匹配准确率: {overall['fuzzy_match_accuracy']:.4f}")
    print(f"平均ROUGE-1 F1: {overall['avg_rouge1_f']:.4f}")
    print(f"平均ROUGE-L F1: {overall['avg_rougeL_f']:.4f}")
    print(f"平均BLEU分数: {overall['avg_bleu_score']:.4f}")
    
    if 'bert_f1' in overall:
        print(f"BERTScore F1: {overall['bert_f1']:.4f}")
    
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GraphRAG PopQA 评测")
    
    # 数据相关参数
    parser.add_argument('--data-file', type=str, help='PopQA数据文件路径')
    parser.add_argument('--use-sample', action='store_true', help='使用样本数据')
    parser.add_argument('--sample-size', type=int, default=100, help='样本数据大小')
    parser.add_argument('--max-samples', type=int, help='最大样本数量限制')
    
    # 模型相关参数
    parser.add_argument('--model-name', type=str, 
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='句子编码器模型名称')
    parser.add_argument('--top-k', type=int, default=5, help='检索文档数量')
    
    # 知识图谱相关参数
    parser.add_argument('--knowledge-graph-path', type=str, 
                       help='预训练知识图谱文件路径')
    parser.add_argument('--save-knowledge-graph', type=str,
                       help='保存知识图谱的路径')
    
    # 输出相关参数
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='结果输出目录')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='保存中间结果')
    
    # 其他参数
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    try:
        # 1. 加载数据
        data = load_data(args, logger)
        
        if not data:
            logger.error("无法加载数据，退出程序")
            return
        
        # 2. 构建或加载GraphRAG
        graph_rag = build_or_load_graph_rag(args, data, logger)
        
        # 3. 运行评测
        predictions = run_evaluation(graph_rag, data, args, logger)
        
        # 4. 评估结果
        evaluation_results = evaluate_results(predictions, data, args, logger)
        
        # 5. 保存结果
        save_results(predictions, evaluation_results, args, logger)
        
        # 6. 打印摘要
        print_summary(evaluation_results, logger)
        
        logger.info("评测完成！")
        
    except Exception as e:
        logger.error(f"评测过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()