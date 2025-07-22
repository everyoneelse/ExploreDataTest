#!/usr/bin/env python3
"""
GraphRAG PopQA 演示脚本
快速演示GraphRAG系统的功能
"""

import os
import sys
from data_loader import PopQADataLoader
from graph_rag import GraphRAG
from evaluator import PopQAEvaluator


def main():
    print("=" * 60)
    print("GraphRAG PopQA 评测系统演示")
    print("=" * 60)
    
    try:
        # 1. 加载样本数据
        print("\n1. 加载样本数据...")
        loader = PopQADataLoader()
        data = loader.get_sample_data(20)  # 使用20个样本进行演示
        print(f"✓ 成功加载 {len(data)} 个样本")
        
        # 显示几个样本
        print("\n样本数据预览:")
        for i, item in enumerate(data[:3]):
            print(f"  {i+1}. 问题: {item['question']}")
            print(f"     答案: {item['answer']}")
            print(f"     三元组: ({item['subject']}, {item['relation']}, {item['object']})")
        
        # 2. 构建GraphRAG
        print("\n2. 构建GraphRAG系统...")
        graph_rag = GraphRAG(model_name="sentence-transformers/all-MiniLM-L6-v2")
        graph_rag.build_knowledge_graph(data)
        print("✓ GraphRAG系统构建完成")
        
        # 3. 测试几个查询
        print("\n3. 测试问答功能...")
        test_queries = [
            "Who is the current president of the United States?",
            "What is the capital of France?",
            "Who wrote the novel 1984?"
        ]
        
        predictions = []
        for i, query in enumerate(test_queries):
            print(f"\n查询 {i+1}: {query}")
            result = graph_rag.generate_answer(query, top_k=3)
            
            print(f"  预测答案: {result['predicted_answer']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  相关文档数: {len(result['relevant_documents'])}")
            print(f"  图谱三元组数: {len(result['graph_context'])}")
            
            # 显示图谱上下文
            if result['graph_context']:
                print("  图谱信息:")
                for j, (s, r, o) in enumerate(result['graph_context'][:2]):
                    print(f"    - {s} {r} {o}")
            
            predictions.append({
                'predicted_answer': result['predicted_answer'],
                'confidence': result['confidence']
            })
        
        # 4. 评估性能
        print("\n4. 评估系统性能...")
        evaluator = PopQAEvaluator()
        
        # 构建对应的真实答案
        ground_truths = []
        for i, query in enumerate(test_queries):
            # 找到对应的真实答案
            for item in data:
                if query.lower() in item['question'].lower():
                    ground_truths.append(item)
                    break
            else:
                # 如果找不到，使用默认答案
                ground_truths.append({
                    'id': i,
                    'question': query,
                    'answer': 'Unknown',
                    'subject': 'Unknown'
                })
        
        # 运行评估
        if len(predictions) == len(ground_truths):
            results = evaluator.evaluate_batch(predictions, ground_truths)
            
            print("\n评估结果:")
            overall = results['overall_metrics']
            print(f"  精确匹配准确率: {overall['exact_match_accuracy']:.3f}")
            print(f"  包含匹配准确率: {overall['contains_match_accuracy']:.3f}")
            print(f"  平均ROUGE-1 F1: {overall['avg_rouge1_f']:.3f}")
            print(f"  平均BLEU分数: {overall['avg_bleu_score']:.3f}")
            
            # 生成简化报告
            report_dir = "demo_results"
            os.makedirs(report_dir, exist_ok=True)
            evaluator.generate_report(results, report_dir)
            print(f"\n✓ 详细报告已保存到: {report_dir}/")
        
        # 5. 知识图谱统计
        print("\n5. 知识图谱统计信息:")
        kg = graph_rag.knowledge_graph
        print(f"  实体数量: {len(kg.entity_to_id)}")
        print(f"  关系数量: {kg.graph.number_of_edges()}")
        print(f"  文档数量: {len(graph_rag.document_store)}")
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("要运行完整评测，请使用: python main_evaluation.py --use-sample")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())