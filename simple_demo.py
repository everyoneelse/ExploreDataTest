#!/usr/bin/env python3
"""
GraphRAG PopQA 简化演示脚本
展示GraphRAG系统的基本功能，不依赖复杂的第三方库
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os


class SimpleDataLoader:
    """简化的数据加载器"""
    
    def get_sample_data(self, n: int = 20) -> List[Dict[str, Any]]:
        """获取样本数据"""
        sample_data = [
            {
                "id": 0,
                "question": "Who is the current president of the United States?",
                "answer": "Joe Biden",
                "subject": "Joe Biden",
                "relation": "president of",
                "object": "United States"
            },
            {
                "id": 1,
                "question": "What is the capital of France?",
                "answer": "Paris",
                "subject": "France",
                "relation": "capital",
                "object": "Paris"
            },
            {
                "id": 2,
                "question": "Who wrote the novel '1984'?",
                "answer": "George Orwell",
                "subject": "1984",
                "relation": "author",
                "object": "George Orwell"
            },
            {
                "id": 3,
                "question": "What is the largest planet in our solar system?",
                "answer": "Jupiter",
                "subject": "Jupiter",
                "relation": "largest planet in",
                "object": "solar system"
            },
            {
                "id": 4,
                "question": "When was the iPhone first released?",
                "answer": "2007",
                "subject": "iPhone",
                "relation": "first released in",
                "object": "2007"
            },
            {
                "id": 5,
                "question": "Who painted the Mona Lisa?",
                "answer": "Leonardo da Vinci",
                "subject": "Mona Lisa",
                "relation": "painted by",
                "object": "Leonardo da Vinci"
            },
            {
                "id": 6,
                "question": "What is the chemical symbol for gold?",
                "answer": "Au",
                "subject": "gold",
                "relation": "chemical symbol",
                "object": "Au"
            },
            {
                "id": 7,
                "question": "Which ocean is the largest?",
                "answer": "Pacific Ocean",
                "subject": "Pacific Ocean",
                "relation": "largest",
                "object": "ocean"
            },
            {
                "id": 8,
                "question": "Who discovered penicillin?",
                "answer": "Alexander Fleming",
                "subject": "penicillin",
                "relation": "discovered by",
                "object": "Alexander Fleming"
            },
            {
                "id": 9,
                "question": "What is the speed of light?",
                "answer": "299792458 m/s",
                "subject": "light",
                "relation": "speed",
                "object": "299792458 m/s"
            }
        ]
        
        # 复制样本数据到指定数量
        extended_data = []
        for i in range(n):
            sample = sample_data[i % len(sample_data)].copy()
            sample["id"] = i
            extended_data.append(sample)
        
        return extended_data


class SimpleKnowledgeGraph:
    """简化的知识图谱"""
    
    def __init__(self):
        self.triples = []
        self.entities = set()
        self.relations = set()
    
    def add_triple(self, subject: str, relation: str, obj: str):
        """添加三元组"""
        if subject and relation and obj:
            self.triples.append((subject, relation, obj))
            self.entities.add(subject)
            self.entities.add(obj)
            self.relations.add(relation)
    
    def get_related_triples(self, query: str) -> List[tuple]:
        """获取与查询相关的三元组"""
        query_lower = query.lower()
        related = []
        
        for subj, rel, obj in self.triples:
            if (query_lower in subj.lower() or 
                query_lower in obj.lower() or
                any(word in subj.lower() or word in obj.lower() 
                    for word in query_lower.split())):
                related.append((subj, rel, obj))
        
        return related[:5]  # 返回前5个相关三元组


class SimpleGraphRAG:
    """简化的GraphRAG系统"""
    
    def __init__(self):
        self.knowledge_graph = SimpleKnowledgeGraph()
        self.documents = []
    
    def build_knowledge_graph(self, data: List[Dict[str, Any]]):
        """构建知识图谱"""
        print("构建知识图谱...")
        
        for item in data:
            # 添加三元组到知识图谱
            subject = item.get('subject', '').strip()
            relation = item.get('relation', '').strip()
            obj = item.get('object', '').strip()
            
            if subject and relation and obj:
                self.knowledge_graph.add_triple(subject, relation, obj)
            
            # 存储文档
            doc = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'text': f"Question: {item['question']} Answer: {item['answer']}"
            }
            self.documents.append(doc)
        
        print(f"知识图谱构建完成: {len(self.knowledge_graph.entities)} 个实体, "
              f"{len(self.knowledge_graph.triples)} 个三元组")
    
    def simple_retrieval(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """简单的文档检索"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            # 简单的关键词匹配评分
            score = 0
            doc_text = doc['text'].lower()
            
            # 计算查询词在文档中的出现次数
            for word in query_lower.split():
                if word in doc_text:
                    score += doc_text.count(word)
            
            if score > 0:
                scored_docs.append((score, doc))
        
        # 按分数排序并返回前k个
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def generate_answer(self, query: str) -> Dict[str, Any]:
        """生成答案"""
        # 1. 检索相关文档
        relevant_docs = self.simple_retrieval(query, top_k=3)
        
        # 2. 获取相关的知识图谱信息
        graph_context = self.knowledge_graph.get_related_triples(query)
        
        # 3. 简单的答案生成（使用最相关文档的答案）
        predicted_answer = ""
        confidence = 0.0
        
        if relevant_docs:
            predicted_answer = relevant_docs[0]['answer']
            confidence = 0.8  # 简单的固定置信度
        
        return {
            'query': query,
            'predicted_answer': predicted_answer,
            'confidence': confidence,
            'relevant_documents': relevant_docs,
            'graph_context': graph_context
        }


class SimpleEvaluator:
    """简化的评估器"""
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """精确匹配"""
        return predicted.lower().strip() == ground_truth.lower().strip()
    
    def contains_match(self, predicted: str, ground_truth: str) -> bool:
        """包含匹配"""
        pred_lower = predicted.lower().strip()
        gt_lower = ground_truth.lower().strip()
        return gt_lower in pred_lower or pred_lower in gt_lower
    
    def evaluate_predictions(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """评估预测结果"""
        exact_matches = 0
        contains_matches = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            pred_answer = pred.get('predicted_answer', '')
            gt_answer = gt.get('answer', '')
            
            if self.exact_match(pred_answer, gt_answer):
                exact_matches += 1
            
            if self.contains_match(pred_answer, gt_answer):
                contains_matches += 1
        
        return {
            'exact_match_accuracy': exact_matches / total if total > 0 else 0,
            'contains_match_accuracy': contains_matches / total if total > 0 else 0,
            'total_samples': total
        }


def main():
    print("=" * 60)
    print("GraphRAG PopQA 简化演示系统")
    print("=" * 60)
    
    try:
        # 1. 加载样本数据
        print("\n1. 加载样本数据...")
        loader = SimpleDataLoader()
        data = loader.get_sample_data(10)
        print(f"✓ 成功加载 {len(data)} 个样本")
        
        # 显示几个样本
        print("\n样本数据预览:")
        for i, item in enumerate(data[:3]):
            print(f"  {i+1}. 问题: {item['question']}")
            print(f"     答案: {item['answer']}")
            print(f"     三元组: ({item['subject']}, {item['relation']}, {item['object']})")
        
        # 2. 构建GraphRAG
        print("\n2. 构建GraphRAG系统...")
        graph_rag = SimpleGraphRAG()
        graph_rag.build_knowledge_graph(data)
        print("✓ GraphRAG系统构建完成")
        
        # 3. 测试几个查询
        print("\n3. 测试问答功能...")
        test_queries = [
            "Who is the current president of the United States?",
            "What is the capital of France?",
            "Who wrote 1984?"
        ]
        
        predictions = []
        for i, query in enumerate(test_queries):
            print(f"\n查询 {i+1}: {query}")
            result = graph_rag.generate_answer(query)
            
            print(f"  预测答案: {result['predicted_answer']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  相关文档数: {len(result['relevant_documents'])}")
            print(f"  图谱三元组数: {len(result['graph_context'])}")
            
            # 显示图谱上下文
            if result['graph_context']:
                print("  图谱信息:")
                for j, (s, r, o) in enumerate(result['graph_context'][:2]):
                    print(f"    - {s} {r} {o}")
            
            predictions.append(result)
        
        # 4. 简单评估
        print("\n4. 评估系统性能...")
        evaluator = SimpleEvaluator()
        
        # 构建对应的真实答案
        ground_truths = []
        for query in test_queries:
            for item in data:
                if query.lower() in item['question'].lower():
                    ground_truths.append(item)
                    break
            else:
                ground_truths.append({'answer': 'Unknown'})
        
        if len(predictions) == len(ground_truths):
            results = evaluator.evaluate_predictions(predictions, ground_truths)
            print(f"  精确匹配准确率: {results['exact_match_accuracy']:.3f}")
            print(f"  包含匹配准确率: {results['contains_match_accuracy']:.3f}")
            print(f"  总样本数: {results['total_samples']}")
        
        # 5. 知识图谱统计
        print("\n5. 知识图谱统计信息:")
        kg = graph_rag.knowledge_graph
        print(f"  实体数量: {len(kg.entities)}")
        print(f"  关系数量: {len(kg.relations)}")
        print(f"  三元组数量: {len(kg.triples)}")
        print(f"  文档数量: {len(graph_rag.documents)}")
        
        # 6. 保存结果
        print("\n6. 保存结果...")
        results_data = {
            'predictions': predictions,
            'ground_truths': ground_truths,
            'evaluation': results,
            'knowledge_graph_stats': {
                'entities': len(kg.entities),
                'relations': len(kg.relations),
                'triples': len(kg.triples),
                'documents': len(graph_rag.documents)
            }
        }
        
        with open('simple_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print("✓ 结果已保存到 simple_demo_results.json")
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("这是一个简化的GraphRAG实现，展示了基本的工作流程。")
        print("完整版本需要安装更多依赖包来获得更好的性能。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())