import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


class PopQAEvaluator:
    """PopQA评测器"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def normalize_answer(self, answer: str) -> str:
        """标准化答案文本"""
        if not answer:
            return ""
        
        # 转换为小写
        answer = answer.lower().strip()
        
        # 移除标点符号
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # 移除多余的空格
        answer = ' '.join(answer.split())
        
        return answer
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """精确匹配"""
        return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)
    
    def contains_match(self, predicted: str, ground_truth: str) -> bool:
        """包含匹配"""
        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)
        
        return gt_norm in pred_norm or pred_norm in gt_norm
    
    def fuzzy_match(self, predicted: str, ground_truth: str, threshold: float = 0.8) -> bool:
        """模糊匹配（基于编辑距离）"""
        from difflib import SequenceMatcher
        
        pred_norm = self.normalize_answer(predicted)
        gt_norm = self.normalize_answer(ground_truth)
        
        similarity = SequenceMatcher(None, pred_norm, gt_norm).ratio()
        return similarity >= threshold
    
    def calculate_rouge(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        scores = self.rouge_scorer.score(ground_truth, predicted)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def calculate_bleu(self, predicted: str, ground_truth: str) -> float:
        """计算BLEU分数"""
        pred_tokens = predicted.lower().split()
        gt_tokens = [ground_truth.lower().split()]  # BLEU expects list of reference lists
        
        if not pred_tokens:
            return 0.0
        
        return sentence_bleu(gt_tokens, pred_tokens, smoothing_function=self.smoothing.method1)
    
    def calculate_bert_score(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """计算BERTScore"""
        try:
            P, R, F1 = bert_score(predictions, ground_truths, lang="en", verbose=False)
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"计算BERTScore时出错: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    def evaluate_single_prediction(self, predicted: str, ground_truth: str) -> Dict[str, Any]:
        """评估单个预测"""
        if not predicted:
            predicted = ""
        
        metrics = {
            'exact_match': self.exact_match(predicted, ground_truth),
            'contains_match': self.contains_match(predicted, ground_truth),
            'fuzzy_match': self.fuzzy_match(predicted, ground_truth),
            'bleu_score': self.calculate_bleu(predicted, ground_truth)
        }
        
        # 添加ROUGE分数
        rouge_scores = self.calculate_rouge(predicted, ground_truth)
        metrics.update(rouge_scores)
        
        return metrics
    
    def evaluate_batch(self, predictions: List[Dict[str, Any]], 
                      ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估"""
        print("开始批量评估...")
        
        # 确保预测和真实答案数量匹配
        assert len(predictions) == len(ground_truths), "预测和真实答案数量不匹配"
        
        # 单个样本的评估结果
        individual_results = []
        pred_texts = []
        gt_texts = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_answer = pred.get('predicted_answer', '')
            gt_answer = gt.get('answer', '')
            
            pred_texts.append(pred_answer)
            gt_texts.append(gt_answer)
            
            # 单个样本评估
            single_metrics = self.evaluate_single_prediction(pred_answer, gt_answer)
            single_metrics.update({
                'question_id': gt.get('id', ''),
                'question': gt.get('question', ''),
                'predicted_answer': pred_answer,
                'ground_truth_answer': gt_answer,
                'confidence': pred.get('confidence', 0.0)
            })
            
            individual_results.append(single_metrics)
        
        # 计算整体指标
        overall_metrics = self._calculate_overall_metrics(individual_results)
        
        # 计算BERTScore
        bert_scores = self.calculate_bert_score(pred_texts, gt_texts)
        overall_metrics.update(bert_scores)
        
        # 按类别分析（如果有subject信息）
        category_analysis = self._analyze_by_category(individual_results, ground_truths)
        
        return {
            'overall_metrics': overall_metrics,
            'individual_results': individual_results,
            'category_analysis': category_analysis,
            'total_samples': len(predictions)
        }
    
    def _calculate_overall_metrics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算整体指标"""
        metrics = {}
        
        # 计算平均值
        metric_names = ['exact_match', 'contains_match', 'fuzzy_match', 'bleu_score',
                       'rouge1_f', 'rouge2_f', 'rougeL_f']
        
        for metric in metric_names:
            values = [result[metric] for result in individual_results if metric in result]
            metrics[f'avg_{metric}'] = np.mean(values) if values else 0.0
        
        # 计算准确率（基于不同匹配方式）
        metrics['exact_match_accuracy'] = np.mean([r['exact_match'] for r in individual_results])
        metrics['contains_match_accuracy'] = np.mean([r['contains_match'] for r in individual_results])
        metrics['fuzzy_match_accuracy'] = np.mean([r['fuzzy_match'] for r in individual_results])
        
        return metrics
    
    def _analyze_by_category(self, individual_results: List[Dict[str, Any]], 
                           ground_truths: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """按类别分析性能"""
        category_results = defaultdict(list)
        
        for result, gt in zip(individual_results, ground_truths):
            subject = gt.get('subject', 'unknown')
            if subject:
                category_results[subject].append(result)
        
        category_metrics = {}
        for category, results in category_results.items():
            if len(results) >= 2:  # 只分析有足够样本的类别
                category_metrics[category] = {
                    'count': len(results),
                    'exact_match_accuracy': np.mean([r['exact_match'] for r in results]),
                    'contains_match_accuracy': np.mean([r['contains_match'] for r in results]),
                    'avg_rouge1_f': np.mean([r['rouge1_f'] for r in results]),
                    'avg_bleu_score': np.mean([r['bleu_score'] for r in results])
                }
        
        return category_metrics
    
    def generate_report(self, evaluation_results: Dict[str, Any], 
                       output_dir: str = "evaluation_results") -> str:
        """生成评估报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文本报告
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PopQA GraphRAG 评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 整体性能
            overall = evaluation_results['overall_metrics']
            f.write("整体性能指标:\n")
            f.write(f"- 精确匹配准确率: {overall['exact_match_accuracy']:.4f}\n")
            f.write(f"- 包含匹配准确率: {overall['contains_match_accuracy']:.4f}\n")
            f.write(f"- 模糊匹配准确率: {overall['fuzzy_match_accuracy']:.4f}\n")
            f.write(f"- 平均BLEU分数: {overall['avg_bleu_score']:.4f}\n")
            f.write(f"- 平均ROUGE-1 F1: {overall['avg_rouge1_f']:.4f}\n")
            f.write(f"- 平均ROUGE-2 F1: {overall['avg_rouge2_f']:.4f}\n")
            f.write(f"- 平均ROUGE-L F1: {overall['avg_rougeL_f']:.4f}\n")
            
            if 'bert_f1' in overall:
                f.write(f"- BERTScore F1: {overall['bert_f1']:.4f}\n")
            
            f.write(f"\n总样本数: {evaluation_results['total_samples']}\n\n")
            
            # 类别分析
            if evaluation_results['category_analysis']:
                f.write("按类别分析:\n")
                for category, metrics in evaluation_results['category_analysis'].items():
                    f.write(f"\n{category} ({metrics['count']} 个样本):\n")
                    f.write(f"  - 精确匹配: {metrics['exact_match_accuracy']:.4f}\n")
                    f.write(f"  - 包含匹配: {metrics['contains_match_accuracy']:.4f}\n")
                    f.write(f"  - ROUGE-1 F1: {metrics['avg_rouge1_f']:.4f}\n")
                    f.write(f"  - BLEU: {metrics['avg_bleu_score']:.4f}\n")
        
        # 保存详细结果
        results_path = os.path.join(output_dir, "detailed_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 生成可视化图表
        self._generate_visualizations(evaluation_results, output_dir)
        
        print(f"评估报告已保存到: {report_path}")
        return report_path
    
    def _generate_visualizations(self, evaluation_results: Dict[str, Any], output_dir: str):
        """生成可视化图表"""
        try:
            plt.style.use('default')
            
            # 1. 整体性能条形图
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            overall = evaluation_results['overall_metrics']
            
            metrics = ['exact_match_accuracy', 'contains_match_accuracy', 'fuzzy_match_accuracy',
                      'avg_rouge1_f', 'avg_rouge2_f', 'avg_rougeL_f', 'avg_bleu_score']
            values = [overall.get(metric, 0) for metric in metrics]
            labels = ['精确匹配', '包含匹配', '模糊匹配', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
            
            bars = ax.bar(labels, values, color='skyblue', alpha=0.7)
            ax.set_ylabel('分数')
            ax.set_title('GraphRAG在PopQA上的整体性能')
            ax.set_ylim(0, 1)
            
            # 在条形图上添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 类别性能比较（如果有足够的类别）
            category_analysis = evaluation_results['category_analysis']
            if len(category_analysis) >= 2:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                categories = list(category_analysis.keys())[:10]  # 只显示前10个类别
                exact_match_scores = [category_analysis[cat]['exact_match_accuracy'] for cat in categories]
                
                bars = ax.bar(categories, exact_match_scores, color='lightcoral', alpha=0.7)
                ax.set_ylabel('精确匹配准确率')
                ax.set_title('不同类别的精确匹配性能')
                ax.set_ylim(0, 1)
                
                for bar, value in zip(bars, exact_match_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'category_performance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"可视化图表已保存到: {output_dir}")
            
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")


if __name__ == "__main__":
    # 测试评估器
    evaluator = PopQAEvaluator()
    
    # 创建测试数据
    predictions = [
        {'predicted_answer': 'Joe Biden', 'confidence': 0.95},
        {'predicted_answer': 'Paris', 'confidence': 0.90},
        {'predicted_answer': 'George Orwell', 'confidence': 0.85}
    ]
    
    ground_truths = [
        {'id': 0, 'question': 'Who is the president?', 'answer': 'Joe Biden', 'subject': 'Joe Biden'},
        {'id': 1, 'question': 'Capital of France?', 'answer': 'Paris', 'subject': 'France'},
        {'id': 2, 'question': 'Who wrote 1984?', 'answer': 'George Orwell', 'subject': '1984'}
    ]
    
    # 评估
    results = evaluator.evaluate_batch(predictions, ground_truths)
    
    # 生成报告
    report_path = evaluator.generate_report(results)
    print(f"测试完成，报告路径: {report_path}")