import json
import pandas as pd
from datasets import load_dataset
from typing import List, Dict, Any, Optional
import requests
import os


class PopQADataLoader:
    """PopQA数据集加载器"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None
        
    def load_from_huggingface(self, split: str = "test") -> List[Dict[str, Any]]:
        """从Hugging Face加载PopQA数据集"""
        try:
            dataset = load_dataset("akariasai/PopQA", split=split)
            self.data = [
                {
                    "id": i,
                    "question": item["question"],
                    "answer": item["answer"],
                    "subject": item.get("subject", ""),
                    "relation": item.get("relation", ""),
                    "object": item.get("object", "")
                }
                for i, item in enumerate(dataset)
            ]
            print(f"成功加载 {len(self.data)} 条PopQA数据")
            return self.data
        except Exception as e:
            print(f"从Hugging Face加载数据失败: {e}")
            return []
    
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """从本地文件加载PopQA数据"""
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.endswith('.jsonl'):
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError("不支持的文件格式，请使用.json或.jsonl文件")
            
            self.data = [
                {
                    "id": i,
                    "question": item["question"],
                    "answer": item["answer"],
                    "subject": item.get("subject", ""),
                    "relation": item.get("relation", ""),
                    "object": item.get("object", "")
                }
                for i, item in enumerate(data)
            ]
            print(f"成功从文件加载 {len(self.data)} 条PopQA数据")
            return self.data
        except Exception as e:
            print(f"从文件加载数据失败: {e}")
            return []
    
    def get_sample_data(self, n: int = 100) -> List[Dict[str, Any]]:
        """获取样本数据用于测试"""
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
            }
        ]
        
        # 复制样本数据到指定数量
        extended_data = []
        for i in range(n):
            sample = sample_data[i % len(sample_data)].copy()
            sample["id"] = i
            extended_data.append(sample)
        
        self.data = extended_data
        print(f"生成 {len(self.data)} 条样本数据")
        return self.data
    
    def get_data(self) -> List[Dict[str, Any]]:
        """获取当前加载的数据"""
        return self.data if self.data else []
    
    def save_data(self, output_path: str):
        """保存数据到文件"""
        if not self.data:
            print("没有数据可保存")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {output_path}")


if __name__ == "__main__":
    # 测试数据加载器
    loader = PopQADataLoader()
    
    # 首先尝试从Hugging Face加载
    data = loader.load_from_huggingface()
    
    # 如果失败，使用样本数据
    if not data:
        print("使用样本数据进行测试")
        data = loader.get_sample_data(50)
    
    print(f"数据示例:")
    for i, item in enumerate(data[:3]):
        print(f"{i+1}. 问题: {item['question']}")
        print(f"   答案: {item['answer']}")
        print(f"   主体: {item['subject']}")
        print(f"   关系: {item['relation']}")
        print(f"   客体: {item['object']}")
        print()