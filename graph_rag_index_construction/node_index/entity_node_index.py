"""
Entity-based Node Index Implementation
基于实体的节点索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from sentence_transformers import SentenceTransformer
import spacy
from collections import defaultdict
import json


class EntityNodeIndex:
    """
    基于实体的节点索引
    
    该类实现了基于命名实体识别的节点索引构建方法，
    将文档中的实体作为图的节点，并建立实体间的关系。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 similarity_threshold: float = 0.7):
        """
        初始化实体节点索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            spacy_model: SpaCy模型名称
            similarity_threshold: 实体相似度阈值
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"SpaCy model {spacy_model} not found. Please install it with: python -m spacy download {spacy_model}")
            raise
            
        self.similarity_threshold = similarity_threshold
        self.graph = nx.Graph()
        self.entity_embeddings = {}
        self.entity_contexts = defaultdict(list)
        self.entity_types = {}
        
    def extract_entities(self, texts: List[str]) -> Dict[str, Dict]:
        """
        从文本中提取命名实体
        
        Args:
            texts: 输入文本列表
            
        Returns:
            实体信息字典
        """
        entities = {}
        
        for doc_id, text in enumerate(texts):
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_name = ent.text.lower().strip()
                entity_type = ent.label_
                
                if len(entity_name) < 2:  # 过滤过短的实体
                    continue
                    
                if entity_name not in entities:
                    entities[entity_name] = {
                        'type': entity_type,
                        'contexts': [],
                        'documents': set(),
                        'frequency': 0
                    }
                
                # 获取实体上下文
                start = max(0, ent.start - 10)
                end = min(len(doc), ent.end + 10)
                context = doc[start:end].text
                
                entities[entity_name]['contexts'].append(context)
                entities[entity_name]['documents'].add(doc_id)
                entities[entity_name]['frequency'] += 1
                
        return entities
    
    def compute_entity_embeddings(self, entities: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """
        计算实体的嵌入表示
        
        Args:
            entities: 实体信息字典
            
        Returns:
            实体嵌入字典
        """
        embeddings = {}
        
        for entity_name, entity_info in entities.items():
            # 使用实体名称和上下文计算嵌入
            contexts = entity_info['contexts'][:5]  # 限制上下文数量
            text_for_embedding = f"{entity_name} " + " ".join(contexts)
            
            embedding = self.embedding_model.encode(text_for_embedding)
            embeddings[entity_name] = embedding
            
        return embeddings
    
    def build_entity_graph(self, 
                          entities: Dict[str, Dict], 
                          embeddings: Dict[str, np.ndarray]) -> nx.Graph:
        """
        构建实体图
        
        Args:
            entities: 实体信息字典
            embeddings: 实体嵌入字典
            
        Returns:
            实体图
        """
        graph = nx.Graph()
        
        # 添加节点
        for entity_name, entity_info in entities.items():
            graph.add_node(entity_name, 
                          type=entity_info['type'],
                          frequency=entity_info['frequency'],
                          documents=list(entity_info['documents']),
                          embedding=embeddings[entity_name])
        
        # 添加边（基于相似度和共现）
        entity_names = list(entities.keys())
        
        for i, entity1 in enumerate(entity_names):
            for j, entity2 in enumerate(entity_names[i+1:], i+1):
                # 计算嵌入相似度
                sim = np.dot(embeddings[entity1], embeddings[entity2]) / (
                    np.linalg.norm(embeddings[entity1]) * np.linalg.norm(embeddings[entity2])
                )
                
                # 计算文档共现
                docs1 = entities[entity1]['documents']
                docs2 = entities[entity2]['documents']
                co_occurrence = len(docs1.intersection(docs2))
                
                # 如果相似度高或共现频繁，添加边
                if sim > self.similarity_threshold or co_occurrence > 0:
                    weight = 0.7 * sim + 0.3 * min(co_occurrence / 5.0, 1.0)
                    graph.add_edge(entity1, entity2, 
                                 weight=weight,
                                 similarity=sim,
                                 co_occurrence=co_occurrence)
        
        return graph
    
    def index_documents(self, documents: List[str]) -> Dict[str, Any]:
        """
        索引文档集合
        
        Args:
            documents: 文档列表
            
        Returns:
            索引结果
        """
        print("Extracting entities...")
        entities = self.extract_entities(documents)
        
        print(f"Found {len(entities)} unique entities")
        
        print("Computing embeddings...")
        embeddings = self.compute_entity_embeddings(entities)
        
        print("Building graph...")
        self.graph = self.build_entity_graph(entities, embeddings)
        self.entity_embeddings = embeddings
        
        # 存储实体上下文和类型信息
        for entity_name, entity_info in entities.items():
            self.entity_contexts[entity_name] = entity_info['contexts']
            self.entity_types[entity_name] = entity_info['type']
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return {
            'num_entities': len(entities),
            'num_edges': self.graph.number_of_edges(),
            'entity_types': {etype: sum(1 for e in entities.values() if e['type'] == etype) 
                           for etype in set(e['type'] for e in entities.values())},
            'graph': self.graph
        }
    
    def search_entities(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        搜索相关实体
        
        Args:
            query: 查询文本
            top_k: 返回的实体数量
            
        Returns:
            相关实体列表
        """
        query_embedding = self.embedding_model.encode(query)
        
        similarities = []
        for entity_name, entity_embedding in self.entity_embeddings.items():
            sim = np.dot(query_embedding, entity_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
            )
            similarities.append((entity_name, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_entity_subgraph(self, entity: str, depth: int = 2) -> nx.Graph:
        """
        获取实体的子图
        
        Args:
            entity: 中心实体
            depth: 搜索深度
            
        Returns:
            实体子图
        """
        if entity not in self.graph:
            return nx.Graph()
        
        # 使用BFS获取指定深度的邻居
        nodes = set([entity])
        current_level = set([entity])
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                neighbors = set(self.graph.neighbors(node))
                next_level.update(neighbors)
            nodes.update(next_level)
            current_level = next_level
        
        return self.graph.subgraph(nodes).copy()
    
    def get_entity_context(self, entity: str, max_contexts: int = 3) -> List[str]:
        """
        获取实体的上下文信息
        
        Args:
            entity: 实体名称
            max_contexts: 最大上下文数量
            
        Returns:
            上下文列表
        """
        return self.entity_contexts.get(entity, [])[:max_contexts]
    
    def save_index(self, filepath: str):
        """
        保存索引到文件
        
        Args:
            filepath: 文件路径
        """
        index_data = {
            'graph_data': nx.node_link_data(self.graph),
            'entity_embeddings': {k: v.tolist() for k, v in self.entity_embeddings.items()},
            'entity_contexts': dict(self.entity_contexts),
            'entity_types': self.entity_types
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def load_index(self, filepath: str):
        """
        从文件加载索引
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.graph = nx.node_link_graph(index_data['graph_data'])
        self.entity_embeddings = {k: np.array(v) for k, v in index_data['entity_embeddings'].items()}
        self.entity_contexts = defaultdict(list, index_data['entity_contexts'])
        self.entity_types = index_data['entity_types']