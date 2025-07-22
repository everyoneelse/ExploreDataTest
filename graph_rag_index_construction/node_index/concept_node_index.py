"""
Concept-based Node Index Implementation
基于概念的节点索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
import json
import re


class ConceptNodeIndex:
    """
    基于概念的节点索引
    
    该类实现了基于概念抽取的节点索引构建方法，
    通过主题建模和关键词提取构建概念图。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 num_concepts: int = 50,
                 min_concept_freq: int = 3,
                 concept_similarity_threshold: float = 0.6):
        """
        初始化概念节点索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            num_concepts: 概念数量
            min_concept_freq: 概念最小频率
            concept_similarity_threshold: 概念相似度阈值
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.num_concepts = num_concepts
        self.min_concept_freq = min_concept_freq
        self.concept_similarity_threshold = concept_similarity_threshold
        
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        
        # 存储概念信息
        self.concepts = {}
        self.concept_embeddings = {}
        self.concept_documents = defaultdict(list)
        self.document_concepts = defaultdict(list)
        self.graph = nx.Graph()
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        提取关键短语
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键短语
            
        Returns:
            关键短语列表 (phrase, score)
        """
        # 简单的关键短语提取方法
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # 过滤停用词和标点
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # 计算词频
        word_freq = Counter(words)
        
        # 提取n-gram短语
        phrases = []
        for i in range(len(words)):
            for j in range(i+1, min(i+4, len(words)+1)):  # 1-3 gram
                phrase = ' '.join(words[i:j])
                if len(phrase) > 2:  # 过滤太短的短语
                    phrases.append(phrase)
        
        # 计算短语频率
        phrase_freq = Counter(phrases)
        
        # 计算TF-IDF分数（简化版本）
        total_words = len(words)
        scored_phrases = []
        
        for phrase, freq in phrase_freq.items():
            if freq >= 2:  # 至少出现2次
                tf = freq / total_words
                # 简化的IDF计算
                idf = np.log(total_words / freq)
                score = tf * idf
                scored_phrases.append((phrase, score))
        
        # 按分数排序
        scored_phrases.sort(key=lambda x: x[1], reverse=True)
        return scored_phrases[:top_k]
    
    def extract_concepts_from_document(self, doc_id: str, text: str) -> List[Dict[str, Any]]:
        """
        从文档中提取概念
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            
        Returns:
            概念列表
        """
        concepts = []
        
        # 提取关键短语作为概念
        key_phrases = self.extract_key_phrases(text, top_k=20)
        
        for phrase, score in key_phrases:
            concept_id = f"concept_{phrase.replace(' ', '_')}"
            
            concepts.append({
                'concept_id': concept_id,
                'phrase': phrase,
                'score': score,
                'source_doc': doc_id,
                'type': 'key_phrase'
            })
        
        return concepts
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """
        添加文档并提取概念
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            metadata: 文档元数据
        """
        if metadata is None:
            metadata = {}
        
        # 提取文档概念
        doc_concepts = self.extract_concepts_from_document(doc_id, text)
        
        # 处理每个概念
        for concept_info in doc_concepts:
            concept_id = concept_info['concept_id']
            phrase = concept_info['phrase']
            score = concept_info['score']
            
            # 如果概念不存在，创建新概念
            if concept_id not in self.concepts:
                self.concepts[concept_id] = {
                    'phrase': phrase,
                    'frequency': 0,
                    'documents': [],
                    'total_score': 0.0,
                    'type': concept_info['type']
                }
            
            # 更新概念信息
            self.concepts[concept_id]['frequency'] += 1
            self.concepts[concept_id]['total_score'] += score
            self.concepts[concept_id]['documents'].append(doc_id)
            
            # 建立文档-概念关系
            self.concept_documents[concept_id].append(doc_id)
            self.document_concepts[doc_id].append(concept_id)
    
    def build_concept_graph(self) -> None:
        """
        构建概念图
        """
        # 过滤低频概念
        filtered_concepts = {
            cid: info for cid, info in self.concepts.items()
            if info['frequency'] >= self.min_concept_freq
        }
        
        print(f"Filtered concepts: {len(filtered_concepts)} from {len(self.concepts)}")
        
        # 生成概念嵌入
        concept_phrases = [info['phrase'] for info in filtered_concepts.values()]
        if concept_phrases:
            concept_embeddings = self.embedding_model.encode(concept_phrases)
            
            for i, (concept_id, info) in enumerate(filtered_concepts.items()):
                self.concept_embeddings[concept_id] = concept_embeddings[i]
                
                # 添加概念节点到图
                avg_score = info['total_score'] / info['frequency']
                self.graph.add_node(concept_id,
                                  phrase=info['phrase'],
                                  frequency=info['frequency'],
                                  avg_score=avg_score,
                                  type='concept',
                                  documents=info['documents'])
        
        # 建立概念间的相似度连接
        concept_ids = list(filtered_concepts.keys())
        for i, concept_id1 in enumerate(concept_ids):
            for concept_id2 in concept_ids[i+1:]:
                # 计算概念相似度
                emb1 = self.concept_embeddings[concept_id1]
                emb2 = self.concept_embeddings[concept_id2]
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                if similarity >= self.concept_similarity_threshold:
                    self.graph.add_edge(concept_id1, concept_id2,
                                      weight=similarity,
                                      relation_type='concept_similarity')
        
        # 建立基于共现的概念连接
        self._build_cooccurrence_connections()
    
    def _build_cooccurrence_connections(self) -> None:
        """
        基于共现建立概念连接
        """
        # 计算概念共现矩阵
        concept_ids = list(self.concept_embeddings.keys())
        cooccurrence_matrix = np.zeros((len(concept_ids), len(concept_ids)))
        
        concept_to_idx = {cid: i for i, cid in enumerate(concept_ids)}
        
        # 统计概念共现
        for doc_id, concepts in self.document_concepts.items():
            doc_concepts = [c for c in concepts if c in concept_to_idx]
            
            for i, concept1 in enumerate(doc_concepts):
                for concept2 in doc_concepts[i+1:]:
                    idx1 = concept_to_idx[concept1]
                    idx2 = concept_to_idx[concept2]
                    cooccurrence_matrix[idx1][idx2] += 1
                    cooccurrence_matrix[idx2][idx1] += 1
        
        # 根据共现频率添加边
        min_cooccurrence = 2  # 最小共现次数
        for i, concept_id1 in enumerate(concept_ids):
            for j, concept_id2 in enumerate(concept_ids[i+1:], i+1):
                cooccurrence_count = cooccurrence_matrix[i][j]
                
                if cooccurrence_count >= min_cooccurrence:
                    # 计算共现强度
                    freq1 = self.concepts[concept_id1]['frequency']
                    freq2 = self.concepts[concept_id2]['frequency']
                    
                    # PMI (Pointwise Mutual Information)
                    total_docs = len(self.document_concepts)
                    pmi = np.log((cooccurrence_count * total_docs) / (freq1 * freq2))
                    
                    if pmi > 0:  # 正相关
                        # 如果已存在边，更新权重
                        if self.graph.has_edge(concept_id1, concept_id2):
                            current_weight = self.graph[concept_id1][concept_id2]['weight']
                            self.graph[concept_id1][concept_id2]['weight'] = max(current_weight, pmi)
                        else:
                            self.graph.add_edge(concept_id1, concept_id2,
                                              weight=pmi,
                                              relation_type='cooccurrence',
                                              cooccurrence_count=cooccurrence_count)
    
    def get_related_concepts(self, concept_id: str, max_related: int = 5) -> List[Tuple[str, float]]:
        """
        获取相关概念
        
        Args:
            concept_id: 概念ID
            max_related: 最大相关概念数量
            
        Returns:
            相关概念列表 (concept_id, weight)
        """
        if concept_id not in self.graph:
            return []
        
        related = []
        for neighbor in self.graph.neighbors(concept_id):
            weight = self.graph[concept_id][neighbor]['weight']
            related.append((neighbor, weight))
        
        # 按权重排序
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_related]
    
    def find_concepts_by_query(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        根据查询找到相关概念
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            相关概念列表 (concept_id, similarity_score)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for concept_id, embedding in self.concept_embeddings.items():
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((concept_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_concept_documents(self, concept_id: str) -> List[str]:
        """
        获取包含特定概念的文档
        
        Args:
            concept_id: 概念ID
            
        Returns:
            文档ID列表
        """
        return self.concept_documents.get(concept_id, [])
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        批量构建概念索引
        
        Args:
            documents: 文档列表，每个文档包含id, text, metadata
        """
        print(f"Building concept index for {len(documents)} documents...")
        
        # 处理所有文档
        for doc in documents:
            self.add_document(
                doc_id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata', {})
            )
        
        print("Building concept graph...")
        # 构建概念图
        self.build_concept_graph()
        
        print(f"Concept index built successfully!")
        print(f"- Total concepts extracted: {len(self.concepts)}")
        print(f"- Filtered concepts: {len(self.concept_embeddings)}")
        print(f"- Graph nodes: {self.graph.number_of_nodes()}")
        print(f"- Graph edges: {self.graph.number_of_edges()}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            图统计信息字典
        """
        if not self.graph.nodes():
            return {}
        
        return {
            'num_concepts': len(self.concepts),
            'num_filtered_concepts': len(self.concept_embeddings),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'concept_frequency_stats': {
                'min': min(info['frequency'] for info in self.concepts.values()) if self.concepts else 0,
                'max': max(info['frequency'] for info in self.concepts.values()) if self.concepts else 0,
                'mean': np.mean([info['frequency'] for info in self.concepts.values()]) if self.concepts else 0
            }
        }
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        index_data = {
            'concepts': self.concepts,
            'concept_documents': dict(self.concept_documents),
            'document_concepts': dict(self.document_concepts),
            'graph_data': nx.node_link_data(self.graph),
            'config': {
                'num_concepts': self.num_concepts,
                'min_concept_freq': self.min_concept_freq,
                'concept_similarity_threshold': self.concept_similarity_threshold
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def load_index(self, filepath: str) -> None:
        """
        从文件加载索引
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.concepts = index_data['concepts']
        self.concept_documents = defaultdict(list, index_data['concept_documents'])
        self.document_concepts = defaultdict(list, index_data['document_concepts'])
        self.graph = nx.node_link_graph(index_data['graph_data'])
        
        # 重新生成概念嵌入向量
        filtered_concepts = {
            cid: info for cid, info in self.concepts.items()
            if info['frequency'] >= self.min_concept_freq
        }
        
        concept_phrases = [info['phrase'] for info in filtered_concepts.values()]
        if concept_phrases:
            concept_embeddings = self.embedding_model.encode(concept_phrases)
            
            for i, concept_id in enumerate(filtered_concepts.keys()):
                self.concept_embeddings[concept_id] = concept_embeddings[i]