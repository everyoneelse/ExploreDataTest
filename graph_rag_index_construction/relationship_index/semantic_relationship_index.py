"""
Semantic Relationship Index Implementation
语义关系索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import defaultdict
import json
import re


class SemanticRelationshipIndex:
    """
    语义关系索引
    
    该类实现了基于语义理解的关系索引构建方法，
    通过NLP技术识别和建模文档间的语义关系。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 spacy_model: str = "en_core_web_sm",
                 similarity_threshold: float = 0.6,
                 relation_types: Optional[List[str]] = None):
        """
        初始化语义关系索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            spacy_model: SpaCy模型名称
            similarity_threshold: 语义相似度阈值
            relation_types: 支持的关系类型列表
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        
        # 加载SpaCy模型
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"SpaCy model {spacy_model} not found. Please install it with: python -m spacy download {spacy_model}")
            self.nlp = None
        
        # 预定义的关系类型
        if relation_types is None:
            self.relation_types = [
                'similarity',           # 相似关系
                'causation',           # 因果关系
                'comparison',          # 比较关系
                'contradiction',       # 矛盾关系
                'elaboration',         # 阐述关系
                'example',             # 举例关系
                'temporal',            # 时间关系
                'spatial',             # 空间关系
                'part_whole',          # 部分-整体关系
                'classification'       # 分类关系
            ]
        else:
            self.relation_types = relation_types
        
        # 关系模式和关键词
        self.relation_patterns = {
            'causation': {
                'keywords': ['because', 'since', 'due to', 'as a result', 'therefore', 'consequently', 'leads to', 'causes'],
                'patterns': [r'because of', r'due to', r'as a result of', r'leads to', r'causes?']
            },
            'comparison': {
                'keywords': ['similar', 'different', 'like', 'unlike', 'compared to', 'in contrast', 'whereas', 'while'],
                'patterns': [r'similar to', r'different from', r'compared to', r'in contrast to', r'whereas']
            },
            'contradiction': {
                'keywords': ['however', 'but', 'nevertheless', 'on the contrary', 'despite', 'although'],
                'patterns': [r'however', r'but', r'nevertheless', r'on the contrary', r'despite']
            },
            'elaboration': {
                'keywords': ['furthermore', 'moreover', 'additionally', 'in addition', 'also', 'besides'],
                'patterns': [r'furthermore', r'moreover', r'additionally', r'in addition']
            },
            'example': {
                'keywords': ['for example', 'such as', 'including', 'namely', 'for instance'],
                'patterns': [r'for example', r'such as', r'including', r'namely', r'for instance']
            },
            'temporal': {
                'keywords': ['before', 'after', 'during', 'while', 'when', 'then', 'next', 'previously'],
                'patterns': [r'before', r'after', r'during', r'while', r'when', r'then']
            }
        }
        
        # 存储关系信息
        self.documents = {}
        self.document_embeddings = {}
        self.relationships = []
        self.graph = nx.MultiDiGraph()  # 使用多重有向图支持多种关系类型
        
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """
        添加文档到索引
        
        Args:
            doc_id: 文档ID
            text: 文档文本
            metadata: 文档元数据
        """
        if metadata is None:
            metadata = {}
            
        # 存储文档信息
        self.documents[doc_id] = {
            'text': text,
            'metadata': metadata,
            'sentences': self._split_sentences(text),
            'entities': self._extract_entities(text) if self.nlp else [],
            'keywords': self._extract_keywords(text)
        }
        
        # 生成文档嵌入
        embedding = self.embedding_model.encode([text])[0]
        self.document_embeddings[doc_id] = embedding
        
        # 添加节点到图
        self.graph.add_node(doc_id,
                          text=text[:200] + "..." if len(text) > 200 else text,
                          metadata=metadata,
                          embedding=embedding)
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        分割句子
        
        Args:
            text: 输入文本
            
        Returns:
            句子列表
        """
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # 简单的句子分割
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表
        """
        if not self.nlp:
            # 简单的关键词提取
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 3:  # 过滤短词
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:top_k]]
        
        doc = self.nlp(text)
        keywords = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                len(token.text) > 2 and
                token.pos_ in ['NOUN', 'ADJ', 'VERB']):
                keywords.append(token.lemma_.lower())
        
        # 计算频率并返回top_k
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _detect_relation_type(self, text1: str, text2: str) -> List[Tuple[str, float]]:
        """
        检测两个文本间的关系类型
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            关系类型和置信度列表
        """
        detected_relations = []
        combined_text = f"{text1} {text2}".lower()
        
        for relation_type, patterns_info in self.relation_patterns.items():
            confidence = 0.0
            
            # 检查关键词
            keywords = patterns_info['keywords']
            keyword_count = sum(1 for keyword in keywords if keyword in combined_text)
            keyword_score = keyword_count / len(keywords) if keywords else 0
            
            # 检查模式
            patterns = patterns_info['patterns']
            pattern_count = sum(1 for pattern in patterns if re.search(pattern, combined_text))
            pattern_score = pattern_count / len(patterns) if patterns else 0
            
            # 综合置信度
            confidence = (keyword_score + pattern_score) / 2
            
            if confidence > 0:
                detected_relations.append((relation_type, confidence))
        
        # 按置信度排序
        detected_relations.sort(key=lambda x: x[1], reverse=True)
        return detected_relations
    
    def _calculate_semantic_similarity(self, doc_id1: str, doc_id2: str) -> float:
        """
        计算两个文档的语义相似度
        
        Args:
            doc_id1: 第一个文档ID
            doc_id2: 第二个文档ID
            
        Returns:
            相似度分数
        """
        emb1 = self.document_embeddings[doc_id1]
        emb2 = self.document_embeddings[doc_id2]
        
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def _calculate_entity_overlap(self, doc_id1: str, doc_id2: str) -> float:
        """
        计算两个文档的实体重叠度
        
        Args:
            doc_id1: 第一个文档ID
            doc_id2: 第二个文档ID
            
        Returns:
            实体重叠度
        """
        entities1 = set(ent['text'].lower() for ent in self.documents[doc_id1]['entities'])
        entities2 = set(ent['text'].lower() for ent in self.documents[doc_id2]['entities'])
        
        if not entities1 and not entities2:
            return 0.0
        
        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_keyword_overlap(self, doc_id1: str, doc_id2: str) -> float:
        """
        计算两个文档的关键词重叠度
        
        Args:
            doc_id1: 第一个文档ID
            doc_id2: 第二个文档ID
            
        Returns:
            关键词重叠度
        """
        keywords1 = set(self.documents[doc_id1]['keywords'])
        keywords2 = set(self.documents[doc_id2]['keywords'])
        
        if not keywords1 and not keywords2:
            return 0.0
        
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def build_semantic_relationships(self) -> None:
        """
        构建语义关系
        """
        print("Building semantic relationships...")
        
        doc_ids = list(self.documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                # 计算多种相似度指标
                semantic_sim = self._calculate_semantic_similarity(doc_id1, doc_id2)
                entity_overlap = self._calculate_entity_overlap(doc_id1, doc_id2)
                keyword_overlap = self._calculate_keyword_overlap(doc_id1, doc_id2)
                
                # 综合相似度
                combined_similarity = (semantic_sim * 0.6 + 
                                     entity_overlap * 0.2 + 
                                     keyword_overlap * 0.2)
                
                if combined_similarity >= self.similarity_threshold:
                    # 检测关系类型
                    text1 = self.documents[doc_id1]['text']
                    text2 = self.documents[doc_id2]['text']
                    detected_relations = self._detect_relation_type(text1, text2)
                    
                    # 如果没有检测到特定关系，使用相似关系
                    if not detected_relations or detected_relations[0][1] < 0.3:
                        relation_type = 'similarity'
                        confidence = combined_similarity
                    else:
                        relation_type, confidence = detected_relations[0]
                        # 结合语义相似度调整置信度
                        confidence = (confidence + combined_similarity) / 2
                    
                    # 创建关系记录
                    relationship = {
                        'source': doc_id1,
                        'target': doc_id2,
                        'relation_type': relation_type,
                        'confidence': confidence,
                        'semantic_similarity': semantic_sim,
                        'entity_overlap': entity_overlap,
                        'keyword_overlap': keyword_overlap,
                        'combined_similarity': combined_similarity
                    }
                    
                    self.relationships.append(relationship)
                    
                    # 添加边到图
                    self.graph.add_edge(doc_id1, doc_id2,
                                      relation_type=relation_type,
                                      confidence=confidence,
                                      weight=combined_similarity,
                                      semantic_similarity=semantic_sim,
                                      entity_overlap=entity_overlap,
                                      keyword_overlap=keyword_overlap)
    
    def get_related_documents(self, doc_id: str, 
                            relation_type: Optional[str] = None,
                            min_confidence: float = 0.5,
                            top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        获取相关文档
        
        Args:
            doc_id: 文档ID
            relation_type: 关系类型过滤
            min_confidence: 最小置信度
            top_k: 返回前k个结果
            
        Returns:
            相关文档列表 (doc_id, relation_type, confidence)
        """
        if doc_id not in self.graph:
            return []
        
        related = []
        
        # 获取所有邻居
        for neighbor in self.graph.neighbors(doc_id):
            for edge_data in self.graph[doc_id][neighbor].values():
                rel_type = edge_data['relation_type']
                confidence = edge_data['confidence']
                
                if confidence >= min_confidence:
                    if relation_type is None or rel_type == relation_type:
                        related.append((neighbor, rel_type, confidence))
        
        # 按置信度排序
        related.sort(key=lambda x: x[2], reverse=True)
        return related[:top_k]
    
    def find_documents_by_relation(self, relation_type: str, 
                                 min_confidence: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        根据关系类型查找文档对
        
        Args:
            relation_type: 关系类型
            min_confidence: 最小置信度
            
        Returns:
            文档对列表 (source_doc, target_doc, confidence)
        """
        results = []
        
        for rel in self.relationships:
            if (rel['relation_type'] == relation_type and 
                rel['confidence'] >= min_confidence):
                results.append((rel['source'], rel['target'], rel['confidence']))
        
        # 按置信度排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def query_semantic_relationships(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        查询语义关系
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            查询结果列表 (doc_id, similarity, relation_info)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = []
        for doc_id, doc_embedding in self.document_embeddings.items():
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            
            # 获取文档的关系信息
            relations = self.get_related_documents(doc_id, top_k=3)
            relation_info = f"Relations: {', '.join([f'{rel[1]}({rel[2]:.2f})' for rel in relations])}"
            
            results.append((doc_id, similarity, relation_info))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        构建语义关系索引
        
        Args:
            documents: 文档列表
        """
        print(f"Building semantic relationship index for {len(documents)} documents...")
        
        # 添加所有文档
        for doc in documents:
            self.add_document(
                doc_id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata', {})
            )
        
        # 构建语义关系
        self.build_semantic_relationships()
        
        print(f"Semantic relationship index built successfully!")
        print(f"- Documents: {len(self.documents)}")
        print(f"- Relationships: {len(self.relationships)}")
        print(f"- Graph nodes: {self.graph.number_of_nodes()}")
        print(f"- Graph edges: {self.graph.number_of_edges()}")
        
        # 关系类型统计
        relation_stats = {}
        for rel in self.relationships:
            rel_type = rel['relation_type']
            relation_stats[rel_type] = relation_stats.get(rel_type, 0) + 1
        
        print("Relationship type distribution:")
        for rel_type, count in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rel_type}: {count}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            图统计信息字典
        """
        if not self.graph.nodes():
            return {}
        
        # 关系类型统计
        relation_stats = {}
        for rel in self.relationships:
            rel_type = rel['relation_type']
            relation_stats[rel_type] = relation_stats.get(rel_type, 0) + 1
        
        return {
            'num_documents': len(self.documents),
            'num_relationships': len(self.relationships),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()) if self.graph.nodes() else 0,
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph.to_undirected()),
            'relation_type_distribution': relation_stats,
            'avg_confidence': np.mean([rel['confidence'] for rel in self.relationships]) if self.relationships else 0
        }
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        index_data = {
            'documents': self.documents,
            'relationships': self.relationships,
            'graph_data': nx.node_link_data(self.graph),
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'relation_types': self.relation_types,
                'relation_patterns': self.relation_patterns
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
        
        self.documents = index_data['documents']
        self.relationships = index_data['relationships']
        self.graph = nx.node_link_graph(index_data['graph_data'])
        
        # 重新生成文档嵌入向量
        for doc_id, doc_info in self.documents.items():
            embedding = self.embedding_model.encode([doc_info['text']])[0]
            self.document_embeddings[doc_id] = embedding