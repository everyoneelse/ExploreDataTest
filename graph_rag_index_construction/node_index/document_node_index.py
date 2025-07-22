"""
Document-based Node Index Implementation
基于文档的节点索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import hashlib


class DocumentNodeIndex:
    """
    基于文档的节点索引
    
    该类实现了基于文档的节点索引构建方法，
    将文档作为图的节点，基于文档内容相似度建立连接。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        初始化文档节点索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            similarity_threshold: 文档相似度阈值
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 存储文档信息
        self.documents = {}
        self.document_embeddings = {}
        self.document_chunks = {}
        self.graph = nx.Graph()
        
    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        将文档分割成块
        
        Args:
            text: 文档文本
            doc_id: 文档ID
            
        Returns:
            文档块列表
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = f"{doc_id}_chunk_{i // (self.chunk_size - self.chunk_overlap)}"
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'hash': chunk_hash,
                'start_idx': i,
                'end_idx': min(i + self.chunk_size, len(words)),
                'parent_doc': doc_id
            })
            
        return chunks
    
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
            'word_count': len(text.split()),
            'char_count': len(text)
        }
        
        # 分块处理
        chunks = self.chunk_document(text, doc_id)
        self.document_chunks[doc_id] = chunks
        
        # 生成嵌入向量
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts)
        
        # 存储文档级别的嵌入（所有块的平均）
        doc_embedding = np.mean(chunk_embeddings, axis=0)
        self.document_embeddings[doc_id] = {
            'doc_embedding': doc_embedding,
            'chunk_embeddings': chunk_embeddings,
            'chunk_texts': chunk_texts
        }
        
        # 添加节点到图
        self.graph.add_node(doc_id, 
                           text=text[:200] + "..." if len(text) > 200 else text,
                           metadata=metadata,
                           embedding=doc_embedding,
                           chunk_count=len(chunks))
    
    def build_document_similarities(self) -> Dict[Tuple[str, str], float]:
        """
        计算文档间相似度
        
        Returns:
            文档对相似度字典
        """
        similarities = {}
        doc_ids = list(self.documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                # 计算文档级别相似度
                emb1 = self.document_embeddings[doc_id1]['doc_embedding']
                emb2 = self.document_embeddings[doc_id2]['doc_embedding']
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                similarities[(doc_id1, doc_id2)] = similarity
                
                # 如果相似度超过阈值，添加边
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(doc_id1, doc_id2, 
                                      weight=similarity,
                                      relation_type='document_similarity')
        
        return similarities
    
    def build_chunk_level_connections(self, cross_doc_threshold: float = 0.7) -> None:
        """
        基于块级别构建跨文档连接
        
        Args:
            cross_doc_threshold: 跨文档块相似度阈值
        """
        all_chunks = []
        chunk_to_doc = {}
        
        # 收集所有块
        for doc_id, chunks in self.document_chunks.items():
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_to_doc[chunk['chunk_id']] = doc_id
        
        # 计算所有块的嵌入
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts)
        
        # 找到跨文档的高相似度块对
        for i, chunk1 in enumerate(all_chunks):
            doc1 = chunk_to_doc[chunk1['chunk_id']]
            
            for j, chunk2 in enumerate(all_chunks[i+1:], i+1):
                doc2 = chunk_to_doc[chunk2['chunk_id']]
                
                # 只考虑跨文档的块
                if doc1 != doc2:
                    similarity = cosine_similarity([chunk_embeddings[i]], 
                                                 [chunk_embeddings[j]])[0][0]
                    
                    if similarity >= cross_doc_threshold:
                        # 加强两个文档间的连接
                        if self.graph.has_edge(doc1, doc2):
                            current_weight = self.graph[doc1][doc2]['weight']
                            self.graph[doc1][doc2]['weight'] = max(current_weight, similarity)
                        else:
                            self.graph.add_edge(doc1, doc2, 
                                              weight=similarity,
                                              relation_type='chunk_similarity')
    
    def get_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        获取与查询最相似的文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            相似文档列表 (doc_id, similarity_score)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for doc_id, embeddings in self.document_embeddings.items():
            doc_embedding = embeddings['doc_embedding']
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities.append((doc_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_document_neighbors(self, doc_id: str, max_neighbors: int = 5) -> List[Tuple[str, float]]:
        """
        获取文档的邻居节点
        
        Args:
            doc_id: 文档ID
            max_neighbors: 最大邻居数量
            
        Returns:
            邻居文档列表 (neighbor_id, weight)
        """
        if doc_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(doc_id):
            weight = self.graph[doc_id][neighbor]['weight']
            neighbors.append((neighbor, weight))
        
        # 按权重排序
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        批量构建文档索引
        
        Args:
            documents: 文档列表，每个文档包含id, text, metadata
        """
        print(f"Building document index for {len(documents)} documents...")
        
        # 添加所有文档
        for doc in documents:
            self.add_document(
                doc_id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata', {})
            )
        
        print("Computing document similarities...")
        # 构建文档相似度
        similarities = self.build_document_similarities()
        
        print("Building chunk-level connections...")
        # 构建块级别连接
        self.build_chunk_level_connections()
        
        print(f"Index built successfully!")
        print(f"- Documents: {len(self.documents)}")
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
            'num_documents': len(self.documents),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes()),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'total_chunks': sum(len(chunks) for chunks in self.document_chunks.values())
        }
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        index_data = {
            'documents': self.documents,
            'document_chunks': self.document_chunks,
            'graph_data': nx.node_link_data(self.graph),
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
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
        self.document_chunks = index_data['document_chunks']
        self.graph = nx.node_link_graph(index_data['graph_data'])
        
        # 重新生成嵌入向量（因为numpy数组无法直接序列化）
        for doc_id, text_info in self.documents.items():
            chunks = self.document_chunks[doc_id]
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            doc_embedding = np.mean(chunk_embeddings, axis=0)
            
            self.document_embeddings[doc_id] = {
                'doc_embedding': doc_embedding,
                'chunk_embeddings': chunk_embeddings,
                'chunk_texts': chunk_texts
            }