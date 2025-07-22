"""
Louvain Community Index Implementation
基于Louvain算法的社区索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import json

try:
    import community as community_louvain
except ImportError:
    print("Warning: python-louvain not installed. Please install with: pip install python-louvain")
    community_louvain = None


class LouvainCommunityIndex:
    """
    基于Louvain算法的社区索引
    
    该类实现了基于Louvain社区发现算法的索引构建方法，
    通过社区结构组织和检索文档。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 resolution: float = 1.0,
                 randomize: Optional[int] = None):
        """
        初始化Louvain社区索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            similarity_threshold: 相似度阈值
            resolution: Louvain算法分辨率参数
            randomize: 随机种子
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.resolution = resolution
        self.randomize = randomize
        
        # 存储文档和社区信息
        self.documents = {}
        self.document_embeddings = {}
        self.communities = {}
        self.document_to_community = {}
        self.community_summaries = {}
        self.community_keywords = {}
        
        # 图结构
        self.graph = nx.Graph()
        self.community_graph = nx.Graph()  # 社区间的图
        
        # 社区统计信息
        self.modularity_score = 0.0
        self.num_communities = 0
    
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
        
        # 生成文档嵌入
        embedding = self.embedding_model.encode([text])[0]
        self.document_embeddings[doc_id] = embedding
        
        # 添加节点到图
        self.graph.add_node(doc_id,
                          text=text[:200] + "..." if len(text) > 200 else text,
                          metadata=metadata,
                          embedding=embedding)
    
    def build_similarity_graph(self) -> None:
        """
        构建文档相似度图
        """
        print("Building document similarity graph...")
        
        doc_ids = list(self.documents.keys())
        
        for i, doc_id1 in enumerate(doc_ids):
            for doc_id2 in doc_ids[i+1:]:
                # 计算相似度
                emb1 = self.document_embeddings[doc_id1]
                emb2 = self.document_embeddings[doc_id2]
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                # 如果相似度超过阈值，添加边
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(doc_id1, doc_id2, weight=similarity)
        
        print(f"Similarity graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def detect_communities(self) -> None:
        """
        使用Louvain算法检测社区
        """
        if community_louvain is None:
            raise ImportError("python-louvain package is required for Louvain community detection")
        
        print("Detecting communities using Louvain algorithm...")
        
        # 运行Louvain算法
        partition = community_louvain.best_partition(
            self.graph, 
            resolution=self.resolution,
            random_state=self.randomize
        )
        
        # 计算模块度
        self.modularity_score = community_louvain.modularity(partition, self.graph)
        
        # 组织社区信息
        self.communities = defaultdict(list)
        self.document_to_community = {}
        
        for doc_id, community_id in partition.items():
            self.communities[community_id].append(doc_id)
            self.document_to_community[doc_id] = community_id
        
        self.num_communities = len(self.communities)
        
        print(f"Found {self.num_communities} communities with modularity {self.modularity_score:.4f}")
        
        # 生成社区摘要和关键词
        self._generate_community_summaries()
        
        # 构建社区间图
        self._build_community_graph()
    
    def _generate_community_summaries(self) -> None:
        """
        生成社区摘要和关键词
        """
        print("Generating community summaries...")
        
        for community_id, doc_ids in self.communities.items():
            # 收集社区中所有文档的文本
            community_texts = [self.documents[doc_id]['text'] for doc_id in doc_ids]
            
            # 生成社区摘要（取前几个文档的开头）
            summary_parts = []
            for i, text in enumerate(community_texts[:3]):  # 只取前3个文档
                summary_parts.append(text[:100] + "...")
            
            community_summary = " | ".join(summary_parts)
            self.community_summaries[community_id] = community_summary
            
            # 提取社区关键词
            all_text = " ".join(community_texts)
            keywords = self._extract_community_keywords(all_text)
            self.community_keywords[community_id] = keywords
    
    def _extract_community_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取社区关键词
        
        Args:
            text: 社区文本
            top_k: 返回前k个关键词
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取
        words = text.lower().split()
        
        # 过滤停用词和短词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # 计算词频
        word_freq = Counter(filtered_words)
        
        # 返回最频繁的词
        return [word for word, freq in word_freq.most_common(top_k)]
    
    def _build_community_graph(self) -> None:
        """
        构建社区间图
        """
        print("Building inter-community graph...")
        
        # 为每个社区添加节点
        for community_id, doc_ids in self.communities.items():
            # 计算社区的中心嵌入（所有文档嵌入的平均值）
            community_embeddings = [self.document_embeddings[doc_id] for doc_id in doc_ids]
            community_embedding = np.mean(community_embeddings, axis=0)
            
            self.community_graph.add_node(
                community_id,
                size=len(doc_ids),
                summary=self.community_summaries[community_id],
                keywords=self.community_keywords[community_id],
                embedding=community_embedding,
                documents=doc_ids
            )
        
        # 计算社区间连接
        community_ids = list(self.communities.keys())
        
        for i, comm_id1 in enumerate(community_ids):
            for comm_id2 in community_ids[i+1:]:
                # 计算社区间文档连接数
                inter_connections = 0
                total_possible = len(self.communities[comm_id1]) * len(self.communities[comm_id2])
                
                for doc1 in self.communities[comm_id1]:
                    for doc2 in self.communities[comm_id2]:
                        if self.graph.has_edge(doc1, doc2):
                            inter_connections += 1
                
                # 如果有足够的连接，添加社区间边
                if inter_connections > 0:
                    connection_strength = inter_connections / total_possible
                    
                    # 计算社区嵌入相似度
                    emb1 = self.community_graph.nodes[comm_id1]['embedding']
                    emb2 = self.community_graph.nodes[comm_id2]['embedding']
                    embedding_similarity = cosine_similarity([emb1], [emb2])[0][0]
                    
                    # 综合权重
                    weight = (connection_strength + embedding_similarity) / 2
                    
                    if weight > 0.1:  # 阈值
                        self.community_graph.add_edge(
                            comm_id1, comm_id2,
                            weight=weight,
                            inter_connections=inter_connections,
                            connection_strength=connection_strength,
                            embedding_similarity=embedding_similarity
                        )
    
    def get_community_documents(self, community_id: int) -> List[str]:
        """
        获取社区中的文档
        
        Args:
            community_id: 社区ID
            
        Returns:
            文档ID列表
        """
        return self.communities.get(community_id, [])
    
    def get_document_community(self, doc_id: str) -> Optional[int]:
        """
        获取文档所属社区
        
        Args:
            doc_id: 文档ID
            
        Returns:
            社区ID
        """
        return self.document_to_community.get(doc_id)
    
    def get_related_communities(self, community_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        获取相关社区
        
        Args:
            community_id: 社区ID
            top_k: 返回前k个结果
            
        Returns:
            相关社区列表 (community_id, weight)
        """
        if community_id not in self.community_graph:
            return []
        
        related = []
        for neighbor in self.community_graph.neighbors(community_id):
            weight = self.community_graph[community_id][neighbor]['weight']
            related.append((neighbor, weight))
        
        # 按权重排序
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_k]
    
    def query_communities(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        查询相关社区
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            相关社区列表 (community_id, similarity, summary)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = []
        for community_id in self.communities.keys():
            community_embedding = self.community_graph.nodes[community_id]['embedding']
            similarity = cosine_similarity([query_embedding], [community_embedding])[0][0]
            summary = self.community_summaries[community_id]
            
            results.append((community_id, similarity, summary))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_community_info(self, community_id: int) -> Optional[Dict[str, Any]]:
        """
        获取社区详细信息
        
        Args:
            community_id: 社区ID
            
        Returns:
            社区信息字典
        """
        if community_id not in self.communities:
            return None
        
        return {
            'community_id': community_id,
            'size': len(self.communities[community_id]),
            'documents': self.communities[community_id],
            'summary': self.community_summaries[community_id],
            'keywords': self.community_keywords[community_id],
            'related_communities': self.get_related_communities(community_id)
        }
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        构建Louvain社区索引
        
        Args:
            documents: 文档列表
        """
        print(f"Building Louvain community index for {len(documents)} documents...")
        
        # 添加所有文档
        for doc in documents:
            self.add_document(
                doc_id=doc['id'],
                text=doc['text'],
                metadata=doc.get('metadata', {})
            )
        
        # 构建相似度图
        self.build_similarity_graph()
        
        # 检测社区
        self.detect_communities()
        
        print(f"Louvain community index built successfully!")
        print(f"- Documents: {len(self.documents)}")
        print(f"- Communities: {self.num_communities}")
        print(f"- Modularity: {self.modularity_score:.4f}")
        print(f"- Graph edges: {self.graph.number_of_edges()}")
        print(f"- Community graph edges: {self.community_graph.number_of_edges()}")
        
        # 社区大小统计
        community_sizes = [len(docs) for docs in self.communities.values()]
        print(f"- Avg community size: {np.mean(community_sizes):.2f}")
        print(f"- Max community size: {max(community_sizes)}")
        print(f"- Min community size: {min(community_sizes)}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            图统计信息字典
        """
        if not self.graph.nodes():
            return {}
        
        community_sizes = [len(docs) for docs in self.communities.values()]
        
        return {
            'num_documents': len(self.documents),
            'num_communities': self.num_communities,
            'modularity_score': self.modularity_score,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'community_size_stats': {
                'mean': np.mean(community_sizes) if community_sizes else 0,
                'std': np.std(community_sizes) if community_sizes else 0,
                'min': min(community_sizes) if community_sizes else 0,
                'max': max(community_sizes) if community_sizes else 0
            },
            'community_graph_edges': self.community_graph.number_of_edges()
        }
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        index_data = {
            'documents': self.documents,
            'communities': dict(self.communities),
            'document_to_community': self.document_to_community,
            'community_summaries': self.community_summaries,
            'community_keywords': self.community_keywords,
            'graph_data': nx.node_link_data(self.graph),
            'community_graph_data': nx.node_link_data(self.community_graph),
            'modularity_score': self.modularity_score,
            'num_communities': self.num_communities,
            'config': {
                'similarity_threshold': self.similarity_threshold,
                'resolution': self.resolution,
                'randomize': self.randomize
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
        self.communities = defaultdict(list, index_data['communities'])
        self.document_to_community = index_data['document_to_community']
        self.community_summaries = index_data['community_summaries']
        self.community_keywords = index_data['community_keywords']
        self.graph = nx.node_link_graph(index_data['graph_data'])
        self.community_graph = nx.node_link_graph(index_data['community_graph_data'])
        self.modularity_score = index_data['modularity_score']
        self.num_communities = index_data['num_communities']
        
        # 重新生成文档嵌入向量
        for doc_id, doc_info in self.documents.items():
            embedding = self.embedding_model.encode([doc_info['text']])[0]
            self.document_embeddings[doc_id] = embedding