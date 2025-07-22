"""
Hierarchical Node Index Implementation
分层节点索引实现
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any, Union
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import json
import hashlib


class HierarchicalNodeIndex:
    """
    分层节点索引
    
    该类实现了分层的节点索引构建方法，
    通过聚类和层次结构组织文档和概念。
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_levels: int = 4,
                 cluster_threshold: float = 0.7,
                 min_cluster_size: int = 2,
                 max_cluster_size: int = 20):
        """
        初始化分层节点索引
        
        Args:
            embedding_model: 句子嵌入模型名称
            max_levels: 最大层级数
            cluster_threshold: 聚类阈值
            min_cluster_size: 最小聚类大小
            max_cluster_size: 最大聚类大小
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_levels = max_levels
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # 存储层次结构信息
        self.hierarchy = {}  # level -> nodes
        self.node_embeddings = {}
        self.node_metadata = {}
        self.parent_child_relations = defaultdict(list)  # parent -> [children]
        self.child_parent_relations = {}  # child -> parent
        self.level_graphs = {}  # level -> graph
        
        # 主图包含所有层级的节点和边
        self.graph = nx.DiGraph()
        
        # 文档到节点的映射
        self.document_to_nodes = defaultdict(list)
        self.node_to_documents = defaultdict(list)
    
    def _generate_node_id(self, content: str, level: int, node_type: str) -> str:
        """
        生成节点ID
        
        Args:
            content: 节点内容
            level: 层级
            node_type: 节点类型
            
        Returns:
            节点ID
        """
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{node_type}_L{level}_{content_hash}"
    
    def add_leaf_nodes(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加叶子节点（最底层文档节点）
        
        Args:
            documents: 文档列表
        """
        level = 0
        self.hierarchy[level] = []
        
        print(f"Adding {len(documents)} leaf nodes at level {level}...")
        
        for doc in documents:
            doc_id = doc['id']
            text = doc['text']
            metadata = doc.get('metadata', {})
            
            # 生成节点ID
            node_id = self._generate_node_id(text, level, 'doc')
            
            # 生成嵌入向量
            embedding = self.embedding_model.encode([text])[0]
            
            # 存储节点信息
            self.node_embeddings[node_id] = embedding
            self.node_metadata[node_id] = {
                'original_id': doc_id,
                'text': text,
                'metadata': metadata,
                'level': level,
                'node_type': 'document',
                'children': [],
                'summary': text[:200] + "..." if len(text) > 200 else text
            }
            
            # 添加到层次结构
            self.hierarchy[level].append(node_id)
            
            # 建立文档映射
            self.document_to_nodes[doc_id].append(node_id)
            self.node_to_documents[node_id].append(doc_id)
            
            # 添加到图
            self.graph.add_node(node_id, **self.node_metadata[node_id])
    
    def _cluster_nodes(self, node_ids: List[str], level: int) -> List[List[str]]:
        """
        对节点进行聚类
        
        Args:
            node_ids: 节点ID列表
            level: 当前层级
            
        Returns:
            聚类结果列表
        """
        if len(node_ids) <= self.min_cluster_size:
            return [node_ids]
        
        # 获取节点嵌入向量
        embeddings = [self.node_embeddings[node_id] for node_id in node_ids]
        embeddings_array = np.array(embeddings)
        
        # 计算最优聚类数量
        max_clusters = min(len(node_ids) // self.min_cluster_size, 
                          len(node_ids) // 2)
        max_clusters = max(2, max_clusters)
        
        # 使用层次聚类
        try:
            clustering = AgglomerativeClustering(
                n_clusters=max_clusters,
                metric='cosine',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(embeddings_array)
        except:
            # 如果层次聚类失败，使用K-means
            clustering = KMeans(n_clusters=max_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(embeddings_array)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(node_ids[i])
        
        # 过滤太小的聚类
        valid_clusters = []
        for cluster_nodes in clusters.values():
            if len(cluster_nodes) >= self.min_cluster_size:
                valid_clusters.append(cluster_nodes)
            else:
                # 将小聚类合并到最近的大聚类中
                if valid_clusters:
                    # 找到最相似的聚类
                    best_cluster_idx = 0
                    best_similarity = -1
                    
                    for idx, valid_cluster in enumerate(valid_clusters):
                        # 计算聚类间相似度
                        cluster_emb = np.mean([self.node_embeddings[nid] for nid in cluster_nodes], axis=0)
                        valid_emb = np.mean([self.node_embeddings[nid] for nid in valid_cluster], axis=0)
                        similarity = cosine_similarity([cluster_emb], [valid_emb])[0][0]
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_cluster_idx = idx
                    
                    valid_clusters[best_cluster_idx].extend(cluster_nodes)
                else:
                    valid_clusters.append(cluster_nodes)
        
        return valid_clusters
    
    def _create_parent_node(self, children_ids: List[str], level: int) -> str:
        """
        创建父节点
        
        Args:
            children_ids: 子节点ID列表
            level: 层级
            
        Returns:
            父节点ID
        """
        # 计算父节点嵌入（子节点嵌入的平均值）
        children_embeddings = [self.node_embeddings[child_id] for child_id in children_ids]
        parent_embedding = np.mean(children_embeddings, axis=0)
        
        # 生成父节点摘要
        children_summaries = []
        for child_id in children_ids:
            child_meta = self.node_metadata[child_id]
            summary = child_meta.get('summary', child_meta.get('text', ''))[:100]
            children_summaries.append(summary)
        
        combined_summary = " | ".join(children_summaries)
        parent_summary = f"Cluster of {len(children_ids)} items: {combined_summary[:300]}..."
        
        # 生成父节点ID
        parent_id = self._generate_node_id(combined_summary, level, 'cluster')
        
        # 存储父节点信息
        self.node_embeddings[parent_id] = parent_embedding
        self.node_metadata[parent_id] = {
            'level': level,
            'node_type': 'cluster',
            'children': children_ids,
            'summary': parent_summary,
            'cluster_size': len(children_ids),
            'child_types': [self.node_metadata[child]['node_type'] for child in children_ids]
        }
        
        # 建立父子关系
        self.parent_child_relations[parent_id] = children_ids
        for child_id in children_ids:
            self.child_parent_relations[child_id] = parent_id
        
        # 继承文档映射
        for child_id in children_ids:
            if child_id in self.node_to_documents:
                self.node_to_documents[parent_id].extend(self.node_to_documents[child_id])
        
        # 添加到图
        self.graph.add_node(parent_id, **self.node_metadata[parent_id])
        
        # 添加父子边
        for child_id in children_ids:
            self.graph.add_edge(parent_id, child_id, 
                              relation_type='parent_child',
                              weight=1.0)
        
        return parent_id
    
    def build_hierarchy(self) -> None:
        """
        构建层次结构
        """
        print("Building hierarchical structure...")
        
        current_level = 0
        
        while current_level < self.max_levels - 1:
            current_nodes = self.hierarchy.get(current_level, [])
            
            if len(current_nodes) <= self.min_cluster_size:
                print(f"Stopping at level {current_level}: too few nodes ({len(current_nodes)})")
                break
            
            print(f"Processing level {current_level} with {len(current_nodes)} nodes...")
            
            # 对当前层级的节点进行聚类
            clusters = self._cluster_nodes(current_nodes, current_level)
            
            if len(clusters) == 1 and len(clusters[0]) == len(current_nodes):
                print(f"Stopping at level {current_level}: no meaningful clustering")
                break
            
            # 创建下一层级
            next_level = current_level + 1
            self.hierarchy[next_level] = []
            
            # 为每个聚类创建父节点
            for cluster_nodes in clusters:
                if len(cluster_nodes) >= self.min_cluster_size:
                    parent_id = self._create_parent_node(cluster_nodes, next_level)
                    self.hierarchy[next_level].append(parent_id)
            
            print(f"Created level {next_level} with {len(self.hierarchy[next_level])} parent nodes")
            
            current_level = next_level
        
        # 创建层级内的连接
        self._build_level_connections()
    
    def _build_level_connections(self) -> None:
        """
        构建层级内的连接
        """
        print("Building connections within each level...")
        
        for level, nodes in self.hierarchy.items():
            if len(nodes) <= 1:
                continue
                
            # 创建层级图
            level_graph = nx.Graph()
            self.level_graphs[level] = level_graph
            
            # 添加节点
            for node_id in nodes:
                level_graph.add_node(node_id, **self.node_metadata[node_id])
            
            # 计算节点间相似度并添加边
            for i, node_id1 in enumerate(nodes):
                for node_id2 in nodes[i+1:]:
                    emb1 = self.node_embeddings[node_id1]
                    emb2 = self.node_embeddings[node_id2]
                    
                    similarity = cosine_similarity([emb1], [emb2])[0][0]
                    
                    if similarity >= self.cluster_threshold:
                        level_graph.add_edge(node_id1, node_id2,
                                           weight=similarity,
                                           relation_type='similarity')
                        
                        # 也添加到主图
                        self.graph.add_edge(node_id1, node_id2,
                                          weight=similarity,
                                          relation_type='similarity')
    
    def query_hierarchy(self, query: str, top_k: int = 5, 
                       start_level: Optional[int] = None) -> List[Tuple[str, float, int]]:
        """
        在层次结构中查询
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            start_level: 起始查询层级
            
        Returns:
            查询结果列表 (node_id, similarity, level)
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = []
        
        # 确定查询层级
        levels_to_search = []
        if start_level is not None:
            levels_to_search = [start_level]
        else:
            levels_to_search = list(self.hierarchy.keys())
        
        for level in levels_to_search:
            nodes = self.hierarchy.get(level, [])
            
            for node_id in nodes:
                node_embedding = self.node_embeddings[node_id]
                similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                results.append((node_id, similarity, level))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_node_path(self, node_id: str) -> List[str]:
        """
        获取节点的路径（从根到叶子）
        
        Args:
            node_id: 节点ID
            
        Returns:
            路径节点ID列表
        """
        path = []
        current_node = node_id
        
        # 向上遍历到根节点
        while current_node:
            path.insert(0, current_node)
            current_node = self.child_parent_relations.get(current_node)
        
        return path
    
    def get_subtree(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        获取节点的子树
        
        Args:
            node_id: 根节点ID
            max_depth: 最大深度
            
        Returns:
            子树结构
        """
        def _build_subtree(node, depth):
            if depth >= max_depth:
                return None
            
            node_info = {
                'node_id': node,
                'metadata': self.node_metadata.get(node, {}),
                'children': []
            }
            
            children = self.parent_child_relations.get(node, [])
            for child in children:
                child_subtree = _build_subtree(child, depth + 1)
                if child_subtree:
                    node_info['children'].append(child_subtree)
            
            return node_info
        
        return _build_subtree(node_id, 0)
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        构建分层索引
        
        Args:
            documents: 文档列表
        """
        print(f"Building hierarchical index for {len(documents)} documents...")
        
        # 添加叶子节点
        self.add_leaf_nodes(documents)
        
        # 构建层次结构
        self.build_hierarchy()
        
        print(f"Hierarchical index built successfully!")
        print(f"- Total levels: {len(self.hierarchy)}")
        for level, nodes in self.hierarchy.items():
            print(f"  Level {level}: {len(nodes)} nodes")
        print(f"- Total nodes: {self.graph.number_of_nodes()}")
        print(f"- Total edges: {self.graph.number_of_edges()}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Returns:
            图统计信息字典
        """
        stats = {
            'num_levels': len(self.hierarchy),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'level_distribution': {
                level: len(nodes) for level, nodes in self.hierarchy.items()
            }
        }
        
        # 计算每层的统计信息
        for level, nodes in self.hierarchy.items():
            if nodes:
                level_graph = self.level_graphs.get(level)
                if level_graph:
                    stats[f'level_{level}_stats'] = {
                        'nodes': len(nodes),
                        'edges': level_graph.number_of_edges(),
                        'density': nx.density(level_graph),
                        'connected_components': nx.number_connected_components(level_graph)
                    }
        
        return stats
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        index_data = {
            'hierarchy': self.hierarchy,
            'node_metadata': self.node_metadata,
            'parent_child_relations': dict(self.parent_child_relations),
            'child_parent_relations': self.child_parent_relations,
            'document_to_nodes': dict(self.document_to_nodes),
            'node_to_documents': dict(self.node_to_documents),
            'graph_data': nx.node_link_data(self.graph),
            'config': {
                'max_levels': self.max_levels,
                'cluster_threshold': self.cluster_threshold,
                'min_cluster_size': self.min_cluster_size,
                'max_cluster_size': self.max_cluster_size
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
        
        self.hierarchy = index_data['hierarchy']
        self.node_metadata = index_data['node_metadata']
        self.parent_child_relations = defaultdict(list, index_data['parent_child_relations'])
        self.child_parent_relations = index_data['child_parent_relations']
        self.document_to_nodes = defaultdict(list, index_data['document_to_nodes'])
        self.node_to_documents = defaultdict(list, index_data['node_to_documents'])
        self.graph = nx.node_link_graph(index_data['graph_data'])
        
        # 重新生成嵌入向量
        all_node_ids = []
        all_texts = []
        
        for node_id, metadata in self.node_metadata.items():
            all_node_ids.append(node_id)
            if metadata['node_type'] == 'document':
                all_texts.append(metadata['text'])
            else:
                all_texts.append(metadata['summary'])
        
        if all_texts:
            embeddings = self.embedding_model.encode(all_texts)
            for i, node_id in enumerate(all_node_ids):
                self.node_embeddings[node_id] = embeddings[i]
        
        # 重建层级图
        self._build_level_connections()