"""
Graph Utilities
图工具类
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional, Any
import json
from collections import defaultdict


class GraphUtils:
    """
    图操作工具类
    """
    
    @staticmethod
    def merge_graphs(graphs: List[nx.Graph], 
                    merge_strategy: str = 'union') -> nx.Graph:
        """
        合并多个图
        
        Args:
            graphs: 图列表
            merge_strategy: 合并策略 ('union', 'intersection')
            
        Returns:
            合并后的图
        """
        if not graphs:
            return nx.Graph()
        
        if merge_strategy == 'union':
            merged = nx.Graph()
            for graph in graphs:
                merged = nx.union(merged, graph)
            return merged
        
        elif merge_strategy == 'intersection':
            merged = graphs[0].copy()
            for graph in graphs[1:]:
                merged = nx.intersection(merged, graph)
            return merged
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    @staticmethod
    def calculate_centrality_measures(graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        计算图的中心性指标
        
        Args:
            graph: 输入图
            
        Returns:
            中心性指标字典
        """
        measures = {}
        
        # 度中心性
        measures['degree_centrality'] = nx.degree_centrality(graph)
        
        # 特征向量中心性
        try:
            measures['eigenvector_centrality'] = nx.eigenvector_centrality(graph)
        except:
            measures['eigenvector_centrality'] = {}
        
        # 接近中心性
        if nx.is_connected(graph):
            measures['closeness_centrality'] = nx.closeness_centrality(graph)
        else:
            measures['closeness_centrality'] = {}
        
        # 介数中心性
        measures['betweenness_centrality'] = nx.betweenness_centrality(graph)
        
        # PageRank
        measures['pagerank'] = nx.pagerank(graph)
        
        return measures
    
    @staticmethod
    def find_communities_networkx(graph: nx.Graph, 
                                 method: str = 'greedy_modularity') -> Dict[int, List[str]]:
        """
        使用NetworkX内置方法发现社区
        
        Args:
            graph: 输入图
            method: 社区发现方法
            
        Returns:
            社区字典
        """
        communities = {}
        
        if method == 'greedy_modularity':
            community_generator = nx.community.greedy_modularity_communities(graph)
            for i, community in enumerate(community_generator):
                communities[i] = list(community)
        
        elif method == 'label_propagation':
            community_generator = nx.community.label_propagation_communities(graph)
            for i, community in enumerate(community_generator):
                communities[i] = list(community)
        
        else:
            raise ValueError(f"Unknown community detection method: {method}")
        
        return communities
    
    @staticmethod
    def calculate_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
        """
        计算图的基本指标
        
        Args:
            graph: 输入图
            
        Returns:
            图指标字典
        """
        metrics = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_connected': nx.is_connected(graph),
            'num_connected_components': nx.number_connected_components(graph)
        }
        
        if graph.number_of_nodes() > 0:
            # 度分布统计
            degrees = [d for n, d in graph.degree()]
            metrics['degree_stats'] = {
                'mean': np.mean(degrees),
                'std': np.std(degrees),
                'min': min(degrees),
                'max': max(degrees)
            }
            
            # 聚类系数
            metrics['avg_clustering'] = nx.average_clustering(graph)
            
            # 如果图连通，计算直径和平均路径长度
            if nx.is_connected(graph):
                metrics['diameter'] = nx.diameter(graph)
                metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(graph)
        
        return metrics
    
    @staticmethod
    def filter_graph_by_degree(graph: nx.Graph, 
                              min_degree: int = 1,
                              max_degree: Optional[int] = None) -> nx.Graph:
        """
        根据度数过滤图节点
        
        Args:
            graph: 输入图
            min_degree: 最小度数
            max_degree: 最大度数
            
        Returns:
            过滤后的图
        """
        filtered_graph = graph.copy()
        
        # 获取需要移除的节点
        nodes_to_remove = []
        for node, degree in graph.degree():
            if degree < min_degree:
                nodes_to_remove.append(node)
            elif max_degree is not None and degree > max_degree:
                nodes_to_remove.append(node)
        
        # 移除节点
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        return filtered_graph
    
    @staticmethod
    def extract_subgraph(graph: nx.Graph, 
                        nodes: List[str], 
                        include_neighbors: bool = False,
                        neighbor_hops: int = 1) -> nx.Graph:
        """
        提取子图
        
        Args:
            graph: 输入图
            nodes: 节点列表
            include_neighbors: 是否包含邻居节点
            neighbor_hops: 邻居跳数
            
        Returns:
            子图
        """
        if include_neighbors:
            # 包含邻居节点
            extended_nodes = set(nodes)
            current_nodes = set(nodes)
            
            for hop in range(neighbor_hops):
                next_nodes = set()
                for node in current_nodes:
                    if node in graph:
                        next_nodes.update(graph.neighbors(node))
                extended_nodes.update(next_nodes)
                current_nodes = next_nodes
            
            subgraph_nodes = list(extended_nodes)
        else:
            subgraph_nodes = nodes
        
        # 只包含存在于原图中的节点
        valid_nodes = [n for n in subgraph_nodes if n in graph]
        
        return graph.subgraph(valid_nodes).copy()
    
    @staticmethod
    def find_shortest_paths(graph: nx.Graph, 
                           source: str, 
                           targets: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        找到最短路径
        
        Args:
            graph: 输入图
            source: 源节点
            targets: 目标节点列表，如果为None则计算到所有节点的路径
            
        Returns:
            最短路径字典
        """
        paths = {}
        
        if targets is None:
            targets = list(graph.nodes())
        
        for target in targets:
            if target != source and nx.has_path(graph, source, target):
                try:
                    path = nx.shortest_path(graph, source, target)
                    paths[target] = path
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    @staticmethod
    def detect_bridges_and_articulation_points(graph: nx.Graph) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        检测桥和关节点
        
        Args:
            graph: 输入图
            
        Returns:
            (桥列表, 关节点列表)
        """
        bridges = list(nx.bridges(graph))
        articulation_points = list(nx.articulation_points(graph))
        
        return bridges, articulation_points
    
    @staticmethod
    def graph_to_adjacency_matrix(graph: nx.Graph, 
                                 node_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        将图转换为邻接矩阵
        
        Args:
            graph: 输入图
            node_order: 节点顺序
            
        Returns:
            (邻接矩阵, 节点列表)
        """
        if node_order is None:
            node_order = list(graph.nodes())
        
        adj_matrix = nx.adjacency_matrix(graph, nodelist=node_order).toarray()
        
        return adj_matrix, node_order
    
    @staticmethod
    def save_graph_formats(graph: nx.Graph, 
                          filepath_base: str, 
                          formats: List[str] = ['gml', 'graphml', 'json']) -> None:
        """
        以多种格式保存图
        
        Args:
            graph: 输入图
            filepath_base: 文件路径基础名（不含扩展名）
            formats: 保存格式列表
        """
        for fmt in formats:
            filepath = f"{filepath_base}.{fmt}"
            
            if fmt == 'gml':
                nx.write_gml(graph, filepath)
            elif fmt == 'graphml':
                nx.write_graphml(graph, filepath)
            elif fmt == 'json':
                graph_data = nx.node_link_data(graph)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
            elif fmt == 'edgelist':
                nx.write_edgelist(graph, filepath)
            else:
                print(f"Warning: Unknown format {fmt}")
    
    @staticmethod
    def load_graph_from_format(filepath: str, 
                              fmt: Optional[str] = None) -> nx.Graph:
        """
        从文件加载图
        
        Args:
            filepath: 文件路径
            fmt: 文件格式，如果为None则根据扩展名推断
            
        Returns:
            图对象
        """
        if fmt is None:
            fmt = filepath.split('.')[-1].lower()
        
        if fmt == 'gml':
            return nx.read_gml(filepath)
        elif fmt == 'graphml':
            return nx.read_graphml(filepath)
        elif fmt == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            return nx.node_link_graph(graph_data)
        elif fmt == 'edgelist':
            return nx.read_edgelist(filepath)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    
    @staticmethod
    def compare_graphs(graph1: nx.Graph, graph2: nx.Graph) -> Dict[str, Any]:
        """
        比较两个图的相似性
        
        Args:
            graph1: 第一个图
            graph2: 第二个图
            
        Returns:
            比较结果字典
        """
        # 节点集合比较
        nodes1 = set(graph1.nodes())
        nodes2 = set(graph2.nodes())
        
        node_intersection = nodes1.intersection(nodes2)
        node_union = nodes1.union(nodes2)
        
        # 边集合比较
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())
        
        edge_intersection = edges1.intersection(edges2)
        edge_union = edges1.union(edges2)
        
        comparison = {
            'node_jaccard': len(node_intersection) / len(node_union) if node_union else 0,
            'edge_jaccard': len(edge_intersection) / len(edge_union) if edge_union else 0,
            'common_nodes': len(node_intersection),
            'common_edges': len(edge_intersection),
            'total_unique_nodes': len(node_union),
            'total_unique_edges': len(edge_union),
            'graph1_only_nodes': len(nodes1 - nodes2),
            'graph2_only_nodes': len(nodes2 - nodes1),
            'graph1_only_edges': len(edges1 - edges2),
            'graph2_only_edges': len(edges2 - edges1)
        }
        
        return comparison