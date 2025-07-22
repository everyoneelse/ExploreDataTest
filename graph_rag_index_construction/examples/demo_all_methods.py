#!/usr/bin/env python3
"""
Demo script for all Graph-based RAG Index Construction methods
演示所有Graph-based RAG索引构建方法的脚本
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from node_index import EntityNodeIndex, DocumentNodeIndex, ConceptNodeIndex, HierarchicalNodeIndex
from relationship_index import SemanticRelationshipIndex
from community_index import LouvainCommunityIndex
import json
import time


def create_sample_documents():
    """
    创建示例文档数据
    """
    documents = [
        {
            'id': 'doc1',
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. Deep learning uses neural networks with multiple layers to model complex patterns.',
            'metadata': {'category': 'AI', 'author': 'Alice'}
        },
        {
            'id': 'doc2', 
            'text': 'Natural language processing enables computers to understand human language. Transformers have revolutionized NLP with attention mechanisms and pre-trained models like BERT and GPT.',
            'metadata': {'category': 'NLP', 'author': 'Bob'}
        },
        {
            'id': 'doc3',
            'text': 'Computer vision allows machines to interpret visual information. Convolutional neural networks are particularly effective for image recognition and object detection tasks.',
            'metadata': {'category': 'CV', 'author': 'Charlie'}
        },
        {
            'id': 'doc4',
            'text': 'Reinforcement learning trains agents to make decisions through trial and error. Q-learning and policy gradient methods are popular approaches in this field.',
            'metadata': {'category': 'RL', 'author': 'Diana'}
        },
        {
            'id': 'doc5',
            'text': 'Data science combines statistics, programming, and domain expertise to extract insights from data. Python and R are commonly used languages for data analysis.',
            'metadata': {'category': 'DS', 'author': 'Eve'}
        },
        {
            'id': 'doc6',
            'text': 'Graph neural networks extend deep learning to graph-structured data. They can capture relationships between entities and are useful for social network analysis.',
            'metadata': {'category': 'GNN', 'author': 'Frank'}
        },
        {
            'id': 'doc7',
            'text': 'Explainable AI aims to make machine learning models more interpretable. SHAP and LIME are popular techniques for understanding model predictions.',
            'metadata': {'category': 'XAI', 'author': 'Grace'}
        },
        {
            'id': 'doc8',
            'text': 'Edge computing brings computation closer to data sources. This reduces latency and bandwidth usage, making it ideal for IoT applications.',
            'metadata': {'category': 'Edge', 'author': 'Henry'}
        }
    ]
    return documents


def demo_entity_node_index(documents):
    """
    演示实体节点索引
    """
    print("\n" + "="*50)
    print("ENTITY NODE INDEX DEMO")
    print("="*50)
    
    # 初始化索引
    entity_index = EntityNodeIndex()
    
    # 构建索引
    start_time = time.time()
    entity_index.build_index(documents)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.2f} seconds")
    
    # 查询实体
    query = "neural networks"
    print(f"\nQuerying entities for: '{query}'")
    similar_entities = entity_index.find_similar_entities(query, top_k=3)
    
    for entity, similarity in similar_entities:
        print(f"  - {entity}: {similarity:.3f}")
    
    # 获取图统计信息
    stats = entity_index.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  - Entities: {stats.get('num_entities', 0)}")
    print(f"  - Documents: {stats.get('num_documents', 0)}")
    print(f"  - Graph edges: {stats.get('num_edges', 0)}")


def demo_document_node_index(documents):
    """
    演示文档节点索引
    """
    print("\n" + "="*50)
    print("DOCUMENT NODE INDEX DEMO")
    print("="*50)
    
    # 初始化索引
    doc_index = DocumentNodeIndex()
    
    # 构建索引
    start_time = time.time()
    doc_index.build_index(documents)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.2f} seconds")
    
    # 查询相似文档
    query = "deep learning algorithms"
    print(f"\nQuerying documents for: '{query}'")
    similar_docs = doc_index.get_similar_documents(query, top_k=3)
    
    for doc_id, similarity in similar_docs:
        print(f"  - {doc_id}: {similarity:.3f}")
        print(f"    Text: {documents[int(doc_id.replace('doc', ''))-1]['text'][:80]}...")
    
    # 获取图统计信息
    stats = doc_index.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  - Documents: {stats.get('num_documents', 0)}")
    print(f"  - Graph edges: {stats.get('num_edges', 0)}")
    print(f"  - Density: {stats.get('density', 0):.3f}")


def demo_concept_node_index(documents):
    """
    演示概念节点索引
    """
    print("\n" + "="*50)
    print("CONCEPT NODE INDEX DEMO")
    print("="*50)
    
    # 初始化索引
    concept_index = ConceptNodeIndex()
    
    # 构建索引
    start_time = time.time()
    concept_index.build_index(documents)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.2f} seconds")
    
    # 查询概念
    query = "machine learning"
    print(f"\nQuerying concepts for: '{query}'")
    similar_concepts = concept_index.find_concepts_by_query(query, top_k=3)
    
    for concept_id, similarity in similar_concepts:
        concept_phrase = concept_index.concepts[concept_id]['phrase']
        print(f"  - {concept_phrase}: {similarity:.3f}")
    
    # 获取图统计信息
    stats = concept_index.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  - Total concepts: {stats.get('num_concepts', 0)}")
    print(f"  - Filtered concepts: {stats.get('num_filtered_concepts', 0)}")
    print(f"  - Graph edges: {stats.get('num_edges', 0)}")


def demo_hierarchical_node_index(documents):
    """
    演示分层节点索引
    """
    print("\n" + "="*50)
    print("HIERARCHICAL NODE INDEX DEMO")
    print("="*50)
    
    # 初始化索引
    hierarchical_index = HierarchicalNodeIndex(max_levels=3, min_cluster_size=2)
    
    # 构建索引
    start_time = time.time()
    hierarchical_index.build_index(documents)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.2f} seconds")
    
    # 查询层次结构
    query = "artificial intelligence"
    print(f"\nQuerying hierarchy for: '{query}'")
    results = hierarchical_index.query_hierarchy(query, top_k=3)
    
    for node_id, similarity, level in results:
        node_meta = hierarchical_index.node_metadata[node_id]
        print(f"  - Level {level}, {node_id}: {similarity:.3f}")
        print(f"    Summary: {node_meta.get('summary', '')[:80]}...")
    
    # 获取图统计信息
    stats = hierarchical_index.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  - Levels: {stats.get('num_levels', 0)}")
    print(f"  - Total nodes: {stats.get('total_nodes', 0)}")
    print(f"  - Total edges: {stats.get('total_edges', 0)}")


def demo_semantic_relationship_index(documents):
    """
    演示语义关系索引
    """
    print("\n" + "="*50)
    print("SEMANTIC RELATIONSHIP INDEX DEMO")
    print("="*50)
    
    # 初始化索引
    semantic_index = SemanticRelationshipIndex()
    
    # 构建索引
    start_time = time.time()
    semantic_index.build_index(documents)
    build_time = time.time() - start_time
    
    print(f"Build time: {build_time:.2f} seconds")
    
    # 查询语义关系
    query = "neural networks"
    print(f"\nQuerying semantic relationships for: '{query}'")
    results = semantic_index.query_semantic_relationships(query, top_k=3)
    
    for doc_id, similarity, relation_info in results:
        print(f"  - {doc_id}: {similarity:.3f}")
        print(f"    {relation_info}")
    
    # 获取图统计信息
    stats = semantic_index.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  - Documents: {stats.get('num_documents', 0)}")
    print(f"  - Relationships: {stats.get('num_relationships', 0)}")
    print(f"  - Avg confidence: {stats.get('avg_confidence', 0):.3f}")


def demo_louvain_community_index(documents):
    """
    演示Louvain社区索引
    """
    print("\n" + "="*50)
    print("LOUVAIN COMMUNITY INDEX DEMO")
    print("="*50)
    
    try:
        # 初始化索引
        community_index = LouvainCommunityIndex()
        
        # 构建索引
        start_time = time.time()
        community_index.build_index(documents)
        build_time = time.time() - start_time
        
        print(f"Build time: {build_time:.2f} seconds")
        
        # 查询社区
        query = "machine learning"
        print(f"\nQuerying communities for: '{query}'")
        communities = community_index.query_communities(query, top_k=3)
        
        for community_id, similarity, summary in communities:
            print(f"  - Community {community_id}: {similarity:.3f}")
            print(f"    Summary: {summary[:100]}...")
            
            # 显示社区中的文档
            docs = community_index.get_community_documents(community_id)
            print(f"    Documents: {docs}")
        
        # 获取图统计信息
        stats = community_index.get_graph_statistics()
        print(f"\nGraph Statistics:")
        print(f"  - Documents: {stats.get('num_documents', 0)}")
        print(f"  - Communities: {stats.get('num_communities', 0)}")
        print(f"  - Modularity: {stats.get('modularity_score', 0):.3f}")
        
    except ImportError as e:
        print(f"Louvain community detection not available: {e}")
        print("Please install python-louvain: pip install python-louvain")


def main():
    """
    主函数，运行所有演示
    """
    print("Graph-based RAG Index Construction Methods Demo")
    print("=" * 60)
    
    # 创建示例文档
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # 演示各种索引方法
    try:
        demo_entity_node_index(documents)
    except Exception as e:
        print(f"Entity Node Index demo failed: {e}")
    
    try:
        demo_document_node_index(documents)
    except Exception as e:
        print(f"Document Node Index demo failed: {e}")
    
    try:
        demo_concept_node_index(documents)
    except Exception as e:
        print(f"Concept Node Index demo failed: {e}")
    
    try:
        demo_hierarchical_node_index(documents)
    except Exception as e:
        print(f"Hierarchical Node Index demo failed: {e}")
    
    try:
        demo_semantic_relationship_index(documents)
    except Exception as e:
        print(f"Semantic Relationship Index demo failed: {e}")
    
    try:
        demo_louvain_community_index(documents)
    except Exception as e:
        print(f"Louvain Community Index demo failed: {e}")
    
    print("\n" + "="*60)
    print("Demo completed!")


if __name__ == "__main__":
    main()