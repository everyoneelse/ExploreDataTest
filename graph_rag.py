import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import json
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import faiss


class KnowledgeGraph:
    """知识图谱构建和管理"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self.entity_to_id = {}
        self.id_to_entity = {}
        
    def add_triple(self, subject: str, relation: str, obj: str):
        """添加三元组到知识图谱"""
        # 添加节点
        if subject not in self.entity_to_id:
            entity_id = len(self.entity_to_id)
            self.entity_to_id[subject] = entity_id
            self.id_to_entity[entity_id] = subject
            self.graph.add_node(entity_id, name=subject)
            
        if obj not in self.entity_to_id:
            entity_id = len(self.entity_to_id)
            self.entity_to_id[obj] = entity_id
            self.id_to_entity[entity_id] = obj
            self.graph.add_node(entity_id, name=obj)
        
        # 添加边
        subj_id = self.entity_to_id[subject]
        obj_id = self.entity_to_id[obj]
        self.graph.add_edge(subj_id, obj_id, relation=relation)
    
    def get_neighbors(self, entity: str, max_hops: int = 2) -> List[Tuple[str, str, str]]:
        """获取实体的邻居节点"""
        if entity not in self.entity_to_id:
            return []
        
        entity_id = self.entity_to_id[entity]
        neighbors = []
        
        # 获取直接邻居
        for neighbor_id in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor_id)
            relation = edge_data.get('relation', 'unknown')
            neighbor_name = self.id_to_entity[neighbor_id]
            neighbors.append((entity, relation, neighbor_name))
        
        # 获取反向邻居
        for predecessor_id in self.graph.predecessors(entity_id):
            edge_data = self.graph.get_edge_data(predecessor_id, entity_id)
            relation = edge_data.get('relation', 'unknown')
            predecessor_name = self.id_to_entity[predecessor_id]
            neighbors.append((predecessor_name, relation, entity))
        
        return neighbors
    
    def get_subgraph(self, entities: List[str], max_hops: int = 2) -> List[Tuple[str, str, str]]:
        """获取包含指定实体的子图"""
        subgraph_triples = set()
        
        for entity in entities:
            neighbors = self.get_neighbors(entity, max_hops)
            for triple in neighbors:
                subgraph_triples.add(triple)
        
        return list(subgraph_triples)


class GraphRAG:
    """GraphRAG主类，结合知识图谱和检索增强生成"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 kg_path: Optional[str] = None):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.knowledge_graph = KnowledgeGraph()
        self.document_store = []
        self.document_embeddings = None
        self.faiss_index = None
        
        if kg_path and os.path.exists(kg_path):
            self.load_knowledge_graph(kg_path)
    
    def build_knowledge_graph(self, data: List[Dict[str, Any]]):
        """从数据构建知识图谱"""
        print("构建知识图谱...")
        
        for item in data:
            subject = item.get('subject', '').strip()
            relation = item.get('relation', '').strip()
            obj = item.get('object', '').strip()
            
            if subject and relation and obj:
                self.knowledge_graph.add_triple(subject, relation, obj)
            
            # 同时将问题-答案对作为文档存储
            document = {
                'id': item['id'],
                'text': f"Question: {item['question']} Answer: {item['answer']}",
                'question': item['question'],
                'answer': item['answer'],
                'subject': subject,
                'relation': relation,
                'object': obj
            }
            self.document_store.append(document)
        
        print(f"知识图谱构建完成: {len(self.knowledge_graph.entity_to_id)} 个实体")
        self._build_embeddings()
    
    def _build_embeddings(self):
        """构建文档嵌入和FAISS索引"""
        print("构建文档嵌入...")
        
        texts = [doc['text'] for doc in self.document_store]
        self.document_embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # 构建FAISS索引
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # 归一化嵌入向量
        normalized_embeddings = self.document_embeddings / np.linalg.norm(
            self.document_embeddings, axis=1, keepdims=True
        )
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        print(f"文档嵌入构建完成: {len(texts)} 个文档")
    
    def extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体（简单实现）"""
        # 这里使用简单的关键词匹配，实际应用中可以使用NER模型
        entities = []
        text_lower = text.lower()
        
        for entity in self.knowledge_graph.entity_to_id.keys():
            if entity.lower() in text_lower:
                entities.append(entity)
        
        return entities
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if self.faiss_index is None:
            return []
        
        # 编码查询
        query_embedding = self.encoder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # 搜索最相似的文档
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.document_store):
                doc = self.document_store[idx].copy()
                doc['similarity_score'] = float(score)
                doc['rank'] = i + 1
                relevant_docs.append(doc)
        
        return relevant_docs
    
    def get_graph_context(self, query: str, max_triples: int = 10) -> List[Tuple[str, str, str]]:
        """获取与查询相关的图上下文"""
        # 提取查询中的实体
        entities = self.extract_entities(query)
        
        if not entities:
            return []
        
        # 获取相关的子图
        subgraph_triples = self.knowledge_graph.get_subgraph(entities, max_hops=2)
        
        # 按相关性排序（这里简单按照实体在查询中的出现顺序）
        scored_triples = []
        for triple in subgraph_triples:
            score = 0
            for entity in entities:
                if entity.lower() in ' '.join(triple).lower():
                    score += 1
            scored_triples.append((score, triple))
        
        # 排序并返回前N个
        scored_triples.sort(key=lambda x: x[0], reverse=True)
        return [triple for _, triple in scored_triples[:max_triples]]
    
    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """使用GraphRAG生成答案"""
        # 1. 检索相关文档
        relevant_docs = self.retrieve_relevant_documents(query, top_k)
        
        # 2. 获取图上下文
        graph_context = self.get_graph_context(query, max_triples=10)
        
        # 3. 构建上下文
        context_parts = []
        
        # 添加文档上下文
        if relevant_docs:
            context_parts.append("相关文档:")
            for doc in relevant_docs:
                context_parts.append(f"- {doc['text']} (相似度: {doc['similarity_score']:.3f})")
        
        # 添加图上下文
        if graph_context:
            context_parts.append("\n相关知识图谱信息:")
            for subj, rel, obj in graph_context:
                context_parts.append(f"- {subj} {rel} {obj}")
        
        context = "\n".join(context_parts)
        
        # 4. 简单的答案生成（基于最相似文档的答案）
        predicted_answer = ""
        confidence = 0.0
        
        if relevant_docs:
            # 使用最相似文档的答案
            best_doc = relevant_docs[0]
            predicted_answer = best_doc['answer']
            confidence = best_doc['similarity_score']
        
        return {
            'query': query,
            'predicted_answer': predicted_answer,
            'confidence': confidence,
            'context': context,
            'relevant_documents': relevant_docs,
            'graph_context': graph_context
        }
    
    def save_knowledge_graph(self, path: str):
        """保存知识图谱"""
        data = {
            'graph': self.knowledge_graph.graph,
            'entity_to_id': self.knowledge_graph.entity_to_id,
            'id_to_entity': self.knowledge_graph.id_to_entity,
            'document_store': self.document_store
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        # 保存嵌入
        if self.document_embeddings is not None:
            np.save(path.replace('.pkl', '_embeddings.npy'), self.document_embeddings)
        
        print(f"知识图谱已保存到 {path}")
    
    def load_knowledge_graph(self, path: str):
        """加载知识图谱"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.knowledge_graph.graph = data['graph']
            self.knowledge_graph.entity_to_id = data['entity_to_id']
            self.knowledge_graph.id_to_entity = data['id_to_entity']
            self.document_store = data['document_store']
            
            # 加载嵌入
            embedding_path = path.replace('.pkl', '_embeddings.npy')
            if os.path.exists(embedding_path):
                self.document_embeddings = np.load(embedding_path)
                
                # 重建FAISS索引
                dimension = self.document_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                normalized_embeddings = self.document_embeddings / np.linalg.norm(
                    self.document_embeddings, axis=1, keepdims=True
                )
                self.faiss_index.add(normalized_embeddings.astype('float32'))
            
            print(f"知识图谱已从 {path} 加载")
        except Exception as e:
            print(f"加载知识图谱失败: {e}")


if __name__ == "__main__":
    # 测试GraphRAG
    from data_loader import PopQADataLoader
    
    # 加载数据
    loader = PopQADataLoader()
    data = loader.get_sample_data(20)
    
    # 创建GraphRAG实例
    graph_rag = GraphRAG()
    
    # 构建知识图谱
    graph_rag.build_knowledge_graph(data)
    
    # 测试查询
    test_queries = [
        "Who is the president of the United States?",
        "What is the capital of France?",
        "Who wrote 1984?"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = graph_rag.generate_answer(query)
        print(f"预测答案: {result['predicted_answer']}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"图上下文: {len(result['graph_context'])} 个三元组")
        print("---")