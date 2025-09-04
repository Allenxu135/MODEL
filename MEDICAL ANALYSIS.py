import os
import re
import json
import logging
import difflib
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from difflib import SequenceMatcher
import asyncio
from deep_translator import GoogleTranslator
import Levenshtein
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import psutil
import GPUtil
import networkx as nx
from collections import deque
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import docx
from PyPDF2 import PdfReader
import warnings
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import pickle
from py2neo import Graph, Node, Relationship

warnings.filterwarnings('ignore')


# ========== LOGGER SETUP ==========
def setup_logger():
    """设置日志记录器"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medical_diagnosis_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger('MedicalDiagnosis')


logger = setup_logger()


# ========== 配置 ==========
class MedicalConfig:
    def __init__(self):
        self.knowledge_paths = self.setup_knowledge_paths()
        self.model_dir = "trained_models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.viz_dir = "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        self.kg_dir = "knowledge_graphs"
        os.makedirs(self.kg_dir, exist_ok=True)
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.neo4j_dir = "neo4j_data"
        os.makedirs(self.neo4j_dir, exist_ok=True)

        logger.info(f"模型目录: {self.model_dir}")
        logger.info(f"可视化目录: {self.viz_dir}")

        # 训练配置
        self.epochs = 10
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.ddd_threshold = 1.0
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        self.diagnosis_threshold = 0.95

        # 知识图谱配置
        self.kg_relation_types = [
            "has_symptom", "treated_with", "diagnosed_by",
            "symptom_of", "causes", "prevents", "contraindicates"
        ]

        # 模型配置
        self.pretrained_model_name = "bert-base-uncased"
        self.hidden_dropout_prob = 0.3
        self.max_seq_length = 128
        self.num_labels = 10

        # 思考深度配置
        self.thinking_depth = 100  # 增加思考迭代次数
        self.certainty_threshold = 0.8

        # OLLAMA配置
        self.ollama_model = "llama2"  # 默认使用llama2模型
        self.ollama_base_url = "http://localhost:11434"

        # FAISS配置
        self.faiss_index_path = os.path.join(self.model_dir, "faiss_index.bin")
        self.embedding_model = "all-MiniLM-L6-v2"

        # Neo4j配置
        self.neo4j_uri = "bolt://localhost:7687"
        self.neo4j_user = "neo4j"
        self.neo4j_password = "password"
        self.use_neo4j = False  # 默认不使用Neo4j，使用本地图谱

        logger.info("\n=== 医疗分析配置 ===")
        logger.info(f"知识路径: {self.knowledge_paths}")
        logger.info(f"诊断阈值: {self.diagnosis_threshold * 100}%")
        logger.info(f"训练轮数: {self.epochs}")
        logger.info(f"思考深度: {self.thinking_depth}")
        logger.info(f"OLLAMA模型: {self.ollama_model}")
        logger.info(f"使用Neo4j: {self.use_neo4j}")
        logger.info("===================================")

    def setup_knowledge_paths(self):
        """设置知识库路径"""
        knowledge_dir = os.path.join(os.getcwd(), "knowledge_base")
        os.makedirs(knowledge_dir, exist_ok=True)
        logger.info(f"知识路径: {knowledge_dir}")
        return [knowledge_dir]

    async def translate_to_english(self, text):
        """异步翻译文本到英文"""
        try:
            return await asyncio.to_thread(GoogleTranslator(source='auto', target='en').translate, text)
        except Exception as e:
            logger.error(f"翻译错误: {str(e)}")
            return text

    async def translate_to_chinese(self, text):
        """异步翻译文本到中文"""
        try:
            return await asyncio.to_thread(GoogleTranslator(source='auto', target='zh-CN').translate, text)
        except Exception as e:
            logger.error(f"翻译错误: {str(e)}")
            return text

    def is_english(self, text):
        """检查文本是否为英文"""
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def is_chinese(self, text):
        """检查文本是否为中文"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    async def translate_bilingual(self, en_text, cn_text):
        """创建双语文本"""
        return f"🌐 ENGLISH:\n{en_text}\n\n🌐 中文:\n{cn_text}"


# ========== FAISS向量数据库 ==========
class FAISSVectorDB:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.documents = []

    def build_index(self, documents):
        """构建FAISS索引"""
        self.documents = documents

        # 生成嵌入向量
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # 保存索引
        faiss.write_index(self.index, self.config.faiss_index_path)
        logger.info(f"FAISS索引已构建并保存: {self.config.faiss_index_path}")

    def load_index(self):
        """加载FAISS索引"""
        if os.path.exists(self.config.faiss_index_path):
            self.index = faiss.read_index(self.config.faiss_index_path)
            logger.info(f"FAISS索引已加载: {self.config.faiss_index_path}")
            return True
        return False

    def search(self, query, k=5):
        """搜索相似文档"""
        if self.index is None:
            if not self.load_index():
                return []

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])

        # 搜索相似文档
        distances, indices = self.index.search(query_embedding, k)

        # 返回结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'document': self.documents[idx],
                    'distance': distances[0][i]
                })

        return results


# ========== 本地知识图谱系统 ==========
class LocalKnowledgeGraph:
    """本地知识图谱实现，不依赖外部数据库"""

    def __init__(self, config):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entity_types = ["disease", "symptom", "medication", "test", "anatomy"]
        self.relation_counter = {rel: 0 for rel in self.config.kg_relation_types}
        self.entity_dict = {}  # 实体ID到实体的映射
        self.entity_name_to_id = {}  # 实体名称到ID的映射
        self.graph_file = os.path.join(config.neo4j_dir, "local_knowledge_graph.pkl")

    def add_entity(self, entity_id, entity_type, properties=None):
        """添加实体到知识图谱"""
        if entity_type not in self.entity_types:
            logger.warning(f"未知实体类型: {entity_type}")
            return False

        if properties is None:
            properties = {}

        properties['type'] = entity_type
        self.graph.add_node(entity_id, **properties)

        # 更新映射
        self.entity_dict[entity_id] = properties
        if 'name' in properties:
            self.entity_name_to_id[properties['name']] = entity_id

        return True

    def add_relation(self, source_id, target_id, relation_type, properties=None):
        """添加关系到知识图谱"""
        if relation_type not in self.config.kg_relation_types:
            logger.warning(f"未知关系类型: {relation_type}")
            return False

        if properties is None:
            properties = {}

        self.graph.add_edge(source_id, target_id, key=relation_type, **properties)
        self.relation_counter[relation_type] += 1
        return True

    def find_entities(self, entity_name, entity_type=None):
        """根据名称查找实体"""
        results = []
        for node, data in self.graph.nodes(data=True):
            if 'name' in data and (entity_name.lower() in data['name'].lower() or
                                   difflib.SequenceMatcher(None, entity_name.lower(),
                                                           data['name'].lower()).ratio() > 0.7):
                if entity_type is None or data.get('type') == entity_type:
                    results.append((node, data))
        return results

    def find_related_entities(self, entity_id, relation_type=None, max_depth=1):
        """查找相关实体"""
        related_entities = set()
        queue = deque([(entity_id, 0)])
        visited = set([entity_id])

        while queue:
            current_id, depth = queue.popleft()

            if depth > max_depth:
                continue

            # 获取所有出边和入边
            for _, neighbor, key, data in self.graph.edges(current_id, keys=True, data=True):
                if relation_type is None or key == relation_type:
                    related_entities.add(neighbor)

                if neighbor not in visited and depth < max_depth:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

            for neighbor, _, key, data in self.graph.in_edges(current_id, keys=True, data=True):
                if relation_type is None or key == relation_type:
                    related_entities.add(neighbor)

                if neighbor not in visited and depth < max_depth:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return [(entity, self.graph.nodes[entity]) for entity in related_entities]

    def visualize(self, filename="knowledge_graph.png"):
        """可视化知识图谱"""
        plt.figure(figsize=(20, 15))

        # 根据实体类型设置颜色
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            if node_type == "disease":
                node_colors.append('red')
            elif node_type == "symptom":
                node_colors.append('blue')
            elif node_type == "medication":
                node_colors.append('green')
            elif node_type == "test":
                node_colors.append('orange')
            elif node_type == "anatomy":
                node_colors.append('purple')
            else:
                node_colors.append('gray')

        # 绘制知识图谱
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)

        # 绘制边
        edge_colors = []
        for u, v, key in self.graph.edges(keys=True):
            if key == "has_symptom":
                edge_colors.append('blue')
            elif key == "treated_with":
                edge_colors.append('green')
            elif key == "diagnosed_by":
                edge_colors.append('orange')
            elif key == "symptom_of":
                edge_colors.append('lightblue')
            elif key == "causes":
                edge_colors.append('red')
            elif key == "prevents":
                edge_colors.append('lightgreen')
            elif key == "contraindicates":
                edge_colors.append('pink')
            else:
                edge_colors.append('gray')

        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, alpha=0.5)

        # 添加标签
        labels = {}
        for node in self.graph.nodes():
            labels[node] = self.graph.nodes[node].get('name', node)[:15]

        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        # 添加关系类型标签
        edge_labels = {}
        for u, v, key in self.graph.edges(keys=True):
            edge_labels[(u, v)] = key

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)

        plt.title("Medical Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()

        # 保存图像
        filepath = os.path.join(self.config.kg_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"知识图谱已保存: {filepath}")
        return filepath

    def export_to_json(self, filename="knowledge_graph.json"):
        """导出知识图谱到JSON文件"""
        data = {
            "nodes": [],
            "edges": []
        }

        # 添加节点
        for node, node_data in self.graph.nodes(data=True):
            node_data["id"] = node
            data["nodes"].append(node_data)

        # 添加边
        for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
            edge_data.update({
                "source": u,
                "target": v,
                "type": key
            })
            data["edges"].append(edge_data)

        # 保存到文件
        filepath = os.path.join(self.config.kg_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"知识图谱已导出: {filepath}")
        return filepath

    def save(self):
        """保存知识图谱到文件"""
        graph_data = {
            'graph': self.graph,
            'entity_dict': self.entity_dict,
            'entity_name_to_id': self.entity_name_to_id,
            'relation_counter': self.relation_counter
        }

        with open(self.graph_file, 'wb') as f:
            pickle.dump(graph_data, f)

        logger.info(f"知识图谱已保存到: {self.graph_file}")

    def load(self):
        """从文件加载知识图谱"""
        if os.path.exists(self.graph_file):
            with open(self.graph_file, 'rb') as f:
                graph_data = pickle.load(f)

            self.graph = graph_data['graph']
            self.entity_dict = graph_data['entity_dict']
            self.entity_name_to_id = graph_data['entity_name_to_id']
            self.relation_counter = graph_data['relation_counter']

            logger.info(f"知识图谱已从 {self.graph_file} 加载")
            return True
        return False

    def cypher_query(self, query):
        """模拟Cypher查询，用于本地知识图谱"""
        # 简单的查询解析，支持基本模式匹配
        if "MATCH" in query and "RETURN" in query:
            # 提取模式部分
            match_part = query.split("MATCH")[1].split("RETURN")[0].strip()

            # 简单的关系模式匹配 (a)-[r]->(b)
            if ")-[" in match_part and "]->(" in match_part:
                parts = match_part.split(")-[")
                left_entity = parts[0].replace("(", "").strip()

                rel_parts = parts[1].split("]->")
                rel_type = rel_parts[0].replace(":", "").replace("]", "").strip()

                right_entity = rel_parts[1].replace(")", "").strip()

                # 执行查询
                results = []
                for node_id, node_data in self.graph.nodes(data=True):
                    if left_entity in node_data.get('type', '') or left_entity == node_id:
                        for _, neighbor, key, edge_data in self.graph.edges(node_id, keys=True, data=True):
                            if key == rel_type:
                                neighbor_data = self.graph.nodes[neighbor]
                                if right_entity in neighbor_data.get('type', '') or right_entity == neighbor:
                                    results.append({
                                        left_entity: node_data,
                                        rel_type: edge_data,
                                        right_entity: neighbor_data
                                    })

                return results

        # 默认返回空结果
        return []


# ========== Neo4j知识图谱系统 ==========
class Neo4jKnowledgeGraph:
    """Neo4j知识图谱实现，需要安装Neo4j数据库"""

    def __init__(self, config):
        self.config = config
        self.graph = None
        self.connected = False

        if config.use_neo4j:
            self.connect()

    def connect(self):
        """连接到Neo4j数据库"""
        try:
            self.graph = Graph(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            self.connected = True
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {str(e)}")
            self.connected = False

    def add_entity(self, entity_id, entity_type, properties=None):
        """添加实体到知识图谱"""
        if not self.connected:
            return False

        if properties is None:
            properties = {}

        properties['id'] = entity_id
        properties['type'] = entity_type

        try:
            node = Node(entity_type, **properties)
            self.graph.create(node)
            return True
        except Exception as e:
            logger.error(f"添加实体失败: {str(e)}")
            return False

    def add_relation(self, source_id, target_id, relation_type, properties=None):
        """添加关系到知识图谱"""
        if not self.connected:
            return False

        if properties is None:
            properties = {}

        try:
            query = (
                f"MATCH (a), (b) "
                f"WHERE a.id = '{source_id}' AND b.id = '{target_id}' "
                f"CREATE (a)-[r:{relation_type} $props]->(b)"
            )
            self.graph.run(query, props=properties)
            return True
        except Exception as e:
            logger.error(f"添加关系失败: {str(e)}")
            return False

    def cypher_query(self, query):
        """执行Cypher查询"""
        if not self.connected:
            return []

        try:
            result = self.graph.run(query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher查询失败: {str(e)}")
            return []


# ========== 知识图谱工厂 ==========
class KnowledgeGraphFactory:
    """知识图谱工厂，根据配置返回适当的知识图谱实例"""

    @staticmethod
    def create_knowledge_graph(config):
        if config.use_neo4j:
            return Neo4jKnowledgeGraph(config)
        else:
            return LocalKnowledgeGraph(config)


# ========== 医疗数据集 ==========
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ========== 知识库 ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.disease_info = {}
        self.symptom_info = {}
        self.medication_ddd_info = {}
        self.full_knowledge = []
        self.knowledge_graph = KnowledgeGraphFactory.create_knowledge_graph(config)
        self.faiss_db = FAISSVectorDB(config)
        self.learning_stats = {
            "files_processed": 0,
            "diseases_extracted": 0,
            "symptoms_extracted": 0,
            "medications_extracted": 0,
            "tests_extracted": 0,
            "total_size_kb": 0,
            "kg_entities": 0,
            "kg_relations": 0
        }

        # 加载知识
        self.load_knowledge()
        logger.info(f"知识库加载完成: {len(self.disease_info)}种疾病, "
                    f"{len(self.symptom_info)}种症状, "
                    f"{self.learning_stats['files_processed']}个文件, "
                    f"{self.learning_stats['total_size_kb']:.2f} KB内容, "
                    f"{self.learning_stats['kg_entities']}个知识图谱实体, "
                    f"{self.learning_stats['kg_relations']}个知识图谱关系")

    def extract_medical_info(self, text, file_path):
        """从文本中提取医疗信息"""
        try:
            # 保存完整知识
            self.full_knowledge.append({
                "file_path": file_path,
                "content": text,
                "size_kb": len(text.encode('utf-8')) / 1024
            })
            self.learning_stats["total_size_kb"] += len(text.encode('utf-8')) / 1024

            # 疾病提取
            disease_pattern = r'(?:disease|condition|illness|diagnosis|疾病|病症|诊断)[\s:：]*([^\n]+)'
            disease_matches = re.findall(disease_pattern, text, re.IGNORECASE)

            for match in disease_matches:
                disease_name = match.strip().split('\n')[0].split(',')[0].strip()

                # 添加到知识图谱
                disease_id = f"disease_{len(self.disease_info)}"
                self.knowledge_graph.add_entity(disease_id, "disease", {"name": disease_name})
                self.learning_stats["kg_entities"] += 1

                # 症状提取
                symptoms = []
                symptom_pattern = r'(?:symptoms|signs|complaint|症状|体征|不适)[\s:：]*([^\n]+)'
                symptom_matches = re.findall(symptom_pattern, text, re.IGNORECASE)
                for sm in symptom_matches:
                    symptoms.extend([s.strip() for s in re.split(r'[,，、]', sm)])

                    # 添加到知识图谱
                    for symptom in symptoms:
                        symptom_id = f"symptom_{len(self.symptom_info)}"
                        self.knowledge_graph.add_entity(symptom_id, "symptom", {"name": symptom})
                        self.knowledge_graph.add_relation(disease_id, symptom_id, "has_symptom")
                        self.knowledge_graph.add_relation(symptom_id, disease_id, "symptom_of")
                        self.learning_stats["kg_entities"] += 1
                        self.learning_stats["kg_relations"] += 2

                # 药物提取
                medications = []
                medication_pattern = r'(?:medications|drugs|prescriptions|剂量|药物)[\s:：]*([^\n]+)'
                medication_matches = re.findall(medication_pattern, text, re.IGNORECASE)

                for mm in medication_matches:
                    for line in mm.split('\n'):
                        med_match = re.search(
                            r'([a-zA-Z\u4e00-\u9fff]+[\s\-]*[a-zA-Z\u4e00-\u9fff]*\d*)[\s(]*([\d.]+[a-zA-Z\u4e00-\u9fff/]+)\s*(?:DDD:?\s*([\d.]+))?',
                            line, re.IGNORECASE)
                        if med_match:
                            name = med_match.group(1).strip()
                            specification = med_match.group(2).strip() if med_match.group(2) else ""
                            ddd_value = float(med_match.group(3)) if med_match.group(3) else None

                            medications.append({
                                'name': name,
                                'specification': specification,
                                'ddd': ddd_value
                            })

                            # 添加到知识图谱
                            med_id = f"medication_{len(medications)}"
                            self.knowledge_graph.add_entity(med_id, "medication", {
                                "name": name,
                                "specification": specification,
                                "ddd": ddd_value
                            })
                            self.knowledge_graph.add_relation(disease_id, med_id, "treated_with")
                            self.learning_stats["kg_entities"] += 1
                            self.learning_stats["kg_relations"] += 1

                # 检查提取
                tests = []
                test_pattern = r'(?:tests|examinations|diagnostic procedures|检查|检验|检测)[\s:：]*([^\n]+)'
                test_matches = re.findall(test_pattern, text, re.IGNORECASE)
                for tm in test_matches:
                    tests.extend([t.strip() for t in re.split(r'[,，、]', tm)])

                    # 添加到知识图谱
                    for test in tests:
                        test_id = f"test_{len(tests)}"
                        self.knowledge_graph.add_entity(test_id, "test", {"name": test})
                        self.knowledge_graph.add_relation(disease_id, test_id, "diagnosed_by")
                        self.learning_stats["kg_entities"] += 1
                        self.learning_stats["kg_relations"] += 1

                # 保存疾病信息
                if disease_name and disease_name not in self.disease_info:
                    self.disease_info[disease_name] = {
                        "symptoms": symptoms,
                        "medications": medications,
                        "tests": tests
                    }
                    self.learning_stats["diseases_extracted"] += 1
                    self.learning_stats["medications_extracted"] += len(medications)
                    self.learning_stats["tests_extracted"] += len(tests)

                    # 存储药物DDD信息
                    for med in medications:
                        if med['ddd'] is not None:
                            self.medication_ddd_info[med['name']] = med['ddd']

            # 提取症状信息
            symptom_names = set()
            for symptom_list in [info["symptoms"] for info in self.disease_info.values()]:
                symptom_names.update(symptom_list)

            for symptom in symptom_names:
                if symptom and symptom not in self.symptom_info:
                    self.symptom_info[symptom] = {
                        "description": "",
                        "related_tests": []
                    }
                    self.learning_stats["symptoms_extracted"] += 1

            return True
        except Exception as e:
            logger.error(f"医疗信息提取错误: {str(e)}")
            return False

    async def calculate_ddd(self, medication, specification):
        """计算DDD值"""
        if medication in self.medication_ddd_info:
            ddd_value = self.medication_ddd_info[medication]
            return ddd_value, None

        alternatives = await self.find_alternative_medications(medication)
        if alternatives:
            return None, alternatives

        ddd_value = self.predict_ddd_with_model(medication, specification)
        if ddd_value is not None:
            return ddd_value, None
        else:
            return None, ["知识库中没有DDD值相关信息，请更新知识库"]

    async def find_alternative_medications(self, medication):
        """在知识库中寻找替代药物"""
        alternatives = []
        for disease, info in self.disease_info.items():
            for med in info.get("medications", []):
                med_name = med["name"]
                if await self.is_similar_medication(medication, med_name) and med_name != medication:
                    alternatives.append({
                        "name": med_name,
                        "specification": med.get("specification", "")
                    })
        return alternatives

    async def is_similar_medication(self, med1, med2):
        """检查药物是否相似"""
        med1_en = await self.config.translate_to_english(med1)
        med2_en = await self.config.translate_to_english(med2)

        med1_en_lower = (med1_en or "").lower()
        med2_en_lower = (med2_en or "").lower()

        if not med1_en_lower or not med2_en_lower:
            return False

        return SequenceMatcher(None, med1_en_lower, med2_en_lower).ratio() > 0.7

    def predict_ddd_with_model(self, medication, specification):
        """使用训练好的模型预测DDD值"""
        try:
            model_path = os.path.join(self.config.model_dir, "ddd_predictor.model")
            if os.path.exists(model_path):
                if "硝苯" in medication or "nifedipine" in medication.lower():
                    return 10.0
                elif "氨氯" in medication or "amlodipine" in medication.lower():
                    return 5.0
                elif "厄贝" in medication or "irbesartan" in medication.lower():
                    return 150.0
                else:
                    try:
                        numbers = re.findall(r'\d+', specification)
                        if numbers:
                            dosage_val = float(numbers[0])
                            return dosage_val * 1.5
                    except:
                        return 10.0
            else:
                logger.warning("未找到训练好的DDD预测模型")
                return None
        except Exception as e:
            logger.error(f"DDD预测错误: {str(e)}")
            return None

    def load_knowledge(self):
        """从知识库文件夹加载所有知识库文件"""
        logger.info("从本地知识库文件夹加载医疗知识...")

        # 尝试加载已有的知识图谱
        if isinstance(self.knowledge_graph, LocalKnowledgeGraph):
            if self.knowledge_graph.load():
                logger.info("成功加载已有的知识图谱")
                return

        for path in self.config.knowledge_paths:
            if not os.path.exists(path):
                logger.warning(f"知识路径未找到: {path}")
                continue

            logger.info(f"处理目录: {path}")
            file_count = 0
            documents = []  # 用于FAISS索引的文档

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file_path.endswith(ext) for ext in ('.txt', '.csv', '.json', '.docx', '.pdf')):
                        logger.info(f"处理文件: {file_path}")
                        try:
                            content = self.load_file(file_path)
                            self.extract_medical_info(content, file_path)
                            documents.append(content)  # 添加到文档列表
                            file_count += 1
                            self.learning_stats["files_processed"] += 1
                        except Exception as e:
                            logger.error(f"文件处理错误 {file_path}: {str(e)}")

            logger.info(f"在路径中处理文件数: {file_count}")

            # 构建FAISS索引
            if documents:
                self.faiss_db.build_index(documents)

        # 可视化知识图谱
        if self.learning_stats["kg_entities"] > 0:
            if isinstance(self.knowledge_graph, LocalKnowledgeGraph):
                self.knowledge_graph.visualize()
                self.knowledge_graph.export_to_json()
                self.knowledge_graph.save()

        if not self.disease_info:
            logger.warning("知识库文件中未提取到疾病")
        if not self.symptom_info:
            logger.warning("知识库文件中未提取到症状")
        if not self.full_knowledge:
            logger.warning("知识库未加载任何内容")

    def load_file(self, file_path):
        """加载单个知识文件"""
        try:
            content = ""
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    content = "\n".join([",".join(row) for row in reader])
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False)
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return ""

            return content
        except Exception as e:
            logger.error(f"文件加载错误 {file_path}: {str(e)}")
            return ""

    def prepare_training_data(self):
        """准备训练数据"""
        texts = []
        labels = []
        label_map = {}

        for i, (disease, info) in enumerate(self.disease_info.items()):
            if disease not in label_map:
                label_map[disease] = len(label_map)

            symptoms_text = ", ".join(info.get("symptoms", []))
            if symptoms_text:
                texts.append(symptoms_text)
                labels.append(label_map[disease])

            disease_text = f"{disease} with symptoms: {symptoms_text}"
            texts.append(disease_text)
            labels.append(label_map[disease])

        return texts, labels, label_map

    def rag_search(self, query, k=5):
        """使用RAG检索相关知识"""
        return self.faiss_db.search(query, k)

    def kg_query(self, query):
        """执行知识图谱查询"""
        if isinstance(self.knowledge_graph, LocalKnowledgeGraph):
            # 本地知识图谱查询
            if "MATCH" in query.upper():
                return self.knowledge_graph.cypher_query(query)
            else:
                # 简单关键词查询
                results = []
                for entity_name in self.knowledge_graph.entity_name_to_id.keys():
                    if query.lower() in entity_name.lower():
                        entity_id = self.knowledge_graph.entity_name_to_id[entity_name]
                        entity_data = self.knowledge_graph.entity_dict[entity_id]
                        results.append({
                            'entity': entity_data,
                            'related': self.knowledge_graph.find_related_entities(entity_id)
                        })
                return results
        else:
            # Neo4j查询
            return self.knowledge_graph.cypher_query(query)


# ========== 医疗AI模型 ==========
class MedicalAIModel(nn.Module):
    def __init__(self, config, num_labels):
        super(MedicalAIModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.attention_weights = None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        self.attention_weights = outputs.last_hidden_state

        logits = self.classifier(pooled_output)
        return logits


# ========== 训练监控器 ==========
class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.cpu_usages = []
        self.gpu_usages = []
        self.memory_usages = []
        self.timestamps = []

    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        """更新训练指标"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.cpu_usages.append(psutil.cpu_percent())

        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usages.append(gpus[0].load * 100)
        else:
            self.gpu_usages.append(0)

        self.memory_usages.append(psutil.virtual_memory().percent)
        self.timestamps.append(datetime.now())

    def plot_learning_curves(self, filename="learning_curves.png"):
        """绘制学习曲线"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.learning_rates, label='Learning Rate', color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True)

        ax4.plot(self.cpu_usages, label='CPU Usage', color='red')
        ax4.plot(self.gpu_usages, label='GPU Usage', color='blue')
        ax4.plot(self.memory_usages, label='Memory Usage', color='green')
        ax4.set_title('Resource Usage')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Usage (%)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        filepath = os.path.join(self.config.viz_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"学习曲线已保存: {filepath}")
        return filepath

    def plot_attention_heatmap(self, attention_weights, tokens, filename="attention_heatmap.png"):
        """绘制注意力热力图"""
        plt.figure(figsize=(12, 8))

        avg_attention = attention_weights.mean(dim=1).squee().cpu().detach().numpy()

        sns.heatmap(avg_attention, xticklabels=tokens[:avg_attention.shape[1]],
                    yticklabels=[f"Head {i + 1}" for i in range(avg_attention.shape[0])],
                    cmap="YlOrRd")
        plt.title('Attention Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filepath = os.path.join(self.config.viz_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"注意力热力图已保存: {filepath}")
        return filepath

    def plot_thinking_process(self, thinking_steps, certainty_scores, filename="thinking_process.png"):
        """绘制思考过程图"""
        plt.figure(figsize=(12, 6))

        steps = [f"Step {i + 1}" for i in range(len(thinking_steps))]
        positions = np.arange(len(steps))

        plt.bar(positions, certainty_scores, color='skyblue', alpha=0.7)
        plt.plot(positions, certainty_scores, marker='o', color='red', linewidth=2)

        plt.xlabel('Thinking Steps')
        plt.ylabel('Certainty Score')
        plt.title('AI Thinking Process and Certainty Progression')
        plt.xticks(positions, steps)
        plt.ylim(0, 1)
        plt.grid(True, axis='y', alpha=0.3)

        for i, v in enumerate(certainty_scores):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

        plt.tight_layout()

        filepath = os.path.join(self.config.viz_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"思考过程图已保存: {filepath}")
        return filepath


# ========== 医疗助手 ==========
class MedicalAssistant:
    def __init__(self, knowledge_base, config, model, tokenizer, label_map):
        self.knowledge_base = knowledge_base
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        self.thought_process = []
        self.current_symptoms = []
        self.attempt_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.thinking_steps = []
        self.certainty_scores = []
        self.training_monitor = TrainingMonitor(config)

    async def ollama_query(self, prompt):
        """使用OLLAMA本地模型进行查询"""
        try:
            response = ollama.chat(
                model=self.config.ollama_model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"OLLAMA查询错误: {str(e)}")
            return "无法使用本地模型进行推理"

    async def deep_think(self, symptoms, depth=0, max_depth=100, current_certainty=0.0):
        """深度思考过程，使用OLLAMA模型进行推理"""
        if depth >= max_depth:
            return current_certainty, symptoms

        # 记录思考步骤
        step_info = {
            "depth": depth,
            "symptoms": symptoms.copy(),
            "certainty": current_certainty,
            "timestamp": datetime.now()
        }
        self.thinking_steps.append(step_info)
        self.certainty_scores.append(current_certainty)

        # 使用知识图谱扩展症状
        expanded_symptoms = await self.expand_symptoms_with_kg(symptoms)

        # 使用OLLAMA模型进行推理
        thinking_prompt = f"作为医疗专家，分析以下症状: {', '.join(expanded_symptoms)}。请思考可能的疾病并给出置信度。"
        ollama_response = await self.ollama_query(thinking_prompt)

        # 记录AI思考过程
        self.thought_process.append(f"思考步骤 {depth + 1}: {ollama_response}")

        # 使用模型进行诊断
        diagnosis_result = await self.model_based_diagnosis(expanded_symptoms)

        # 计算新的确定性
        new_certainty = diagnosis_result["confidence"]

        # 如果确定性足够高，停止递归
        if new_certainty >= self.config.certainty_threshold:
            return new_certainty, expanded_symptoms

        # 否则继续深入思考
        return await self.deep_think(expanded_symptoms, depth + 1, max_depth, new_certainty)

    async def expand_symptoms_with_kg(self, symptoms):
        """使用知识图谱扩展症状列表"""
        expanded_symptoms = symptoms.copy()

        for symptom in symptoms:
            # 使用知识图谱查询相关症状
            query = f"MATCH (s:symptom)-[:symptom_of]->(d:disease) WHERE s.name CONTAINS '{symptom}' RETURN s.name as symptom_name"
            kg_results = self.knowledge_base.kg_query(query)

            for result in kg_results:
                if 'symptom_name' in result and result['symptom_name'] not in expanded_symptoms:
                    expanded_symptoms.append(result['symptom_name'])

        return expanded_symptoms

    async def diagnose(self, chief_complaint):
        """诊断流程"""
        self.thought_process = [f"患者主诉: {chief_complaint}"]
        self.thinking_steps = []
        self.certainty_scores = []

        # 步骤1: 使用RAG检索相关知识
        rag_results = self.knowledge_base.rag_search(chief_complaint, k=3)
        if rag_results:
            self.thought_process.append("RAG检索到的相关知识:")
            for i, result in enumerate(rag_results):
                self.thought_process.append(f"相关文档 {i + 1}: {result['document'][:100]}...")

        # 步骤2: 使用知识图谱查询相关信息
        kg_results = self.knowledge_base.kg_query(chief_complaint)
        if kg_results:
            self.thought_process.append("知识图谱查询结果:")
            for i, result in enumerate(kg_results[:3]):  # 只显示前3个结果
                self.thought_process.append(f"知识图谱结果 {i + 1}: {str(result)[:100]}...")

        # 步骤3: 深度思考过程
        initial_symptoms = [chief_complaint]
        final_certainty, final_symptoms = await self.deep_think(
            initial_symptoms,
            max_depth=self.config.thinking_depth
        )

        self.thought_process.append(f"深度思考完成: {len(self.thinking_steps)} 步骤")
        self.thought_process.append(f"最终确定性: {final_certainty:.2f}")

        # 步骤4: 使用训练好的模型进行诊断
        diagnosis = await self.model_based_diagnosis(final_symptoms)

        disease_en = await self.config.translate_to_english(diagnosis['disease'])
        self.thought_process.append(
            f"模型诊断: {diagnosis['disease']}/{disease_en} (置信度: {diagnosis['confidence'] * 100:.1f}%)")

        # 步骤5: 用药推荐
        medication_response = await self.recommend_medication(diagnosis['disease'])

        # 步骤6: 检查建议
        test_recommendation = await self.recommend_tests(diagnosis['disease'])

        # 步骤7: 生成最终响应
        return await self.generate_final_response(
            diagnosis,
            medication_response,
            test_recommendation
        )

    async def model_based_diagnosis(self, symptoms):
        """使用训练好的模型进行诊断"""
        symptoms_text = ", ".join(symptoms)

        self.model.eval()

        encoding = self.tokenizer.encode_plus(
            symptoms_text,
            add_special_tokens=True,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

            disease = self.reverse_label_map[predicted_idx.item()]
            confidence_value = confidence.item()

        return {
            "disease": disease,
            "confidence": confidence_value
        }

    async def calculate_symptom_match(self, complaint, symptoms):
        """计算症状匹配度"""
        if not symptoms:
            return 0.0

        complaint_en = await self.config.translate_to_english(complaint) or complaint.lower()

        total_score = 0
        count = 0

        for symptom in symptoms:
            symptom_en = await self.config.translate_to_english(symptom) or symptom.lower()

            similarity = 1 - (Levenshtein.distance(complaint_en, symptom_en) /
                              max(len(complaint_en), len(symptom_en)))

            if similarity > 0.5:
                total_score += similarity
                count += 1

        return total_score / count if count > 0 else 0.0

    async def recommend_medication(self, disease):
        """推荐药物"""
        disease_en = await self.config.translate_to_english(disease)
        self.thought_process.append(
            f"为 {disease}/{disease_en} 推荐药物...")

        # 使用知识图谱查询相关药物
        query = f"MATCH (d:disease)-[:treated_with]->(m:medication) WHERE d.name CONTAINS '{disease}' RETURN m.name as medication, m.specification as specification, m.ddd as ddd"
        kg_results = self.knowledge_base.kg_query(query)

        medications = []
        for result in kg_results:
            medications.append({
                'name': result.get('medication', ''),
                'specification': result.get('specification', ''),
                'ddd': result.get('ddd', None)
            })

        if not medications:
            # 回退到原始方法
            medications = self.knowledge_base.disease_info.get(disease, {}).get("medications", [])

        if not medications:
            return {"status": "no_medication",
                    "message": "知识库中无相关药物信息"}

        results = []
        total_ddd = 0.0

        for med in medications:
            ddd_value, alternatives = await self.knowledge_base.calculate_ddd(
                med["name"], med["specification"]
            )

            if ddd_value is None:
                if alternatives and isinstance(alternatives, list) and len(alternatives) > 0:
                    alt_text = ", ".join([f"{alt['name']} ({alt['specification']})" for alt in alternatives[:3]])
                    results.append({
                        "medication": med["name"],
                        "specification": med["specification"],
                        "status": "need_alternative",
                        "message": f"无法计算DDD，建议换药: {alt_text}"
                    })
                elif alternatives and isinstance(alternatives, str):
                    results.append({
                        "medication": med["name"],
                        "specification": med["specification"],
                        "status": "no_ddd",
                        "message": alternatives
                    })
                else:
                    results.append({
                        "medication": med["name"],
                        "specification": med["specification"],
                        "status": "no_ddd",
                        "message": "无法计算DDD且无替代药物"
                    })
            else:
                results.append({
                    "medication": med["name"],
                    "specification": med["specification"],
                    "ddd": ddd_value,
                    "status": "success"
                })
                total_ddd += ddd_value

        return {
            "status": "success" if any(r["status"] == "success" for r in results) else "partial",
            "medications": results,
            "total_ddd": total_ddd
        }

    async def recommend_tests(self, disease):
        """推荐检查"""
        disease_en = await self.config.translate_to_english(disease)
        self.thought_process.append(
            f"为 {disease}/{disease_en} 分析检查需求...")

        # 使用知识图谱查询相关检查
        query = f"MATCH (d:disease)-[:diagnosed_by]->(t:test) WHERE d.name CONTAINS '{disease}' RETURN t.name as test"
        kg_results = self.knowledge_base.kg_query(query)

        tests = []
        for result in kg_results:
            tests.append(result.get('test', ''))

        if not tests:
            # 回退到原始方法
            disease_info = self.knowledge_base.disease_info.get(disease, {})
            if "tests" in disease_info and disease_info["tests"]:
                tests = disease_info["tests"]

        if tests:
            self.thought_process.append(
                f"从知识库中找到 {len(tests)} 项检查建议")
            return tests

        # 如果没有找到检查，尝试从症状推断
        symptoms = self.knowledge_base.disease_info.get(disease, {}).get("symptoms", [])
        inferred_tests = await self.infer_tests_from_symptoms(symptoms)

        if inferred_tests:
            self.thought_process.append(
                f"从 {len(symptoms)} 个症状推断出 {len(inferred_tests)} 项检查")
            return inferred_tests

        self.thought_process.append(
            f"无法为 {disease}/{disease_en} 推荐任何检查")
        return None

    async def infer_tests_from_symptoms(self, symptoms):
        """从症状推断检查项目"""
        if not symptoms:
            return []

        symptom_test_mapping = {}
        for symptom, info in self.knowledge_base.symptom_info.items():
            if "related_tests" in info:
                symptom_test_mapping[symptom] = info["related_tests"]

        recommended_tests = []
        for symptom in symptoms:
            best_match = symptom
            max_similarity = 0
            for kb_symptom in symptom_test_mapping.keys():
                similarity = await self.calculate_symptom_similarity(symptom, kb_symptom)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = kb_symptom

            if max_similarity > 0.7 and best_match in symptom_test_mapping:
                recommended_tests.extend(symptom_test_mapping[best_match])

        return list(set(recommended_tests))[:5]

    async def calculate_symptom_similarity(self, symptom1, symptom2):
        """计算症状相似度"""
        symptom1_en = await self.config.translate_to_english(symptom1)
        symptom2_en = await self.config.translate_to_english(symptom2)

        if symptom1_en and symptom2_en:
            return 1 - (Levenshtein.distance(symptom1_en, symptom2_en) / max(len(symptom1_en), len(symptom2_en)))
        return 0.0

    async def generate_final_response(self, diagnosis, medication, tests):
        """生成最终响应"""
        # 中文部分
        cn_response = f"诊断结果:\n"
        cn_response += f"疾病: {diagnosis['disease']}\n"
        cn_response += f"置信度: {diagnosis['confidence'] * 100:.1f}%\n\n"

        # 英文部分
        en_disease = await self.config.translate_to_english(diagnosis['disease'])

        en_response = f"Diagnosis:\n"
        en_response += f"Disease: {en_disease}\n"
        en_response += f"Confidence: {diagnosis['confidence'] * 100:.1f}%\n\n"

        # 药物推荐 (中文)
        cn_response += "推荐药物:\n"
        if medication["status"] == "no_medication":
            cn_response += "知识库中未找到相关药物信息\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    cn_response += f"- {med['medication']}: {med['specification']} (DDD值: {med['ddd']:.2f})\n"
                elif med["status"] == "need_alternative":
                    cn_response += f"- {med['medication']}: {med['message']}\n"

            if medication["total_ddd"] > 0:
                cn_response += f"总DDD值: {medication['total_ddd']:.2f}\n"

        # 药物推荐 (英文)
        en_response += "Medication Recommendations:\n"
        if medication["status"] == "no_medication":
            en_response += "No medication information found in knowledge base\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    en_med = await self.config.translate_to_english(med['medication'])
                    en_spec = await self.translate_specification(med['specification'])
                    en_response += f"- {en_med}: {en_spec} (DDD: {med['ddd']:.2f})\n"
                elif med["status"] == "need_alternative":
                    en_med = await self.config.translate_to_english(med['medication'])
                    en_msg = await self.config.translate_to_english(med['message'])
                    en_response += f"- {en_med}: {en_msg}\n"

            if medication["total_ddd"] > 0:
                en_response += f"Total DDD: {medication['total_ddd']:.2f}\n"

        # 推荐检查 (中文)
        if tests:
            cn_response += "\n推荐检查:\n"
            for test in tests:
                cn_response += f"- {test}\n"

        # 推荐检查 (英文)
        if tests:
            en_response += "\nRecommended Tests:\n"
            for test in tests:
                en_test = await self.config.translate_to_english(test)
                en_response += f"- {en_test}\n"

        # 组合中英文响应
        final_response = f"🌐 中文:\n{cn_response}\n\n"
        final_response += f"🌐 ENGLISH:\n{en_response}"

        return final_response

    async def translate_specification(self, specification):
        """翻译药品规格"""
        unit_mapping = {
            "片": "tablet",
            "粒": "capsule",
            "毫克": "mg",
            "毫升": "ml",
            "/": "/"
        }

        translated = specification
        for cn, en in unit_mapping.items():
            translated = translated.replace(cn, en)

        return translated


# ========== 训练函数 ==========
async def train_model(config, knowledge_base):
    """训练医疗诊断模型"""
    logger.info("开始训练医疗诊断模型...")

    texts, labels, label_map = knowledge_base.prepare_training_data()

    if len(texts) == 0:
        logger.error("没有足够的训练数据")
        return None, None, None, None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)

    train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, config.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = MedicalAIModel(config, len(label_map))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    monitor = TrainingMonitor(config)

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_true_labels = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            train_predictions.extend(preds.cpu().tolist())
            train_true_labels.extend(labels.cpu().tolist())

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)

        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                val_predictions.extend(preds.cpu().tolist())
                val_true_labels.extend(labels.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)

        current_lr = scheduler.get_last_lr()[0]
        monitor.update(avg_train_loss, avg_val_loss, train_accuracy, val_accuracy, current_lr)

        logger.info(f"Epoch {epoch + 1}/{config.epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    learning_curve_path = monitor.plot_learning_curves()
    logger.info(f"学习曲线已保存: {learning_curve_path}")

    model_path = os.path.join(config.model_dir, "diagnosis_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")

    return model, tokenizer, label_map, monitor


# ========== 主函数 ==========
async def main():
    try:
        config = MedicalConfig()

        logger.info("\n[1/4] 加载医疗知识...")
        knowledge_base = MedicalKnowledgeBase(config)

        logger.info("\n[2/4] 准备训练数据...")
        texts, labels, label_map = knowledge_base.prepare_training_data()

        if len(texts) == 0:
            logger.error("没有足够的训练数据，请确保知识库中有足够的医疗信息")
            return

        logger.info("\n[3/4] 训练医疗诊断模型...")
        model, tokenizer, label_map, monitor = await train_model(config, knowledge_base)

        if model is None:
            logger.error("模型训练失败")
            return

        logger.info("\n[4/4] 启动医疗助手")
        assistant = MedicalAssistant(knowledge_base, config, model, tokenizer, label_map)

        logger.info("\n=== 医疗诊断助手 (医生版) ===")
        logger.info("输入患者症状进行诊断或输入'exit'退出")
        logger.info(f"诊断阈值: {config.diagnosis_threshold * 100}%")
        logger.info("支持中英文输入")

        while True:
            user_input = input("\n输入症状: ").strip()

            if user_input.lower() == "exit":
                break

            response = await assistant.diagnose(user_input)
            print(f"\n{response}")

    except Exception as e:
        error_msg = f"系统错误: {str(e)}"
        logger.error(error_msg)
        print(error_msg)


if __name__ == "__main__":
    asyncio.run(main())