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
import glob
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

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
        # 创建必要的目录
        self.model_dir = "trained_models"
        self.viz_dir = "charts"  # 改为charts目录
        self.kg_dir = "knowledge_graphs"
        self.data_dir = "data"
        self.neo4j_dir = "neo4j_data"
        self.knowledge_base_dir = "knowledge_base"

        for dir_path in [self.model_dir, self.viz_dir, self.kg_dir, self.data_dir,
                         self.neo4j_dir, self.knowledge_base_dir]:
            os.makedirs(dir_path, exist_ok=True)

        logger.info(f"模型目录: {self.model_dir}")
        logger.info(f"图表目录: {self.viz_dir}")

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
        self.thinking_depth = 100
        self.certainty_threshold = 0.8

        # OLLAMA配置
        self.ollama_model = self.detect_ollama_models()
        self.ollama_base_url = "http://localhost:11434"

        # FAISS配置
        self.faiss_index_path = os.path.join(self.model_dir, "faiss_index.bin")
        self.embedding_model = "all-MiniLM-L6-v2"

        # 知识图谱构建方式
        self.kg_build_method = "auto"  # "auto" 或 "import"

        logger.info("\n=== 医疗分析配置 ===")
        logger.info(f"图表目录: {self.viz_dir}")
        logger.info(f"诊断阈值: {self.diagnosis_threshold * 100}%")
        logger.info(f"训练轮数: {self.epochs}")
        logger.info(f"思考深度: {self.thinking_depth}")
        logger.info(f"OLLAMA模型: {self.ollama_model}")
        logger.info(f"知识图谱构建方式: {self.kg_build_method}")
        logger.info("===================================")

    def detect_ollama_models(self):
        """检测本地可用的OLLAMA模型"""
        try:
            # 获取可用模型列表
            models = ollama.list()
            if models and 'models' in models:
                available_models = [model['name'] for model in models['models']]
                logger.info(f"检测到可用OLLAMA模型: {available_models}")

                # 优先选择医疗相关模型
                medical_models = [model for model in available_models
                                  if any(keyword in model.lower() for keyword in
                                         ['med', 'health', 'bio', 'science'])]

                if medical_models:
                    return medical_models[0]

                # 如果没有医疗相关模型，选择较大的通用模型
                if available_models:
                    # 优先选择较大的模型
                    model_sizes = {
                        'llama2': 1, 'codellama': 2, 'mistral': 3,
                        'mixtral': 4, 'phi': 5, 'gemma': 6
                    }

                    sorted_models = sorted(available_models,
                                           key=lambda x: model_sizes.get(x.split(':')[0], 0),
                                           reverse=True)
                    return sorted_models[0]

            # 默认模型
            return "llama2"
        except Exception as e:
            logger.error(f"检测OLLAMA模型失败: {str(e)}")
            return "llama2"

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


# ========== 文件处理工具 ==========
class FileProcessor:
    """处理各种文件格式的工具类"""

    @staticmethod
    def extract_text_from_file(file_path):
        """从各种文件格式中提取文本"""
        try:
            text = ""
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

            elif ext == '.csv':
                df = pd.read_csv(file_path)
                text = df.to_string()

            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, ensure_ascii=False)

            elif ext == '.docx':
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])

            elif ext == '.pdf':
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"

            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # OCR处理图片
                text = pytesseract.image_to_string(Image.open(file_path))

            elif ext in ['.html', '.htm']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text()

            elif ext in ['.xml']:
                tree = ET.parse(file_path)
                root = tree.getroot()
                text = ET.tostring(root, encoding='unicode', method='text')

            else:
                # 尝试作为文本文件读取
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except:
                    logger.warning(f"不支持的文件格式: {file_path}")
                    return ""

            return text

        except Exception as e:
            logger.error(f"文件处理错误 {file_path}: {str(e)}")
            return ""


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
    """本地知识图谱实现"""

    def __init__(self, config):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entity_types = ["disease", "symptom", "medication", "test", "anatomy"]
        self.relation_counter = {rel: 0 for rel in self.config.kg_relation_types}
        self.entity_dict = {}
        self.entity_name_to_id = {}
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
        if len(self.graph.nodes()) == 0:
            logger.warning("知识图谱为空，无法可视化")
            return None

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
        filepath = os.path.join(self.config.viz_dir, filename)
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

    def import_from_file(self, file_path):
        """从文件导入知识图谱"""
        try:
            file_processor = FileProcessor()
            content = file_processor.extract_text_from_file(file_path)

            if not content:
                logger.error(f"无法从文件提取内容: {file_path}")
                return False

            # 使用OLLAMA分析内容并构建知识图谱
            return self.build_with_ollama(content, file_path)

        except Exception as e:
            logger.error(f"导入知识图谱失败: {str(e)}")
            return False

    def build_with_ollama(self, content, source_info):
        """使用OLLAMA分析内容并构建知识图谱"""
        try:
            # 使用OLLAMA提取医疗实体和关系
            prompt = f"""
            请从以下医疗文本中提取疾病、症状、药物、检查和身体部位等实体，以及它们之间的关系。
            文本内容:
            {content[:4000]}  # 限制文本长度

            请以JSON格式返回结果，包含以下结构:
            {{
                "entities": [
                    {{
                        "type": "疾病/症状/药物/检查/身体部位",
                        "name": "实体名称",
                        "properties": {{}}
                    }}
                ],
                "relations": [
                    {{
                        "source": "源实体名称",
                        "target": "目标实体名称",
                        "type": "关系类型",
                        "properties": {{}}
                    }}
                ]
            }}
            """

            response = ollama.chat(
                model=self.config.ollama_model,
                messages=[{'role': 'user', 'content': prompt}]
            )

            result_text = response['message']['content']

            # 提取JSON部分
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result_data = json.loads(json_match.group())

                # 创建实体映射
                entity_map = {}
                for entity in result_data.get('entities', []):
                    entity_id = f"{entity['type']}_{len(entity_map)}"
                    entity_map[entity['name']] = entity_id
                    self.add_entity(entity_id, entity['type'], {
                        'name': entity['name'],
                        'source': source_info,
                        **entity.get('properties', {})
                    })

                # 创建关系
                for relation in result_data.get('relations', []):
                    source_id = entity_map.get(relation['source'])
                    target_id = entity_map.get(relation['target'])

                    if source_id and target_id:
                        self.add_relation(
                            source_id,
                            target_id,
                            relation['type'],
                            relation.get('properties', {})
                        )

                logger.info(
                    f"使用OLLAMA从 {source_info} 提取了 {len(entity_map)} 个实体和 {len(result_data.get('relations', []))} 个关系")
                return True
            else:
                logger.error("OLLAMA响应中没有找到有效的JSON数据")
                return False

        except Exception as e:
            logger.error(f"使用OLLAMA构建知识图谱失败: {str(e)}")
            return False

    def validate_graph(self):
        """验证知识图谱的完整性"""
        issues = []

        # 检查孤立节点
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            issues.append(f"发现 {len(isolated_nodes)} 个孤立节点")

        # 检查重复实体
        name_count = {}
        for node, data in self.graph.nodes(data=True):
            if 'name' in data:
                name = data['name']
                name_count[name] = name_count.get(name, 0) + 1

        duplicates = {name: count for name, count in name_count.items() if count > 1}
        if duplicates:
            issues.append(f"发现 {len(duplicates)} 个重复实体名称")

        # 检查无效关系
        for u, v, key in self.graph.edges(keys=True):
            if key not in self.config.kg_relation_types:
                issues.append(f"发现无效关系类型: {key}")

        return issues if issues else ["知识图谱验证通过，无发现问题"]


# ========== 知识图谱工厂 ==========
class KnowledgeGraphFactory:
    """知识图谱工厂，根据配置返回适当的知识图谱实例"""

    @staticmethod
    def create_knowledge_graph(config):
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

    def load_knowledge(self):
        """从知识库文件夹加载所有知识库文件"""
        logger.info("从本地知识库文件夹加载医疗知识...")

        # 询问用户知识图谱构建方式
        print("\n请选择知识图谱构建方式:")
        print("1. 自动从知识库学习生成知识图谱")
        print("2. 导入已有的知识图谱文件")
        choice = input("请输入选择 (1 或 2，直接回车默认选择1): ").strip()

        if choice == "2":
            import_file = input("请输入知识图谱文件路径: ").strip()
            if os.path.exists(import_file):
                if self.knowledge_graph.import_from_file(import_file):
                    logger.info("知识图谱导入成功")
                    # 验证知识图谱
                    validation_issues = self.knowledge_graph.validate_graph()
                    for issue in validation_issues:
                        logger.info(f"知识图谱验证: {issue}")

                    # 可视化知识图谱
                    self.knowledge_graph.visualize()
                    self.knowledge_graph.export_to_json()
                    self.knowledge_graph.save()
                    return
                else:
                    logger.error("知识图谱导入失败，将使用自动学习模式")
            else:
                logger.error("文件不存在，将使用自动学习模式")

        # 自动学习模式
        logger.info("使用自动学习模式构建知识图谱...")

        # 尝试加载已有的知识图谱
        if self.knowledge_graph.load():
            logger.info("成功加载已有的知识图谱")
            return

        file_processor = FileProcessor()
        documents = []  # 用于FAISS索引的文档

        # 处理知识库目录中的所有文件
        for root, _, files in os.walk(self.config.knowledge_base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"处理文件: {file_path}")

                try:
                    content = file_processor.extract_text_from_file(file_path)
                    if content:
                        # 使用OLLAMA分析内容并构建知识图谱
                        self.knowledge_graph.build_with_ollama(content, file_path)
                        documents.append(content)
                        self.learning_stats["files_processed"] += 1
                        self.learning_stats["total_size_kb"] += len(content.encode('utf-8')) / 1024
                except Exception as e:
                    logger.error(f"文件处理错误 {file_path}: {str(e)}")

        # 构建FAISS索引
        if documents:
            self.faiss_db.build_index(documents)

        # 可视化知识图谱
        if self.knowledge_graph.graph.number_of_nodes() > 0:
            self.knowledge_graph.visualize()
            self.knowledge_graph.export_to_json()
            self.knowledge_graph.save()

            # 更新学习统计
            self.learning_stats["kg_entities"] = self.knowledge_graph.graph.number_of_nodes()
            self.learning_stats["kg_relations"] = self.knowledge_graph.graph.number_of_edges()

        if self.learning_stats["files_processed"] == 0:
            logger.warning("知识库未加载任何内容")

    def prepare_training_data(self):
        """准备训练数据"""
        texts = []
        labels = []
        label_map = {}

        # 从知识图谱中提取训练数据
        for node, data in self.knowledge_graph.graph.nodes(data=True):
            if data.get('type') == 'disease' and 'name' in data:
                disease_name = data['name']

                if disease_name not in label_map:
                    label_map[disease_name] = len(label_map)

                # 获取相关症状
                symptoms = []
                for _, neighbor, key in self.knowledge_graph.graph.edges(node, keys=True):
                    if key == 'has_symptom':
                        neighbor_data = self.knowledge_graph.graph.nodes[neighbor]
                        if 'name' in neighbor_data:
                            symptoms.append(neighbor_data['name'])

                if symptoms:
                    symptoms_text = ", ".join(symptoms)
                    texts.append(symptoms_text)
                    labels.append(label_map[disease_name])

                    disease_text = f"{disease_name} with symptoms: {symptoms_text}"
                    texts.append(disease_text)
                    labels.append(label_map[disease_name])

        return texts, labels, label_map

    def rag_search(self, query, k=5):
        """使用RAG检索相关知识"""
        return self.faiss_db.search(query, k)

    def kg_query(self, query):
        """执行知识图谱查询"""
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

    def validate_knowledge_graph(self):
        """验证知识图谱的完整性"""
        return self.knowledge_graph.validate_graph()


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

    def plot_resource_usage(self, filename="resource_usage.png"):
        """绘制资源使用情况图"""
        plt.figure(figsize=(12, 6))

        time_points = range(len(self.cpu_usages))

        plt.plot(time_points, self.cpu_usages, label='CPU Usage', color='red', linewidth=2)
        plt.plot(time_points, self.gpu_usages, label='GPU Usage', color='blue', linewidth=2)
        plt.plot(time_points, self.memory_usages, label='Memory Usage', color='green', linewidth=2)

        plt.xlabel('Time')
        plt.ylabel('Usage (%)')
        plt.title('Resource Usage During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = os.path.join(self.config.viz_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"资源使用图已保存: {filepath}")
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
            # 查找相关症状
            related_entities = self.knowledge_graph.find_related_entities(symptom)
            for entity, data in related_entities:
                if data.get('type') == 'symptom' and 'name' in data and data['name'] not in expanded_symptoms:
                    expanded_symptoms.append(data['name'])

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

        # 步骤5: 生成最终响应
        return await self.generate_final_response(diagnosis)

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

    async def generate_final_response(self, diagnosis):
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

        # 组合中英文响应
        final_response = f"🌐 中文:\n{cn_response}\n\n"
        final_response += f"🌐 ENGLISH:\n{en_response}"

        return final_response


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

    # 保存所有图表
    learning_curve_path = monitor.plot_learning_curves()
    resource_usage_path = monitor.plot_resource_usage()

    logger.info(f"学习曲线已保存: {learning_curve_path}")
    logger.info(f"资源使用图已保存: {resource_usage_path}")

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

        # 验证知识图谱
        validation_results = knowledge_base.validate_knowledge_graph()
        logger.info("知识图谱验证结果:")
        for result in validation_results:
            logger.info(f"  - {result}")

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