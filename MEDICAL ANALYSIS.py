import os
import time
import json
import torch
import csv
import fitz  # PyMuPDF for PDF
import docx
import numpy as np
import re  # 新增正则表达式库
import matplotlib.pyplot as plt
import psutil
import GPUtil
from tqdm import tqdm
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, CosineSimilarity
from torch.cuda.amp import GradScaler, autocast
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings
import torch.nn as nn


# ========== CONFIGURATION ==========
class MedicalConfig:
    def __init__(self):
        # 硬件配置
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        print(f"Using device: {self.device}")

        # 模型路径（完全离线）
        self.gpt2_model_path = "C:/MedicalAI/models/gpt2"
        self.ollama_model_path = "C:/MedicalAI/models/ollama"

        # 知识库路径（扫描所有驱动器）
        self.knowledge_paths = self.find_knowledge_paths()

        # 训练参数
        self.epochs = 30  # 增加训练轮数提高精度
        self.batch_size = 2  # 减小批量大小以使用更大模型
        self.learning_rate = 1e-5
        self.max_seq_length = 512

        # 文件处理
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # 输出配置
        self.model_name = "MEDICAL_ANALYSIS"
        self.output_dir = "MEDICAL_ANALYSIS_MODEL"
        os.makedirs(self.output_dir, exist_ok=True)

        # 多语言支持
        self.languages = ["CN", "EN"]
        self.current_lang = "CN"

        # 加载本地模型
        self.tokenizer = self.load_tokenizer()
        self.embedding_model = self.load_ollama_embeddings()

        print("\n=== MEDICAL ANALYSIS CONFIGURATION ===")
        print(f"GPT-2 Model Path: {self.gpt2_model_path}")
        print(f"Ollama Model Path: {self.ollama_model_path}")
        print(f"Knowledge Paths: {self.knowledge_paths}")
        print(f"Training Epochs: {self.epochs}")
        print("=====================================")

    def find_knowledge_paths(self):
        """扫描所有驱动器寻找知识库"""
        paths = []
        drives = [f"{d}:\\" for d in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{d}:\\")]

        for drive in drives:
            for root, dirs, files in os.walk(drive):
                # 搜索医学知识库和症状描述的目录名
                if "medical_knowledge" in root.lower() or "health_data" in root.lower() or "zhishiku" in root.lower():
                    paths.append(root)
                # 最多扫描3层深度
                if root.count(os.sep) - drive.count(os.sep) >= 3:
                    dirs[:] = []  # 停止深入
        return paths if paths else ["./zhishiku"]  # 默认路径改为zhishiku

    def load_tokenizer(self):
        """从本地文件加载tokenizer"""
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_path, local_files_only=True)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.add_special_tokens({'bos_token': '<BOS>'})
            tokenizer.add_special_tokens({'eos_token': '<EOS>'})
            return tokenizer
        except:
            # 如果失败，使用基本tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.add_special_tokens({'bos_token': '<BOS>'})
            tokenizer.add_special_tokens({'eos_token': '<EOS>'})
            return tokenizer

    def load_ollama_embeddings(self):
        """加载Ollama嵌入模型（离线）"""
        return OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.ollama_model_path,
            timeout=300
        )

    def set_language(self, lang):
        """设置当前语言"""
        if lang.upper() in self.languages:
            self.current_lang = lang.upper()
            print(f"Language set to: {self.current_lang}")
        else:
            print(f"Unsupported language. Available: {', '.join(self.languages)}")


# ========== KNOWLEDGE BASE ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.chunks = []
        self.disease_info = {}
        self.symptom_info = {}

        # 加载知识
        self.load_knowledge()

        # 创建向量存储
        self.create_vector_store()

    def load_file(self, file_path):
        """加载单个知识文件"""
        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                return loader.load()
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
                return loader.load()
            elif file_path.endswith('.pdf'):
                return self.load_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self.load_docx(file_path)
            elif file_path.endswith('.json'):
                return self.load_json(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
        return []

    def load_pdf(self, file_path):
        """加载PDF文件"""
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return [text]

    def load_docx(self, file_path):
        """加载DOCX文件"""
        text = ""
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return [text]

    def load_json(self, file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [json.dumps(data, ensure_ascii=False)]

    def extract_medical_info(self, text):
        """增强的医疗信息提取 - 使用混合方法"""
        # 处理多种格式的知识数据
        if text.startswith('{') and text.endswith('}'):  # 可能是JSON
            try:
                data = json.loads(text)
                if 'disease' in data:
                    disease_name = data['disease']
                    self.disease_info[disease_name] = {
                        "symptoms": data.get('symptoms', []),
                        "treatments": data.get('treatments', []),
                        "medications": data.get('medications', []),
                        "description": data.get('description', "")
                    }
                elif 'symptom' in data:
                    symptom_name = data['symptom']
                    self.symptom_info[symptom_name] = {
                        "description": data.get('description', ""),
                        "possible_diseases": data.get('related_diseases', [])
                    }
                return
            except:
                pass  # 不是有效JSON，尝试规则匹配

        # 规则匹配 - 更健壮的匹配策略
        lines = text.split('\n')
        disease_info = {}
        symptom_info = {}

        # 更灵活的匹配规则
        for line in lines:
            if re.search(r'(疾病|disease):?', line, re.IGNORECASE):
                key, value = self.split_key_value(line)
                if key and value:
                    disease_info['name'] = value
            elif re.search(r'(症状|symptoms|signs):?', line, re.IGNORECASE):
                key, value = self.split_key_value(line)
                if key and value:
                    disease_info['symptoms'] = [s.strip() for s in value.split(',')]
            elif re.search(r'(治疗|treatments|therapy):?', line, re.IGNORECASE):
                key, value = self.split_key_value(line)
                if key and value:
                    disease_info['treatments'] = [t.strip() for t in value.split(',')]
            elif re.search(r'(药物|medications|drugs):?', line, re.IGNORECASE):
                key, value = self.split_key_value(line)
                if key and value:
                    disease_info['medications'] = [m.strip() for m in value.split(',')]
            elif re.search(r'(症状|symptom):?', line, re.IGNORECASE) and not disease_info:
                key, value = self.split_key_value(line)
                if key and value:
                    symptom_info['name'] = value
            elif re.search(r'(描述|description):?', line, re.IGNORECASE) and symptom_info:
                key, value = self.split_key_value(line)
                if key and value:
                    symptom_info['description'] = value
            elif re.search(r'(相关疾病|related diseases):?', line, re.IGNORECASE) and symptom_info:
                key, value = self.split_key_value(line)
                if key and value:
                    symptom_info['possible_diseases'] = [d.strip() for d in value.split(',')]

        # 保存提取的信息
        if 'name' in disease_info:
            self.disease_info[disease_info['name']] = {
                "symptoms": disease_info.get('symptoms', []),
                "treatments": disease_info.get('treatments', []),
                "medications": disease_info.get('medications', []),
                "description": ""
            }

        if 'name' in symptom_info:
            self.symptom_info[symptom_info['name']] = {
                "description": symptom_info.get('description', ""),
                "possible_diseases": symptom_info.get('possible_diseases', [])
            }

    def split_key_value(self, line):
        """提取键值对"""
        patterns = [
            r'^(.*?):(.*)$',  # 冒号分隔
            r'^(.*?)=(.*)$',  # 等号分隔
            r'^(.*?)\s{2,}(.*)$'  # 多个空格分隔
        ]

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return None, None

    def load_knowledge(self):
        """加载所有知识库文件"""
        print("Loading medical knowledge...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 确保zhishiku目录存在
        if "./zhishiku" in self.config.knowledge_paths and not os.path.exists("./zhishiku"):
            os.makedirs("./zhishiku")
            print("Created default knowledge directory: ./zhishiku")

        for path in self.config.knowledge_paths:
            if not os.path.exists(path):
                continue

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file_path.endswith(('.txt', '.csv', '.pdf', '.docx', '.json')):
                        try:
                            content = self.load_file(file_path)
                            for doc in content:
                                # 提取医疗信息
                                self.extract_medical_info(doc)

                                # 分割文本
                                chunks = splitter.split_text(doc)
                                self.chunks.extend(chunks)
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")

        print(f"Loaded {len(self.chunks)} knowledge chunks")
        print(f"Extracted {len(self.disease_info)} diseases and {len(self.symptom_info)} symptoms")

        # 保存提取的知识以便检查
        with open('disease_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.disease_info, f, ensure_ascii=False, indent=2)
        with open('symptom_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.symptom_info, f, ensure_ascii=False, indent=2)

    def create_vector_store(self):
        """创建向量存储"""
        print("Creating vector store...")
        embeddings = self.config.embedding_model
        self.vector_store = FAISS.from_texts(self.chunks, embeddings)
        print("Vector store created")

    def retrieve_context(self, query, k=5):
        """检索相关知识"""
        results = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])

    def diagnose(self, symptoms):
        """根据症状诊断疾病 - 使用增强的算法"""
        # 第一步：精确匹配
        possible_diseases = {}
        for symptom in symptoms:
            if symptom in self.symptom_info:
                for disease in self.symptom_info[symptom]["possible_diseases"]:
                    if disease in self.disease_info:
                        possible_diseases[disease] = possible_diseases.get(disease, 0) + 1
        # 第二步：模糊匹配（使用词嵌入）
        embeddings = self.config.embedding_model
        cos_sim = CosineSimilarity(dim=0)
        for symptom in symptoms:
            symptom_embedding = torch.tensor(embeddings.embed_query(symptom)).to(self.config.device)
            for disease, info in self.disease_info.items():
                for dis_symptom in info["symptoms"]:
                    dis_embedding = torch.tensor(embeddings.embed_query(dis_symptom)).to(self.config.device)
                    similarity = cos_sim(symptom_embedding, dis_embedding)
                    if similarity > 0.6:  # 相似度阈值
                        possible_diseases[disease] = possible_diseases.get(disease, 0) + similarity.item()
        # 计算概率
        results = []
        for disease, score in possible_diseases.items():
            total_symptoms = len(self.disease_info[disease]["symptoms"])
            probability = min(score / total_symptoms, 1.0) * 100
            results.append({
                "disease": disease,
                "probability": round(probability, 1),
                "score": score,
                "total_symptoms": total_symptoms
            })
        # 按概率排序
        return sorted(results, key=lambda x: x["probability"], reverse=True)

    def get_treatment_plan(self, disease):
        """获取治疗方案"""
        if disease in self.disease_info:
            return {
                "treatments": self.disease_info[disease]["treatments"],
                "medications": self.disease_info[disease]["medications"]
            }
        return {"treatments": [], "medications": []}


# ========== MEDICAL MODEL ==========
class MedicalAnalysisModel(nn.Module):
    def __init__(self, config):
        super(MedicalAnalysisModel, self).__init__()
        self.config = config

        # 加载本地GPT-2模型
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path, local_files_only=True)
        self.gpt2.resize_token_embeddings(len(config.tokenizer))

        # 知识融合层 - 增强融合方法
        self.knowledge_projection = nn.Sequential(
            nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd)
        )
        self.layer_norm = nn.LayerNorm(self.gpt2.config.n_embd)
        self.knowledge_attention = nn.MultiheadAttention(self.gpt2.config.n_embd, num_heads=4, batch_first=True)

        # 知识增强层
        self.knowledge_enhancer = nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd)

        print("Medical Analysis Model initialized")

    def forward(self, input_ids, attention_mask=None, knowledge_embeds=None):
        # 获取输入嵌入
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        if knowledge_embeds is not None:
            # 知识投影
            projected_knowledge = self.knowledge_projection(knowledge_embeds)

            # 注意力机制融合知识
            attn_output, _ = self.knowledge_attention(
                inputs_embeds,
                projected_knowledge,
                projected_knowledge,
                key_padding_mask=None
            )

            # 残差连接和层归一化
            inputs_embeds = inputs_embeds + attn_output
            inputs_embeds = self.layer_norm(inputs_embeds)

            # 知识增强
            knowledge_factor = torch.sigmoid(self.knowledge_enhancer(projected_knowledge))
            inputs_embeds = inputs_embeds * (1 + knowledge_factor)

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

    def generate(self, input_ids, attention_mask=None, knowledge_embeds=None, **kwargs):
        """生成响应 - 使用增强的方法"""
        inputs_embeds = self.gpt2.transformer.wte(input_ids)

        if knowledge_embeds is not None:
            # 知识投影
            projected_knowledge = self.knowledge_projection(knowledge_embeds)

            # 注意力机制融合知识
            attn_output, _ = self.knowledge_attention(
                inputs_embeds,
                projected_knowledge,
                projected_knowledge,
                key_padding_mask=None
            )

            # 残差连接和层归一化
            inputs_embeds = inputs_embeds + attn_output
            inputs_embeds = self.layer_norm(inputs_embeds)

            # 知识增强
            knowledge_factor = torch.sigmoid(self.knowledge_enhancer(projected_knowledge))
            inputs_embeds = inputs_embeds * (1 + knowledge_factor)

        # 设置更好的生成参数
        generation_config = {
            "max_length": self.config.max_seq_length,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "pad_token_id": self.config.tokenizer.eos_token_id,
            **kwargs
        }

        return self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generation_config
        )


# ========== DATASET ==========
class MedicalDataset(Dataset):
    def __init__(self, knowledge_base, config):
        self.config = config
        self.knowledge_base = knowledge_base
        self.tokenizer = config.tokenizer
        self.examples = []

        # 生成训练数据
        self.generate_training_data()
        print(f"Created {len(self.examples)} training examples")
        self.augment_data()  # 数据增强
        print(f"After augmentation: {len(self.examples)} training examples")

    def generate_training_data(self):
        """生成医疗训练数据"""
        # 1. 疾病诊断示例
        for disease, info in self.knowledge_base.disease_info.items():
            symptoms = info["symptoms"]
            treatments = info["treatments"]
            medications = info["medications"]

            # 中文示例
            self.examples.append({
                "input": f"患者症状: {', '.join(symptoms)}",
                "output": f"<BOS>根据症状分析，患者可能患有{disease}。建议治疗方案: {', '.join(treatments[:3])}。推荐药物: {', '.join(medications[:2])}。<EOS>",
                "knowledge": f"疾病名称: {disease}\n症状: {', '.join(symptoms)}\n治疗方案: {', '.join(treatments)}\n推荐药物: {', '.join(medications)}",
                "lang": "CN"
            })

            # 英文示例
            self.examples.append({
                "input": f"Patient symptoms: {', '.join(symptoms)}",
                "output": f"<BOS>Based on the symptoms, the patient may have {disease}. Recommended treatments: {', '.join(treatments[:3])}. Medications: {', '.join(medications[:2])}.<EOS>",
                "knowledge": f"Disease: {disease}\nSymptoms: {', '.join(symptoms)}\nTreatments: {', '.join(treatments)}\nMedications: {', '.join(medications)}",
                "lang": "EN"
            })

        # 2. 症状分析示例
        for symptom, info in self.knowledge_base.symptom_info.items():
            diseases = info["possible_diseases"]

            # 中文示例
            self.examples.append({
                "input": f"患者主诉: {symptom}",
                "output": f"<BOS>{symptom}可能与以下疾病有关: {', '.join(diseases[:3])}。建议进行进一步检查。<EOS>",
                "knowledge": f"症状名称: {symptom}\n描述: {info['description']}\n相关疾病: {', '.join(diseases)}",
                "lang": "CN"
            })

            # 英文示例
            self.examples.append({
                "input": f"Patient complains: {symptom}",
                "output": f"<BOS>{symptom} may be related to: {', '.join(diseases[:3])}. Further examination is recommended.<EOS>",
                "knowledge": f"Symptom: {symptom}\nDescription: {info['description']}\nRelated diseases: {', '.join(diseases)}",
                "lang": "EN"
            })

    def augment_data(self):
        """数据增强 - 创建变体示例"""
        original_count = len(self.examples)
        for i in range(original_count):
            example = self.examples[i].copy()

            # 1. 同义词替换
            if example["lang"] == "CN":
                example["input"] = example["input"].replace("患者", "病人")
                example["input"] = example["input"].replace("症状", "临床表现")
                example["output"] = example["output"].replace("治疗方案", "治疗方式")
            else:
                example["input"] = example["input"].replace("Patient", "Subject")
                example["input"] = example["input"].replace("symptoms", "manifestations")

            self.examples.append(example)

            # 2. 问题变体
            if "症状" in example["input"] or "symptoms" in example["input"]:
                new_example = example.copy()
                if example["lang"] == "CN":
                    new_example["input"] = f"根据症状'{example['input'].split(':')[1].strip()}'，请诊断可能疾病"
                else:
                    new_example[
                        "input"] = f"Based on symptom '{example['input'].split(':')[1].strip()}', what disease might it be?"
                self.examples.append(new_example)

            # 3. 部分信息查询
            if "疾病" in example["knowledge"] or "disease" in example["knowledge"]:
                new_example = example.copy()
                disease_name = new_example["knowledge"].split(':')[1].split('\n')[0].strip()
                if new_example["lang"] == "CN":
                    new_example["input"] = f"获取疾病'{disease_name}'的治疗方案"
                    new_example[
                        "output"] = f"<BOS>治疗方案: {new_example['output'].split('治疗方式:')[1].split('<EOS>')[0]}<EOS>"
                else:
                    new_example["input"] = f"Get treatment for '{disease_name}'"
                    new_example[
                        "output"] = f"<BOS>Recommended treatments: {new_example['output'].split('Recommended treatments:')[1].split('<EOS>')[0]}<EOS>"
                self.examples.append(new_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # 编码输入
        input_enc = self.config.tokenizer(
            example["input"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码输出
        output_enc = self.config.tokenizer(
            example["output"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码知识
        knowledge_enc = self.config.tokenizer(
            example["knowledge"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 知识嵌入
        knowledge_text = example["knowledge"]
        embeddings = self.config.embedding_model
        knowledge_embeds = torch.tensor(embeddings.embed_query(knowledge_text)).unsqueeze(0)

        # 创建标签（忽略填充部分）
        labels = output_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "knowledge_embeds": knowledge_embeds.squeeze(0),
            "labels": labels.squeeze(0),
            "lang": example["lang"]
        }


# ========== TRAINER ==========
class MedicalTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base
        self.model = MedicalAnalysisModel(config).to(config.device)

        # 准备数据集
        dataset = MedicalDataset(knowledge_base, config)
        self.train_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )

        # 优化器
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        # 学习率调度器
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # 损失函数
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        # 训练监控
        self.loss_history = []
        self.gpu_usage = []
        self.cpu_usage = []
        self.gpu_memory = []
        self.learning_rates = []

        # 混合精度训练
        self.scaler = GradScaler() if config.cuda_available else None

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch in progress_bar:
            # 记录资源使用情况
            self.record_resources()

            # 移到设备
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            knowledge_embeds = batch["knowledge_embeds"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # 混合精度训练
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        knowledge_embeds=knowledge_embeds
                    )

                    # 计算损失
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    knowledge_embeds=knowledge_embeds
                )

                # 计算损失
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
                self.optimizer.step()

            self.scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            self.loss_history.append(loss.item())
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
        return avg_loss

    def record_resources(self):
        """记录资源使用情况"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.append(cpu_percent)

        # GPU使用率
        if self.config.cuda_available:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                gpu_mem = gpus[0].memoryUsed
                self.gpu_usage.append(gpu_percent)
                self.gpu_memory.append(gpu_mem)

    def save_model(self, epoch=None):
        """保存模型"""
        if epoch is not None:
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}_epoch{epoch + 1}.pth")
        else:
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.pth")

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_history[-1] if self.loss_history else 0.0
        }, model_path)

        print(f"Saved model to: {model_path}")
        return model_path

    def generate_sci_charts(self):
        """生成SCI论文所需图表"""
        plt.figure(figsize=(15, 12))

        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history)
        plt.title("Training Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)

        # 学习率
        plt.subplot(2, 2, 2)
        plt.plot(self.learning_rates)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.grid(True)

        # CPU/GPU使用率
        plt.subplot(2, 2, 3)
        if self.gpu_usage:
            plt.plot(self.gpu_usage, label='GPU Usage (%)')
        plt.plot(self.cpu_usage, label='CPU Usage (%)')
        plt.title("Resource Utilization")
        plt.xlabel("Iteration")
        plt.ylabel("Usage (%)")
        plt.legend()
        plt.grid(True)

        # GPU内存使用
        if self.gpu_memory:
            plt.subplot(2, 2, 4)
            plt.plot(self.gpu_memory)
            plt.title("GPU Memory Usage")
            plt.xlabel("Iteration")
            plt.ylabel("Memory (MB)")
            plt.grid(True)

        plt.tight_layout()
        chart_path = os.path.join(self.config.output_dir, "training_metrics.png")
        plt.savefig(chart_path, dpi=300)
        print(f"Saved training charts to: {chart_path}")


# ========== MEDICAL ASSISTANT ==========
class MedicalAssistant:
    def __init__(self, model_path, knowledge_base, config):
        self.config = config
        self.knowledge_base = knowledge_base

        # 加载模型
        checkpoint = torch.load(model_path, map_location=config.device)
        self.model = MedicalAnalysisModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(config.device)
        self.model.eval()

    def analyze_symptoms(self, symptoms_text, lang=None):
        """分析症状并提供医疗建议"""
        if lang:
            self.config.set_language(lang)

        # 提取症状关键词
        symptoms = self.extract_symptoms(symptoms_text)

        if not symptoms:
            return self.response("未识别到有效症状", "No valid symptoms recognized")

        # 诊断疾病
        diagnoses = self.knowledge_base.diagnose(symptoms)

        if not diagnoses:
            return self.response("未识别到可能的疾病", "No possible diseases recognized")

        # 获取最高概率疾病
        top_diagnosis = diagnoses[0]
        disease = top_diagnosis["disease"]

        # 获取治疗方案
        treatment = self.knowledge_base.get_treatment_plan(disease)

        # 生成响应
        if self.config.current_lang == "CN":
            response = (
                f"根据症状分析，患者可能患有{disease}（可能性{top_diagnosis['probability']}%，匹配度{top_diagnosis['score']:.2f}）。\n\n"
                f"建议检查：\n{self.format_list(treatment['treatments'])}\n\n"
                f"推荐药物：\n{self.format_list(treatment['medications'])}\n\n"
                f"基于医学知识库的AI诊断结果"
            )
        else:
            response = (
                f"Based on symptom analysis, the patient may have {disease} (probability {top_diagnosis['probability']}%, match score {top_diagnosis['score']:.2f}).\n\n"
                f"Recommended tests:\n{self.format_list(treatment['treatments'], lang='EN')}\n\n"
                f"Recommended medications:\n{self.format_list(treatment['medications'], lang='EN')}\n\n"
                f"AI diagnosis based on medical knowledge base"
            )

        return response

    def extract_symptoms(self, text):
        """从文本中提取症状关键词"""
        found_symptoms = []
        for symptom in self.knowledge_base.symptom_info.keys():
            if symptom in text:
                found_symptoms.append(symptom)

        # 使用嵌入的相似性匹配
        embeddings = self.config.embedding_model
        cos_sim = CosineSimilarity(dim=0)
        user_embedding = torch.tensor(embeddings.embed_query(text)).to(self.config.device)

        for symptom in self.knowledge_base.symptom_info.keys():
            symptom_embedding = torch.tensor(embeddings.embed_query(symptom)).to(self.config.device)
            similarity = cos_sim(user_embedding, symptom_embedding)
            if similarity > 0.7 and symptom not in found_symptoms:
                found_symptoms.append(symptom)

        return found_symptoms

    def format_list(self, items, lang=None):
        """格式化列表"""
        lang = lang or self.config.current_lang
        if lang == "CN":
            return "\n".join([f"- {item}" for item in items])
        return "\n".join([f"- {item}" for item in items])

    def response(self, cn_text, en_text):
        """根据当前语言返回响应"""
        if self.config.current_lang == "CN":
            return cn_text
        return en_text


# ========== MAIN ==========
def main():
    try:
        # 初始化配置
        config = MedicalConfig()

        # 加载知识库
        print("\n[1/3] Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # 训练模型
        print("\n[2/3] Training Medical Analysis Model...")
        trainer = MedicalTrainer(config, knowledge_base)

        best_loss = float('inf')
        best_epoch = -1

        for epoch in range(config.epochs):
            current_loss = trainer.train_epoch(epoch)

            # 保存最好的模型
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch
                trainer.save_model(epoch)

        # 保存最终模型和图表
        model_path = trainer.save_model()
        trainer.generate_sci_charts()

        # 初始化医疗助手
        print("\n[3/3] Starting Medical Assistant (using best model from epoch {best_epoch+1})")
        assistant = MedicalAssistant(
            os.path.join(config.output_dir, f"{config.model_name}_epoch{best_epoch + 1}.pth"),
            knowledge_base,
            config
        )

        # 交互界面
        print("\n=== MEDICAL ANALYSIS ASSISTANT ===")
        print("Commands: lang [CN/EN], exit")

        while True:
            try:
                # 获取用户输入
                user_input = input("\nPatient Symptoms: ").strip()

                # 处理命令
                if user_input.lower() == "exit":
                    break
                elif user_input.lower().startswith("lang "):
                    lang = user_input.split()[1].upper()
                    config.set_language(lang)
                    continue

                # 分析症状
                start_time = time.time()
                response = assistant.analyze_symptoms(user_input)
                response_time = time.time() - start_time

                # 显示响应
                print(f"\nMedical Analysis:\n{response}")
                print(f"Response time: {response_time:.2f}s")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()