import os
import time
import json
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, CosineSimilarity
from torch.cuda.amp import GradScaler, autocast
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings
import torch.nn as nn
import docx
import csv
import GPUtil

# 强制使用CUDA
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


# PDF 处理模块
def pdf_to_text(file_path):
    """使用 PyPDF2 处理 PDF 文件"""
    from PyPDF2 import PdfReader
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {file_path}: {str(e)}")
        return ""


# ========== CONFIGURATION ==========
class MedicalConfig:
    def __init__(self):
        # 硬件配置
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        print(f"Using device: {self.device}")

        if self.cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # GPT-2 模型路径 (仅使用项目根目录)
        self.gpt2_model_path = os.path.join(os.getcwd(), "gpt2")

        # 检查模型是否存在
        if os.path.exists(self.gpt2_model_path) and any(
                os.path.exists(os.path.join(self.gpt2_model_path, f))
                for f in ["pytorch_model.bin", "model.safetensors"]
        ):
            print(f"Found GPT-2 model at: {self.gpt2_model_path}")
            self.gpt2_exists = True
        else:
            print(f"GPT-2 model not found at: {self.gpt2_model_path}")
            self.gpt2_exists = False

        # Ollama 模型路径
        self.ollama_model_path = os.path.join(os.getcwd(), "ollama_models")
        os.makedirs(self.ollama_model_path, exist_ok=True)
        print(f"Ollama models stored at: {self.ollama_model_path}")

        # 知识库路径 (仅使用项目根目录下的 knowledge_base)
        self.knowledge_paths = self.setup_knowledge_paths()

        # 训练参数
        self.epochs = 30
        self.batch_size = 2
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
        print(f"GPT-2 Model Exists: {self.gpt2_exists}")
        print(f"Knowledge Paths: {self.knowledge_paths}")
        print(f"Using GPU: {self.cuda_available}")
        print("=====================================")

    def setup_knowledge_paths(self):
        """设置知识库路径为项目根目录下的knowledge_base文件夹"""
        knowledge_dir = os.path.join(os.getcwd(), "knowledge_base")

        # 创建目录（如果不存在）
        if not os.path.exists(knowledge_dir):
            print(f"Knowledge base directory not found. Creating at: {knowledge_dir}")
            os.makedirs(knowledge_dir)
            # 添加示例文件
            with open(os.path.join(knowledge_dir, "example.txt"), "w", encoding="utf-8") as f:
                f.write("疾病: 感冒\n症状: 咳嗽,发烧,流鼻涕\n治疗方案: 休息,多喝水,服用退烧药")

        return [knowledge_dir]

    def load_tokenizer(self):
        """加载tokenizer"""
        if not self.gpt2_exists:
            print("GPT-2 model not found, using base tokenizer")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({
                'pad_token': '[PAD]',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })
            return tokenizer

        try:
            print(f"Loading tokenizer from: {self.gpt2_model_path}")
            tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_path)

            # 添加特殊token
            tokenizer.add_special_tokens({
                'pad_token': '[PAD]',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            print("Using base GPT-2 tokenizer as fallback")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.add_special_tokens({
                'pad_token': '[PAD]',
                'bos_token': '<BOS>',
                'eos_token': '<EOS>'
            })
            return tokenizer

    def load_ollama_embeddings(self):
        """加载Ollama嵌入模型（离线）"""
        try:
            print("Loading Ollama embedding model...")
            return OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.ollama_model_path,
                timeout=300
            )
        except Exception as e:
            print(f"Error loading Ollama embeddings: {str(e)}")
            return None


# ========== KNOWLEDGE BASE ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.chunks = []
        self.disease_info = {}
        self.symptom_info = {}
        self.total_chunks = 0

        # 加载知识
        self.load_knowledge()

        # 创建向量存储
        self.create_vector_store()
        print(
            f"Knowledge base loaded. Diseases: {len(self.disease_info)}, Symptoms: {len(self.symptom_info)}, Chunks: {len(self.chunks)}")

    def load_file(self, file_path):
        """加载单个知识文件，包含改进的错误处理"""
        try:
            content = ""
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    content = "\n".join([",".join(row) for row in reader])
            elif file_path.endswith('.pdf'):
                content = pdf_to_text(file_path)
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False)
            else:
                print(f"Unsupported file format: {file_path}")
                return []

            return [content]
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    def extract_medical_info(self, text):
        """简单的医疗信息提取"""
        try:
            # 疾病信息提取（简化规则）
            disease_pattern = re.compile(r'(?:疾病|disease)[\s:：]*([^\n]+)', re.IGNORECASE)
            symptom_pattern = re.compile(r'(?:症状|symptoms)[\s:：]*([^\n]+)', re.IGNORECASE)
            treatment_pattern = re.compile(r'(?:治疗方案|treatments)[\s:：]*([^\n]+)', re.IGNORECASE)
            medication_pattern = re.compile(r'(?:推荐药物|medications)[\s:：]*([^\n]+)', re.IGNORECASE)

            # 尝试提取疾病信息
            disease_match = disease_pattern.search(text)
            if disease_match:
                disease_name = disease_match.group(1).strip().split('\n')[0].split(',')[0]

                # 提取相关症状
                symptoms = []
                symptom_match = symptom_pattern.search(text)
                if symptom_match:
                    symptoms = [s.strip() for s in symptom_match.group(1).split(',')]

                # 提取治疗方案
                treatments = []
                treatment_match = treatment_pattern.search(text)
                if treatment_match:
                    treatments = [t.strip() for t in treatment_match.group(1).split(',')]

                # 提取药物
                medications = []
                medication_match = medication_pattern.search(text)
                if medication_match:
                    medications = [m.strip() for m in medication_match.group(1).split(',')]

                # 保存疾病信息
                self.disease_info[disease_name] = {
                    "symptoms": symptoms,
                    "treatments": treatments,
                    "medications": medications
                }

            # 症状信息提取（简化规则）
            symptom_names = set()
            for line in text.split('\n'):
                if "症状" in line or "symptom" in line.lower():
                    parts = re.split(r'[:：]', line, maxsplit=1)
                    if len(parts) > 1:
                        symptoms = [s.strip() for s in parts[1].split(',')]
                        symptom_names.update(symptoms)

            # 保存症状信息
            for symptom in symptom_names:
                if symptom not in self.symptom_info:
                    self.symptom_info[symptom] = {
                        "description": "",
                        "possible_diseases": []
                    }

            return True
        except Exception as e:
            print(f"Error extracting medical info: {str(e)}")
            return False

    def load_knowledge(self):
        """加载所有知识库文件 - 使用简单的文本处理"""
        print("Loading medical knowledge...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        for path in self.config.knowledge_paths:
            if not os.path.exists(path):
                print(f"Knowledge path not found: {path}")
                continue

            print(f"Processing directory: {path}")
            file_count = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file_path.endswith(ext) for ext in ('.txt', '.csv', '.pdf', '.docx', '.json')):
                        print(f"Processing file: {file_path}")
                        try:
                            # 加载文件内容
                            contents = self.load_file(file_path)

                            for content in contents:
                                # 提取医疗信息
                                success = self.extract_medical_info(content)

                                # 分割文本无论是否提取成功
                                chunks = splitter.split_text(content)
                                self.chunks.extend(chunks)
                                self.total_chunks += len(chunks)

                            file_count += 1
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")

            print(f"Processed {file_count} files in {path}")

    def create_vector_store(self):
        """创建向量存储"""
        if not self.chunks or not self.config.embedding_model:
            print("No knowledge chunks or embedding model, skipping vector store")
            return

        print(f"Creating vector store with {len(self.chunks)} chunks...")
        try:
            self.vector_store = FAISS.from_texts(self.chunks, self.config.embedding_model)
            print("Vector store created")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            self.vector_store = None

    def diagnose(self, symptoms):
        """根据症状诊断疾病 - 简单匹配"""
        possible_diseases = {}

        # 如果没有症状，返回空
        if not symptoms:
            return []

        # 对每个症状，查找相关疾病
        for symptom in symptoms:
            for disease, info in self.disease_info.items():
                if any(symptom.lower() in s.lower() for s in info["symptoms"]):
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1
                elif symptom.lower() in disease.lower():
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1

        # 计算概率
        results = []
        for disease, count in possible_diseases.items():
            total_symptoms = len(self.disease_info[disease]["symptoms"])
            if total_symptoms == 0:
                total_symptoms = 1
            probability = min(count / total_symptoms, 1.0) * 100
            results.append({
                "disease": disease,
                "probability": round(probability, 1),
                "matched_symptoms": count,
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

        # 加载GPT-2模型
        try:
            if config.gpt2_exists:
                print(f"Loading GPT-2 model from: {config.gpt2_model_path}")
                self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path)
            else:
                raise Exception("GPT-2 model not found")
        except Exception as e:
            print(f"Error loading GPT-2 model: {str(e)}")
            self.gpt2 = None
            return

        # 调整token嵌入大小
        self.gpt2.resize_token_embeddings(len(config.tokenizer))
        self.gpt2.to(config.device)

        print(f"Medical Analysis Model initialized on {config.device}")

    def forward(self, input_ids, attention_mask=None, knowledge_embeds=None):
        if self.gpt2 is None:
            raise RuntimeError("GPT-2 model not loaded")

        return self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    def generate(self, input_ids, attention_mask=None, max_length=512, **kwargs):
        """生成响应"""
        generation_config = {
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.config.tokenizer.eos_token_id,
            **kwargs
        }

        return self.gpt2.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )


# ========== TRAINER ==========
class MedicalTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base

        # 检查模型
        if not config.gpt2_exists:
            print("GPT-2 model not found, cannot train")
            self.model = None
            return

        # 准备模型
        self.model = MedicalAnalysisModel(config).to(config.device)
        print(f"Model placed on {config.device}")

        # 准备数据集
        dataset = MedicalDataset(knowledge_base, config)
        self.train_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )

        # 确保有训练数据
        if len(dataset) == 0:
            print("No training data available")
            return

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

        # 混合精度训练
        self.scaler = GradScaler() if config.cuda_available else None
        print(f"Training initialized with {len(dataset)} examples")

    def train_epoch(self, epoch):
        if self.model is None:
            print("Model not available, skipping training")
            return 0

        self.model.train()
        epoch_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
        for batch in progress_bar:
            # 确保所有张量都在正确设备上
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # 混合精度训练
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
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
                    attention_mask=attention_mask
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

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, epoch=None):
        """保存模型"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        if epoch is not None:
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}_epoch{epoch + 1}.pth")
        else:
            model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.pth")

        torch.save(self.model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")
        return model_path


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

    def generate_training_data(self):
        """生成医疗训练数据"""
        # 1. 疾病诊断示例
        for disease, info in self.knowledge_base.disease_info.items():
            symptoms = info["symptoms"]
            treatments = info["treatments"]
            medications = info["medications"]

            if symptoms:
                # 中文示例
                self.examples.append({
                    "input": f"患者症状: {', '.join(symptoms[:3])}",
                    "output": f"<BOS>根据症状分析，患者可能患有{disease}。建议治疗方案: {', '.join(treatments[:2])}。推荐药物: {', '.join(medications[:1])}。<EOS>",
                    "lang": "CN"
                })

                # 英文示例
                self.examples.append({
                    "input": f"Patient symptoms: {', '.join(symptoms[:3])}",
                    "output": f"<BOS>Based on the symptoms, the patient may have {disease}. Recommended treatments: {', '.join(treatments[:2])}. Medications: {', '.join(medications[:1])}.<EOS>",
                    "lang": "EN"
                })

        # 2. 症状分析示例
        for symptom in self.knowledge_base.symptom_info.keys():
            # 中文示例
            self.examples.append({
                "input": f"患者主诉: {symptom}",
                "output": f"<BOS>{symptom}可能与多种疾病相关，建议详细检查。<EOS>",
                "lang": "CN"
            })

            # 英文示例
            self.examples.append({
                "input": f"Patient complains: {symptom}",
                "output": f"<BOS>{symptom} may be related to various diseases. Detailed examination is recommended.<EOS>",
                "lang": "EN"
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # 编码输入
        input_enc = self.tokenizer(
            example["input"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码输出
        output_enc = self.tokenizer(
            example["output"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 创建标签（忽略填充部分）
        labels = output_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "lang": example["lang"]
        }


# ========== MEDICAL ASSISTANT ==========
class MedicalAssistant:
    def __init__(self, model_path, knowledge_base, config):
        self.config = config
        self.knowledge_base = knowledge_base
        self.conversation_context = []

        # 加载模型
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            self.model = None
            return

        try:
            self.model = MedicalAnalysisModel(config)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.model.to(config.device)
            self.model.eval()
            print(f"Medical assistant model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def analyze_symptoms(self, symptoms_text, lang=None):
        """分析症状并提供医疗建议"""
        if lang:
            self.config.set_language(lang)

        # 保存用户输入
        self.conversation_context.append(symptoms_text)

        # 提取症状关键词
        symptoms = self.extract_symptoms(symptoms_text)

        # 尝试诊断
        diagnoses = self.knowledge_base.diagnose(symptoms)

        # 响应逻辑
        if not diagnoses:
            if len(self.conversation_context) > 2:  # 多次尝试后
                return self.response(
                    "基于当前知识库无法确定病因，请更新知识库",
                    "Unable to determine cause based on current knowledge base. Consider updating knowledge base"
                )
            else:  # 首次尝试
                return self.response(
                    "请更详细描述您的症状",
                    "Please describe your symptoms in more detail"
                )

        # 获取最高概率疾病
        top_diagnosis = diagnoses[0]
        disease = top_diagnosis["disease"]
        treatment = self.knowledge_base.get_treatment_plan(disease)

        # 生成响应
        if self.config.current_lang == "CN":
            response = (
                f"根据症状分析，患者可能患有{disease}（匹配度{top_diagnosis['probability']}%）。\n\n"
                f"建议检查：\n{self.format_list(treatment['treatments'])}\n\n"
                f"推荐药物：\n{self.format_list(treatment['medications'])}\n\n"
                f"请注意：此为AI建议，请咨询专业医师确认诊断"
            )
        else:
            response = (
                f"Based on symptom analysis, the patient may have {disease} (confidence {top_diagnosis['probability']}%).\n\n"
                f"Recommended tests:\n{self.format_list(treatment['treatments'])}\n\n"
                f"Recommended medications:\n{self.format_list(treatment['medications'])}\n\n"
                f"Note: This is AI-generated advice. Please consult a medical professional."
            )

        return response

    def extract_symptoms(self, text):
        """从文本中提取症状关键词"""
        found_symptoms = []

        # 遍历知识库中的所有症状
        for symptom in self.knowledge_base.symptom_info.keys():
            if symptom.lower() in text.lower():
                found_symptoms.append(symptom)

        # 如果未找到，尝试模糊匹配
        if not found_symptoms:
            symptom_words = set()
            for word in text.split():
                # 常见症状关键词列表
                symptom_keywords = ["痛", "疼", "咳", "烧", "吐", "晕", "血", "痒", "肿", "不适"]
                if any(keyword in word for keyword in symptom_words) or any(
                        keyword in word for keyword in symptom_keywords):
                    symptom_words.add(word)

            found_symptoms = list(symptom_words)

        return found_symptoms

    def response(self, cn_text, en_text):
        """根据当前语言返回响应"""
        if self.config.current_lang == "CN":
            return cn_text
        return en_text

    def clear_context(self):
        """清除对话上下文"""
        self.conversation_context = []


# ========== MAIN ==========
def main():
    try:
        # 初始化配置
        config = MedicalConfig()

        # 检查GPT-2模型是否存在
        if not config.gpt2_exists:
            print("GPT-2 model not found. Place GPT-2 model in 'gpt2' directory")
            return

        # 加载知识库
        print("\n[1/3] Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # 检查知识库是否为空
        if not knowledge_base.chunks:
            print("知识库为空，已自动创建示例文件")
            # 创建示例文件后重新加载
            time.sleep(1)
            knowledge_base = MedicalKnowledgeBase(config)

        # 训练模型
        print("\n[2/3] Training Medical Analysis Model...")
        trainer = MedicalTrainer(config, knowledge_base)

        best_loss = float('inf')
        best_epoch = -1

        for epoch in range(config.epochs):
            current_loss = trainer.train_epoch(epoch)

            # 保存最好的模型
            if current_loss < best_loss and current_loss > 0:
                best_loss = current_loss
                best_epoch = epoch
                trainer.save_model(epoch)

        # 保存最终模型
        if best_epoch >= 0:
            model_path = trainer.save_model(best_epoch)
        else:
            model_path = os.path.join(config.output_dir, f"{config.model_name}_default.pth")
            print(f"Using default model path: {model_path}")

        # 初始化医疗助手
        print(f"\n[3/3] Starting Medical Assistant (model: {model_path})")
        assistant = MedicalAssistant(model_path, knowledge_base, config)

        # 交互界面
        print("\n=== MEDICAL ANALYSIS ASSISTANT ===")
        print("Commands: lang [CN/EN], clear, exit")

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
                    print(f"Language set to: {lang}")
                    continue
                elif user_input.lower() == "clear":
                    if assistant:
                        assistant.clear_context()
                        print("对话上下文已清除")
                    continue

                # 分析症状
                if assistant and assistant.model:
                    start_time = time.time()
                    response = assistant.analyze_symptoms(user_input)
                    response_time = time.time() - start_time

                    # 显示响应
                    print(f"\nAssistant: {response}")
                    print(f"Response time: {response_time:.2f}s")
                else:
                    print("Medical assistant not available")

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