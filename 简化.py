import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import psutil
import GPUtil
import tqdm
import chardet
import requests
from datetime import datetime
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import autocast
import logging
import traceback

# 配置日志
logging.basicConfig(
    filename='dialogue_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== CONFIGURATION ==========
class Config:
    def __init__(self):
        # 本地模型路径
        self.gpt2_model_path = r"C:\Users\Administrator\gpt2"
        self.tokenizer_path = self.gpt2_model_path

        # 修复: 使用127.0.0.1代替localhost避免IPv6问题
        self.ollama_host = "http://127.0.0.1:11434"

        # 加载分词器
        self.tokenizer = self.load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # 训练参数
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.epochs = 3  # 减少训练轮数以加快测试
        self.batch_size = 2  # 减小批大小以适应更复杂模型
        self.learning_rate = 3e-5  # 更精细的学习率
        self.max_seq_length = 384  # 增加序列长度
        self.rag_top_k = 5  # 增加检索上下文数量

        # 深度学习参数
        self.hidden_size = 768  # 减小隐藏层大小以节省内存
        self.num_layers = 2  # 减少知识融合层数
        self.dropout = 0.1  # 添加dropout防止过拟合
        self.weight_decay = 0.01  # 权重衰减
        self.max_grad_norm = 1.0  # 梯度裁剪

        # 设备配置
        self.cuda_available = torch.cuda.is_available()
        self.gpu_weight = 1.0 if self.cuda_available else 0.0
        self.cpu_weight = 0.0 if self.cuda_available else 1.0
        self.auto_balance = False

        # 自动混合精度 - 仅在CUDA可用时启用
        self.use_amp = True if self.cuda_available else False

        # 获取可用的Ollama模型
        self.available_models = self.get_available_ollama_models()
        if not self.available_models:
            raise RuntimeError("No Ollama models found. Please install at least one model.")

        # 自动选择最佳模型
        self.embedding_model = self.select_best_embedding_model()
        self.generation_models = self.select_generation_models()

        # 输出配置
        self.final_model_dir = "final_dialogue_model"
        self.model_file = "dialogue_model.pth"
        self.save_every = 500
        self.log_every = 100

        # 显示配置
        print("\n=== DEEP LEARNING CONFIGURATION ===")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {self.cuda_available}")
        if self.cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Using AMP: {self.use_amp}")
        print(f"GPT-2 Model Path: {self.gpt2_model_path}")
        print(f"Selected Embedding Model: {self.embedding_model}")
        print(f"Selected Generation Models: {', '.join(self.generation_models)}")
        print(f"Vocab Size: {self.vocab_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Layers: {self.num_layers}")
        print("===================================\n")

    def load_tokenizer(self):
        """从本地文件加载tokenizer"""
        if os.path.isdir(self.tokenizer_path):
            return GPT2Tokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        elif os.path.isfile(self.tokenizer_path):
            dir_path = os.path.dirname(self.tokenizer_path)
            return GPT2Tokenizer.from_pretrained(dir_path, local_files_only=True)
        else:
            raise ValueError(f"Invalid tokenizer path: {self.tokenizer_path}")

    def verify_files(self, required_files, path_type):
        """验证必需文件是否存在"""
        print(f"\nVerifying {path_type} files:")
        for file in required_files:
            file_path = os.path.join(self.gpt2_model_path, file)
            exists = os.path.exists(file_path)
            print(f"{file}: {'Found' if exists else 'MISSING'}")
            if not exists:
                raise FileNotFoundError(f"Required file missing: {file}")
        print("All files verified")

    def get_available_ollama_models(self):
        """获取本地可用的Ollama模型列表"""
        try:
            # 修复: 增加超时时间避免连接失败
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=120)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"Failed to get Ollama models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            return []

    def select_best_embedding_model(self):
        """自动选择最佳嵌入模型"""
        # 优先选择nomic-embed-text
        for model in self.available_models:
            if "nomic-embed-text" in model:
                logger.info(f"Auto-selected embedding model: {model}")
                return model

        # 如果没有nomic，选择其他嵌入模型
        for model in self.available_models:
            if "embed" in model or "bge" in model:
                logger.info(f"Selected embedding model: {model}")
                return model

        # 如果都没有，使用第一个模型
        logger.info(f"Using first available model for embedding: {self.available_models[0]}")
        return self.available_models[0]

    def select_generation_models(self):
        """选择所有非嵌入模型作为生成模型"""
        # 排除嵌入模型和已知问题模型
        skip_models = [self.embedding_model, "gemma3:4b", "gemma:4b"]

        generation_models = [
            model for model in self.available_models
            if model not in skip_models
        ]

        if not generation_models:
            # 如果没有其他模型，使用除问题模型外的所有模型
            generation_models = [m for m in self.available_models if m not in skip_models]

        if not generation_models:
            # 最后尝试使用所有模型
            logger.warning("No non-problematic models found. Using all models with risk.")
            return self.available_models

        logger.info(f"Selected {len(generation_models)} generation models: {', '.join(generation_models)}")
        return generation_models


# ========== KNOWLEDGE BASE ==========
class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.log_file = "knowledge_errors.log"
        self.chunks = []
        self.loaded_files = []

    def log_error(self, message):
        """带时间戳记录错误"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        logger.error(message)

    def load_files(self, path):
        """从目录或文件加载知识"""
        if not os.path.exists(path):
            error = f"Path not found: {path}"
            self.log_error(error)
            return False, error

        try:
            # 处理目录
            if os.path.isdir(path):
                logger.info(f"Loading directory: {path}")
                success = self.load_directory(path)
                desc = f"directory '{os.path.basename(path)}'"
                return success, f"Loaded directory with {len(self.chunks)} chunks"

            # 处理单个文件
            elif os.path.isfile(path) and path.lower().endswith('.txt'):
                logger.info(f"Loading single file: {path}")
                success = self.load_single_file(path)
                desc = f"file '{os.path.basename(path)}'"
                return success, f"Loaded file with {len(self.chunks)} chunks"

            else:
                error = "Only directories or .txt files supported"
                self.log_error(error)
                return False, error

        except Exception as e:
            error = f"Error loading files: {str(e)}"
            self.log_error(error)
            return False, error

    def load_directory(self, directory_path):
        """加载目录中的所有文本文件"""
        self.chunks = []
        self.loaded_files = []

        # 获取所有.txt文件
        txt_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    txt_files.append(os.path.join(root, file))

        if not txt_files:
            logger.warning(f"No .txt files found in directory: {directory_path}")
            return False

        logger.info(f"Found {len(txt_files)} .txt files in directory")

        # 逐个加载文件
        for file_path in txt_files:
            try:
                self.load_single_file(file_path)
                self.loaded_files.append(file_path)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
                continue

        if not self.chunks:
            logger.error("No valid chunks created from directory")
            return False

        # 创建向量存储
        logger.info(f"Creating vector store with {self.config.embedding_model}...")
        embeddings = OllamaEmbeddings(model=self.config.embedding_model)
        self.vector_store = FAISS.from_texts(self.chunks, embeddings)
        logger.info("Vector store created successfully")
        return True

    def load_single_file(self, file_path):
        """加载单个文本文件"""
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()

            # 如果文件为空，跳过
            if len(raw_data) == 0:
                logger.warning(f"Skipping empty file: {file_path}")
                return False

            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence']

            # 如果置信度低，尝试常见编码
            if confidence < 0.7:
                for test_encoding in ['utf-8', 'gbk', 'latin-1', 'iso-8859-1']:
                    try:
                        content = raw_data.decode(test_encoding, errors='strict')
                        encoding = test_encoding
                        logger.info(f"Using fallback encoding {test_encoding} for {file_path}")
                        break
                    except UnicodeDecodeError:
                        continue

        logger.info(f"Using encoding: {encoding} for {file_path}")

        try:
            # 尝试解码内容
            content = raw_data.decode(encoding, errors='replace')

            # 如果内容为空，跳过
            if not content.strip():
                logger.warning(f"Skipping empty content file: {file_path}")
                return False

            # 使用TextLoader处理内容
            loader = TextLoader(file_path, encoding=encoding)
            document = loader.load()[0].page_content

            # 将文本分割成块
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            chunks = splitter.split_text(document)
            self.chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False

    def retrieve_context(self, query, k=None):
        """使用RAG检索相关上下文"""
        if k is None:
            k = self.config.rag_top_k

        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return ""

        try:
            results = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in results])
            logger.info(f"Retrieved context: {context[:100]}...")
            return context
        except Exception as e:
            error = f"Retrieval error: {str(e)}"
            self.log_error(error)
            return ""


# ========== DIALOGUE MODEL ==========
class KnowledgeFusionLayer(nn.Module):
    """深度知识融合层"""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(KnowledgeFusionLayer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = hidden_size
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, knowledge_embeds):
        # 通过全连接层
        x = knowledge_embeds
        for layer in self.layers:
            x = layer(x)

        # 自注意力
        x = x.permute(1, 0, 2)  # 转换形状为(seq_len, batch, features)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x.permute(1, 0, 2))  # 转换回原始形状

        return x


class DialogueModel(nn.Module):
    def __init__(self, config):
        super(DialogueModel, self).__init__()
        self.config = config

        # 加载GPT-2模型作为对话生成核心
        logger.info(f"Loading GPT-2 model from: {config.gpt2_model_path}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path)
        self.gpt2.resize_token_embeddings(config.vocab_size)
        logger.info("GPT-2 model loaded successfully")

        # 初始化知识token的嵌入层
        self.embedding_layer = self.gpt2.transformer.wte

        # 深度知识融合层
        self.knowledge_fusion = KnowledgeFusionLayer(
            input_size=self.gpt2.config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        # 残差连接
        self.residual_projection = nn.Linear(config.hidden_size, self.gpt2.config.hidden_size)

    def forward(self, input_ids, attention_mask=None, knowledge_ids=None):
        # 将token IDs转换为嵌入向量
        inputs_embeds = self.embedding_layer(input_ids)

        if knowledge_ids is not None:
            # 获取知识嵌入 (batch_size, seq_len, hidden_size)
            knowledge_embeds = self.embedding_layer(knowledge_ids)

            # 聚合知识 (平均池化)
            knowledge_embeds = knowledge_embeds.mean(dim=1, keepdim=True)  # (batch_size, 1, hidden_size)

            # 深度知识融合
            fused_knowledge = self.knowledge_fusion(knowledge_embeds)

            # 投影到GPT-2嵌入空间
            projected_knowledge = self.residual_projection(fused_knowledge)

            # 将知识与输入嵌入融合
            inputs_embeds = inputs_embeds + projected_knowledge

        # 通过GPT-2模型生成响应
        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

    def generate(self, input_ids, attention_mask=None, knowledge_ids=None, **kwargs):
        """自定义生成方法"""
        # 将token IDs转换为嵌入向量
        inputs_embeds = self.embedding_layer(input_ids)

        if knowledge_ids is not None:
            # 获取知识嵌入
            knowledge_embeds = self.embedding_layer(knowledge_ids)

            # 聚合知识
            knowledge_embeds = knowledge_embeds.mean(dim=1, keepdim=True)

            # 深度知识融合
            fused_knowledge = self.knowledge_fusion(knowledge_embeds)

            # 投影到GPT-2嵌入空间
            projected_knowledge = self.residual_projection(fused_knowledge)

            # 将知识与输入嵌入融合
            inputs_embeds = inputs_embeds + projected_knowledge

        # 通过GPT-2模型生成响应
        return self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )


# ========== DATASET ==========
class DialogueDataset(Dataset):
    def __init__(self, knowledge_base, config):
        self.config = config
        self.tokenizer = config.tokenizer
        self.examples = []
        self.ollama_host = config.ollama_host
        self.generation_models = config.generation_models

        # 直接从知识块创建简单训练样本
        logger.info(f"Creating training examples from {len(knowledge_base.chunks)} chunks...")
        for chunk in knowledge_base.chunks:
            # 创建简单的问答对
            question = f"Tell me about {chunk[:50].strip()}..."
            answer = chunk[:200].strip()

            self.examples.append({
                "input": question,
                "output": answer,
                "knowledge": chunk
            })

        logger.info(f"Created {len(self.examples)} training examples")

        # 保存示例到文件用于调试
        try:
            with open("training_examples.json", "w", encoding='utf-8') as f:
                json.dump(self.examples[:min(10, len(self.examples))], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving training examples: {str(e)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # 编码输入和输出
        input_ids = self.tokenizer.encode(
            example["input"],
            max_length=self.config.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )

        output_ids = self.tokenizer.encode(
            example["output"],
            max_length=self.config.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )

        # 创建完整序列
        full_ids = input_ids + [self.tokenizer.eos_token_id] + output_ids + [self.tokenizer.eos_token_id]

        # 创建标签（忽略输入部分的损失）
        labels = [-100] * (len(input_ids) + 1)  # 输入部分和第一个EOS token用-100忽略
        labels += output_ids + [self.tokenizer.eos_token_id]  # 只计算输出部分的损失

        # 确保长度一致
        if len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]
        elif len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))

        # 编码知识
        knowledge_ids = self.tokenizer.encode(
            example["knowledge"],
            max_length=self.config.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "knowledge_ids": torch.tensor(knowledge_ids, dtype=torch.long)
        }


# ========== DATA COLLATION ==========
def collate_fn(batch):
    # 填充序列
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100  # 用-100填充标签
    )

    knowledge_ids = pad_sequence(
        [item["knowledge_ids"] for item in batch],
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).int(),
        "labels": labels,
        "knowledge_ids": knowledge_ids
    }


# ========== TRAINER ==========
class ModelTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base

        # 设备配置
        if config.cuda_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        self.model = DialogueModel(config)
        self.model.to(self.device)

        # 自动混合精度 - 使用新版本的初始化方式
        if config.use_amp and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler(
                device_type='cuda',
                enabled=True
            )
            logger.info(f"Enabled Automatic Mixed Precision (AMP) - Using CUDA")
        elif config.use_amp and not torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler(
                device_type='cpu',
                enabled=True
            )
            logger.info(f"Enabled Automatic Mixed Precision (AMP) - Using CPU")
        else:
            self.scaler = None
            logger.info("Disabled Automatic Mixed Precision (AMP)")

        # 准备数据集
        dataset = DialogueDataset(knowledge_base, config)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn,
            pin_memory=True, num_workers=0  # 减少工作线程数
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True, num_workers=0  # 减少工作线程数
        )

        # 优化器和调度器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # 监控
        self.loss_history = []
        self.val_loss_history = []
        self.learning_rates = []
        self.resource_monitor = ResourceMonitor()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.resource_monitor.start()
        batch_count = 0

        for batch_idx, batch in enumerate(tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")):
            # 将数据移动到设备
            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            masks = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            knowledge = batch["knowledge_ids"].to(self.device, non_blocking=True)

            # 使用自动混合精度 (如果启用)
            if self.scaler is not None:
                with autocast(enabled=self.config.use_amp):
                    # 前向传播
                    model_output = self.model(
                        input_ids=inputs,
                        attention_mask=masks,
                        knowledge_ids=knowledge
                    )

                    # 计算损失
                    shift_logits = model_output.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
            else:
                # 不使用 AMP 的前向传播
                model_output = self.model(
                    input_ids=inputs,
                    attention_mask=masks,
                    knowledge_ids=knowledge
                )
                shift_logits = model_output.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

            # 反向传播
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.scheduler.step()

            # 记录指标
            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            self.loss_history.append(loss.item())
            self.learning_rates.append(current_lr)
            batch_count += 1

            # 定期保存和记录
            if (batch_idx + 1) % self.config.save_every == 0:
                self.save_model(f"checkpoint_epoch{epoch + 1}_batch{batch_idx + 1}")

            if (batch_idx + 1) % self.config.log_every == 0:
                self.resource_monitor.log_resources()
                logger.info(f"Batch {batch_idx + 1} Loss: {loss.item():.4f}, LR: {current_lr:.2e}")

        avg_loss = total_loss / batch_count
        logger.info(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f}")
        self.resource_monitor.stop()
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        self.resource_monitor.start()

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validation"):
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                knowledge = batch["knowledge_ids"].to(self.device)

                # 使用自动混合精度
                if self.scaler is not None:
                    with autocast(enabled=self.config.use_amp):
                        model_output = self.model(
                            input_ids=inputs,
                            attention_mask=masks,
                            knowledge_ids=knowledge
                        )

                        shift_logits = model_output.logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = nn.CrossEntropyLoss(ignore_index=-100)(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                else:
                    model_output = self.model(
                        input_ids=inputs,
                        attention_mask=masks,
                        knowledge_ids=knowledge
                    )
                    shift_logits = model_output.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                total_loss += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count
        self.val_loss_history.append(avg_loss)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        self.resource_monitor.stop()
        return avg_loss

    def train(self):
        os.makedirs(self.config.final_model_dir, exist_ok=True)
        start_time = time.time()
        best_val_loss = float('inf')

        # CUDA可用性验证
        if self.config.cuda_available:
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                logger.info("CUDA operation verification successful")
            except Exception as e:
                logger.error(f"CUDA operation failed: {str(e)}")
                logger.info("Falling back to CPU mode")
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.config.use_amp = False
                if self.scaler:
                    self.scaler = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best")

        # 保存最终模型
        self.save_model("final")
        training_time = time.time() - start_time

        # 保存训练元数据
        metadata = {
            "training_time": training_time,
            "final_train_loss": self.loss_history[-1] if self.loss_history else 0,
            "final_val_loss": self.val_loss_history[-1] if self.val_loss_history else 0,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "embedding_model": self.config.embedding_model,
            "generation_models": self.config.generation_models,
            "gpt2_model": self.config.gpt2_model_path,
            "best_val_loss": best_val_loss,
            "knowledge_files": self.knowledge_base.loaded_files
        }

        try:
            with open(os.path.join(self.config.final_model_dir, "metadata.json"), "w", encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

        # 绘制训练损失图
        self.plot_training_loss()

        return self.config.final_model_dir

    def save_model(self, prefix):
        model_path = os.path.join(self.config.final_model_dir, f"{prefix}_{self.config.model_file}")
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss_history': self.loss_history,
                'val_loss_history': self.val_loss_history,
                'config': self.config.__dict__
            }, model_path)
            logger.info(f"Saved model to: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

        # 同时保存最终模型的标准名称
        if prefix == "final":
            final_path = os.path.join(self.config.final_model_dir, "final_model.pth")
            try:
                torch.save(self.model.state_dict(), final_path)
                logger.info(f"Saved final model to: {final_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {str(e)}")

    def plot_training_loss(self):
        if not self.loss_history or not self.val_loss_history:
            return

        plt.figure(figsize=(12, 8))

        # 训练损失
        plt.subplot(2, 1, 1)
        plt.plot(self.loss_history, label='Training Loss')
        plt.title("Training Loss History")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # 验证损失
        plt.subplot(2, 1, 2)
        plt.plot(self.val_loss_history, label='Validation Loss', color='orange')
        plt.title("Validation Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.config.final_model_dir, "training_metrics.png")
        try:
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved training plot to: {plot_path}")
        except Exception as e:
            logger.error(f"Error saving training plot: {str(e)}")


# ========== RESOURCE MONITOR ==========
class ResourceMonitor:
    def __init__(self):
        self.active = False
        self.log_file = "resource_usage.csv"
        self.start_time = None

        # 创建日志文件头
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", encoding='utf-8') as f:
                    f.write(
                        "timestamp,cpu_usage(%),ram_usage(%),gpu_usage(%),gpu_mem(%),ram_used(GB),ram_available(GB),gpu_mem_details\n")
            except Exception as e:
                logger.error(f"Error creating resource log: {str(e)}")

    def start(self):
        self.active = True
        self.start_time = time.time()

    def stop(self):
        self.active = False

    def log_resources(self):
        if not self.active:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # CPU使用率
        cpu_percent = psutil.cpu_percent()

        # RAM使用率
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used = ram.used / (1024 ** 3)  # 转换为GB
        ram_available = ram.available / (1024 ** 3)  # 转换为GB

        # GPU使用率
        gpu_percent = 0
        gpu_mem_percent = 0
        gpu_mem_details = ""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for i, gpu in enumerate(gpus):
                gpu_percent = gpu.load * 100
                gpu_mem_percent = gpu.memoryUtil * 100
                gpu_info.append(f"GPU{i}: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}MB ({gpu_mem_percent:.1f}%)")
            gpu_mem_details = "; ".join(gpu_info)
        except Exception as e:
            logger.error(f"GPU monitoring error: {str(e)}")

        # 写入日志
        log_line = f"{timestamp},{cpu_percent},{ram_percent},{gpu_percent},{gpu_mem_percent},{ram_used:.2f},{ram_available:.2f},\"{gpu_mem_details}\""

        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(log_line + "\n")
        except Exception as e:
            logger.error(f"Error writing resource log: {str(e)}")

        logger.info(
            f"Resources: CPU {cpu_percent}%, RAM {ram_percent}% ({ram_used:.1f}/{ram.total / (1024 ** 3):.1f} GB), GPU {gpu_percent}%")

        return {
            "cpu": cpu_percent,
            "ram": ram_percent,
            "ram_used": ram_used,
            "gpu": gpu_percent,
            "gpu_mem": gpu_mem_percent
        }


# ========== DIALOGUE ENGINE ==========
class DialogueEngine:
    def __init__(self, model_path, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base
        self.model_path = model_path

        # 设备选择
        if config.cuda_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        # 加载模型
        self.model = DialogueModel(config)

        # 检查文件是否存在
        possible_paths = [
            model_path,
            os.path.join(os.path.dirname(model_path), "final_model.pth"),
            os.path.join(os.path.dirname(model_path), "best_dialogue_model.pth"),
            os.path.join(os.path.dirname(model_path), "final_dialogue_model.pth")
        ]

        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # 加载模型状态字典
                    state_dict = torch.load(path, map_location=self.device)
                    if 'model_state_dict' in state_dict:
                        self.model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        self.model.load_state_dict(state_dict)

                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Model loaded successfully from: {path}")
                    loaded = True
                    self.model_path = path
                    break
                except Exception as e:
                    logger.error(f"Error loading model from {path}: {str(e)}")

        if not loaded:
            raise FileNotFoundError(f"No valid model found in possible paths: {possible_paths}")

        # 加载tokenizer
        self.tokenizer = config.tokenizer

    def generate_response(self, query, max_length=150, temperature=0.7, top_p=0.9, top_k=50):
        try:
            # 检索相关知识
            context = self.knowledge_base.retrieve_context(query)
            logger.info(f"Retrieved context length: {len(context)} characters")

            # 将上下文编码为知识ID
            knowledge_ids = self.tokenizer.encode(
                context,
                max_length=self.config.max_seq_length,
                truncation=True,
                add_special_tokens=False
            )
            knowledge_tensor = torch.tensor([knowledge_ids], dtype=torch.long).to(self.device)

            # 编码查询
            input_ids = self.tokenizer.encode(
                query,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)

            # 创建注意力掩码
            attention_mask = torch.ones(input_ids.shape, device=self.device)

            # 使用自定义模型生成响应
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    knowledge_ids=knowledge_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # 解码响应
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # 移除查询部分（如果存在）
            if response.startswith(query):
                response = response[len(query):].strip()

            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I encountered an error while generating a response. Please try again."


# ========== MAIN ==========
def main():
    try:
        # 初始化配置
        logger.info("Initializing configuration...")
        config = Config()

        # 步骤1: 加载知识库
        logger.info("\n[1/4] Loading knowledge base...")
        default_path = "knowledge_base"
        path = input(
            f"Enter path to knowledge directory or file (press Enter for '{default_path}'): ").strip() or default_path

        # 修复Windows路径问题
        path = path.replace("/", "\\")

        logger.info(f"Using knowledge path: {path}")
        knowledge_base = KnowledgeBase(config)
        success, message = knowledge_base.load_files(path)

        if not success:
            logger.error(f"Error: {message}")
            print(f"Error: {message}")
            print(f"Check error log: {knowledge_base.log_file}")
            return

        logger.info(f"Success: {message}")
        print(f"Success: {message}")

        # 步骤2: 训练模型
        logger.info("\n[2/4] Training dialogue model...")
        trainer = ModelTrainer(config, knowledge_base)
        model_dir = trainer.train()

        # 步骤3: 初始化对话引擎
        logger.info("\n[3/4] Initializing dialogue engine...")
        # 使用标准模型文件名
        model_path = os.path.join(model_dir, "final_model.pth")
        engine = DialogueEngine(model_path, config, knowledge_base)
        logger.info("Dialogue engine ready")
        print("Dialogue engine ready")

        # 步骤4: 交互式对话
        logger.info("\n[4/4] Starting conversation")
        print("\n[4/4] Starting conversation (type 'exit' to quit)")
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['exit', 'quit', 'bye']:
                    break

                # 生成响应
                start_time = time.time()
                response = engine.generate_response(
                    query,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=100
                )
                response_time = time.time() - start_time

                # 显示结果
                print(f"\nAssistant: {response}")
                print(f"Response time: {response_time:.2f}s")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                logger.error(f"Conversation error: {str(e)}")
    except Exception as e:
        error_msg = f"Fatal error: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()