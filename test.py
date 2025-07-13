import json
import os
import time
import traceback
from datetime import datetime

import chardet
import matplotlib.pyplot as plt
import psutil
import requests
import torch
import torch.nn as nn
import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup


# ========== CONFIGURATION ==========
class Config:
    def __init__(self):
        # Local model paths
        self.gpt2_model_path = r"C:\Users\Administrator\gpt2"
        self.tokenizer_path = self.gpt2_model_path

        # Use 127.0.0.1 instead of localhost
        self.ollama_host = "http://127.0.0.1:11434"

        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # Training parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.epochs = 3  # Reduced for testing
        self.batch_size = 2  # Reduced for testing
        self.learning_rate = 3e-5
        self.max_seq_length = 384
        self.rag_top_k = 5

        # Deep learning parameters
        self.hidden_size = 1024
        self.num_layers = 4
        self.dropout = 0.2
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # Device configuration
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.device_count)]
            print(f"Found {self.device_count} GPU(s):")
            for i, name in enumerate(self.gpu_names):
                print(f"  GPU {i}: {name}")
        else:
            self.device_count = 0
            self.gpu_names = []
            print("No CUDA-capable GPUs found")

        self.gpu_weight = 1.0 if self.cuda_available else 0.0
        self.cpu_weight = 0.0 if self.cuda_available else 1.0
        self.auto_balance = False

        # Automatic Mixed Precision
        self.use_amp = True if self.cuda_available else False

        # Get available Ollama models
        self.available_models = self.get_available_ollama_models()
        if not self.available_models:
            self.available_models = ["llama2"]  # Fallback
            print("Warning: Using fallback Ollama model")

        # Automatically select best model
        self.embedding_model = self.select_best_embedding_model()
        self.generation_models = self.select_generation_models()

        # Output configuration
        self.final_model_dir = "final_dialogue_model"
        self.model_file = "dialogue_model.pth"
        self.save_every = 500
        self.log_every = 100

        # Display configuration
        print("\n=== DEEP LEARNING CONFIGURATION ===")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {self.cuda_available}")
        if self.cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")

    def load_tokenizer(self):
        """Load GPT2 tokenizer"""
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer loaded successfully")
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {str(e)}")
            # Fallback to pretrained tokenizer
            return GPT2Tokenizer.from_pretrained('gpt2')

    def get_available_ollama_models(self):
        """Retrieve available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                return models
            return []
        except Exception:
            return []

    def select_best_embedding_model(self):
        """Select best model for embeddings"""
        for model in self.available_models:
            if "embed" in model.lower():
                return model
        return self.available_models[0] if self.available_models else "llama2"

    def select_generation_models(self):
        """Select models for generation"""
        return [model for model in self.available_models if "embed" not in model.lower()][:2]


# ========== KNOWLEDGE BASE ==========
class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.log_file = "knowledge_errors.log"
        self.chunks = []
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Knowledge Base Error Log - Created at {datetime.now()}\n\n")

    def log_error(self, message):
        """Log error with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n\n")

    def load_files(self, path):
        """Load knowledge from directory or file"""
        if not os.path.exists(path):
            error = f"Path not found: {path}"
            self.log_error(error)
            return False, error

        try:
            documents = []
            if os.path.isdir(path):
                file_count = 0
                success_count = 0
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.txt'):
                            file_path = os.path.join(root, file)
                            file_count += 1
                            try:
                                # Detect encoding
                                with open(file_path, 'rb') as f:
                                    raw_data = f.read(10000)
                                    result = chardet.detect(raw_data)
                                    encoding = result['encoding'] or 'utf-8'

                                # Load file
                                loader = TextLoader(file_path, encoding=encoding)
                                docs = loader.load()
                                documents.extend(docs)
                                success_count += 1
                            except Exception as e:
                                error_msg = f"Error loading file {file_path}: {str(e)}"
                                self.log_error(f"File: {file_path}\nError: {error_msg}\n{traceback.format_exc()}")
            else:
                # Single file
                try:
                    with open(path, 'rb') as f:
                        raw_data = f.read(10000)
                        result = chardet.detect(raw_data)
                        encoding = result['encoding'] or 'utf-8'

                    loader = TextLoader(path, encoding=encoding)
                    documents = loader.load()
                    success_count = 1
                except Exception as e:
                    error_msg = f"Error loading file {path}: {str(e)}"
                    self.log_error(f"File: {path}\nError: {error_msg}\n{traceback.format_exc()}")
                    return False, error_msg

            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            all_chunks = []
            for doc in documents:
                try:
                    chunks = splitter.split_text(doc.page_content)
                    all_chunks.extend(chunks)
                except Exception as e:
                    self.log_error(f"Text splitting error: {str(e)}\nContent snippet: {doc.page_content[:200]}...")
                    all_chunks.append(doc.page_content)

            if not all_chunks:
                error = "No valid chunks created from documents"
                self.log_error(error)
                return False, error

            self.chunks = all_chunks

            # Create vector store
            try:
                embeddings = OllamaEmbeddings(model=self.config.embedding_model)
                self.vector_store = FAISS.from_texts(self.chunks, embeddings)
                return True, f"Loaded {len(self.chunks)} chunks"
            except Exception as e:
                error = f"Error creating vector store: {str(e)}"
                self.log_error(error)
                return False, error

        except Exception as e:
            error = f"Critical error loading files: {str(e)}\n{traceback.format_exc()}"
            self.log_error(error)
            return False, error

    def retrieve_context(self, query, k=None):
        """Retrieve relevant context using RAG"""
        if k is None:
            k = self.config.rag_top_k

        if not self.vector_store:
            return ""

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            self.log_error(f"Retrieval error: {str(e)}")
            return ""


# ========== DIALOGUE MODEL ==========
class KnowledgeFusionLayer(nn.Module):
    """Deep knowledge fusion layer"""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(KnowledgeFusionLayer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = hidden_size
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, knowledge_embeds):
        x = knowledge_embeds
        for layer in self.layers:
            x = layer(x)

        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm(x.permute(1, 0, 2))

        return x


class DialogueModel(nn.Module):
    def __init__(self, config):
        super(DialogueModel, self).__init__()
        self.config = config

        print(f"Loading GPT-2 model from: {config.gpt2_model_path}")
        try:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path)
            self.gpt2.resize_token_embeddings(config.vocab_size)
            print("GPT-2 model loaded successfully")
        except Exception as e:
            print(f"Error loading GPT-2: {e}, using pretrained model")
            self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
            config.vocab_size = self.gpt2.config.vocab_size

        self.embedding_layer = self.gpt2.transformer.wte

        self.knowledge_fusion = KnowledgeFusionLayer(
            input_size=self.gpt2.config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        self.residual_projection = nn.Linear(config.hidden_size, self.gpt2.config.hidden_size)

    def forward(self, input_ids, attention_mask=None, knowledge_ids=None):
        inputs_embeds = self.embedding_layer(input_ids)

        if knowledge_ids is not None:
            knowledge_embeds = self.embedding_layer(knowledge_ids)
            knowledge_embeds = knowledge_embeds.mean(dim=1, keepdim=True)
            fused_knowledge = self.knowledge_fusion(knowledge_embeds)
            projected_knowledge = self.residual_projection(fused_knowledge)
            inputs_embeds = inputs_embeds + projected_knowledge

        return self.gpt2(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )


# ========== DATASET ==========
class DialogueDataset(Dataset):
    def __init__(self, knowledge_base, config):
        self.config = config
        self.tokenizer = config.tokenizer
        self.examples = []
        self.ollama_host = config.ollama_host
        self.generation_models = config.generation_models

        print(f"Generating training examples with {len(self.generation_models)} Ollama models...")
        for chunk_idx, chunk in enumerate(
                tqdm.tqdm(knowledge_base.chunks[:50], desc="Processing chunks")):  # Limit chunks for testing
            for model_name in self.generation_models:
                try:
                    # Simulate question generation if Ollama not available
                    question = f"What is the main idea of this text: {chunk[:100]}?"
                    answer = f"This text discusses: {chunk[:200]}"

                    self.examples.append({
                        "input": question,
                        "output": answer,
                        "knowledge": chunk,
                        "model": model_name
                    })
                except Exception as e:
                    print(f"Error generating example: {str(e)}")

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

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

        full_ids = input_ids + [self.tokenizer.eos_token_id] + output_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) + 1)
        labels += output_ids + [self.tokenizer.eos_token_id]

        if len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]
        elif len(labels) < len(full_ids):
            labels += [-100] * (len(full_ids) - len(labels))

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
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100
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


# ========== RESOURCE MONITOR ==========
class ResourceMonitor:
    def __init__(self):
        self.active = False
        self.log_file = "resource_usage.csv"
        self.start_time = None

        with open(self.log_file, "w") as f:
            f.write(
                "timestamp,cpu_usage(%),ram_usage(%),gpu_usage(%),gpu_mem(%),ram_used(GB),ram_available(GB),gpu_mem_details\n")

    def start(self):
        self.active = True
        self.start_time = time.time()

    def stop(self):
        self.active = False

    def log_resources(self):
        if not self.active:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used = ram.used / (1024 ** 3)
        ram_available = ram.available / (1024 ** 3)

        gpu_percent = 0
        gpu_mem_percent = 0
        gpu_mem_details = "N/A"

        # Log to file
        with open(self.log_file, "a") as f:
            f.write(
                f"{timestamp},{cpu_percent},{ram_percent},{gpu_percent},{gpu_mem_percent},{ram_used:.2f},{ram_available:.2f},{gpu_mem_details}\n")


# ========== TRAINER ==========
class ModelTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base

        if config.cuda_available:
            self.device = torch.device("cuda")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.model = DialogueModel(config)
        self.model.to(self.device)

        if config.use_amp and config.cuda_available:
            self.scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)
        elif config.use_amp and not config.cuda_available:
            self.scaler = torch.amp.GradScaler(device_type='cpu', enabled=True)
        else:
            self.scaler = None

        dataset = DialogueDataset(knowledge_base, config)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn,
            pin_memory=True, num_workers=0  # Reduced for stability
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True, num_workers=0  # Reduced for stability
        )

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

        self.loss_history = []
        self.val_loss_history = []
        self.learning_rates = []
        self.resource_monitor = ResourceMonitor()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.resource_monitor.start()
        batch_count = 0

        if self.config.cuda_available:
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")):
            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            masks = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            knowledge = batch["knowledge_ids"].to(self.device, non_blocking=True)

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

            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            self.loss_history.append(loss.item())
            self.learning_rates.append(current_lr)
            batch_count += 1

            if (batch_idx + 1) % self.config.save_every == 0:
                self.save_model(f"checkpoint_epoch{epoch + 1}_batch{batch_idx + 1}")

            if (batch_idx + 1) % self.config.log_every == 0:
                self.resource_monitor.log_resources()

            if self.config.cuda_available:
                torch.cuda.empty_cache()

        avg_loss = total_loss / batch_count
        self.resource_monitor.stop()
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        self.resource_monitor.start()

        if self.config.cuda_available:
            torch.cuda.empty_cache()

        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validation"):
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                knowledge = batch["knowledge_ids"].to(self.device)

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

            if self.config.cuda_available:
                torch.cuda.empty_cache()

        avg_loss = total_loss / batch_count
        self.val_loss_history.append(avg_loss)
        self.resource_monitor.stop()
        return avg_loss

    def train(self):
        os.makedirs(self.config.final_model_dir, exist_ok=True)
        start_time = time.time()
        best_val_loss = float('inf')

        if self.config.cuda_available:
            try:
                test_tensor = torch.tensor([1.0]).cuda()
            except Exception:
                self.device = torch.device("cpu")
                self.model.to(self.device)
                self.config.use_amp = False
                if self.scaler:
                    self.scaler = None

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")

        self.save_model("final_model")
        training_time = time.time() - start_time

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
            "best_val_loss": best_val_loss
        }

        with open(os.path.join(self.config.final_model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self.plot_training_loss()

        return self.config.final_model_dir

    def save_model(self, prefix):
        model_path = os.path.join(self.config.final_model_dir, f"{prefix}_{self.config.model_file}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'config': self.config.__dict__
        }, model_path)
        print(f"Model saved: {model_path}")

    def plot_training_loss(self):
        if not self.loss_history or not self.val_loss_history:
            return

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.plot(self.loss_history, label='Training Loss', color='blue', linewidth=2)
        plt.title("Training Loss History")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 1, 2)
        plt.plot(self.val_loss_history, label='Validation Loss', color='red', linewidth=2, marker='o')
        plt.title("Validation Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(self.config.final_model_dir, "training_metrics.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Training metrics saved: {plot_path}")


# ========== DIALOGUE ENGINE ==========
class DialogueEngine:
    def __init__(self, model_path, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base

        if config.cuda_available and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = DialogueModel(config)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

        self.tokenizer = config.tokenizer

    def generate_response(self, query, max_length=150, temperature=0.7, top_p=0.9, top_k=50):
        try:
            context = self.knowledge_base.retrieve_context(query)
            knowledge_ids = self.tokenizer.encode(
                context,
                max_length=self.config.max_seq_length,
                truncation=True,
                add_special_tokens=False
            )
            knowledge_tensor = torch.tensor([knowledge_ids], dtype=torch.long).to(self.device)
            input_ids = self.tokenizer.encode(query, return_tensors="pt", add_special_tokens=False).to(
                self.device)

            with torch.no_grad():
                attention_mask = torch.ones(input_ids.shape, device=self.device)
                knowledge_embeds = self.model.embedding_layer(knowledge_tensor)
                knowledge_embeds = knowledge_embeds.mean(dim=1, keepdim=True)
                fused_knowledge = self.model.knowledge_fusion(knowledge_embeds)
                projected_knowledge = self.model.residual_projection(fused_knowledge)

                output = self.model.gpt2.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=self.model.embedding_layer(input_ids) + projected_knowledge,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response[len(query):].strip() if response.startswith(query) else response
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return "I encountered an error while generating a response."


# ========== MAIN ==========
def main():
    try:
        config = Config()

        print("\n[1/4] Loading knowledge base...")
        default_kb_path = "knowledge_base"
        if not os.path.exists(default_kb_path):
            os.makedirs(default_kb_path)
            print("Created default knowledge base directory")
            # Add sample knowledge file
            with open(os.path.join(default_kb_path, "sample.txt"), "w") as f:
                f.write("This is a sample knowledge base file.\n")
                f.write("Large language models are powerful AI systems.\n")
                f.write("They can understand and generate human-like text.\n")
            print("Added sample knowledge file")

        knowledge_base = KnowledgeBase(config)
        success, message = knowledge_base.load_files(default_kb_path)

        if not success:
            print(f"Error: {message}")
            return

        print("\n[2/4] Training dialogue model...")
        trainer = ModelTrainer(config, knowledge_base)
        model_dir = trainer.train()

        print("\n[3/4] Initializing dialogue engine...")
        model_path = os.path.join(model_dir, "final_model_dialogue_model.pth")
        engine = DialogueEngine(model_path, config, knowledge_base)

        print("\n[4/4] Starting conversation (type 'exit' to quit)")
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ['exit', 'quit', 'bye']:
                    break

                start_time = time.time()
                response = engine.generate_response(
                    query,
                    max_length=200,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=100
                )
                response_time = time.time() - start_time

                print(f"\nAssistant: {response}")
                print(f"Response time: {response_time:.2f}s")

            except KeyboardInterrupt:
                break
    except Exception as e:
        with open("fatal_error.log", "w") as f:
            f.write(f"Fatal error at {datetime.now()}\n")
            f.write(str(e))
            f.write("\n\n")
            f.write(traceback.format_exc())
        print(f"Fatal error occurred: {str(e)}")


if __name__ == "__main__":
    main()