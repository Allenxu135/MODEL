import os
import time
import json
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import psutil
import GPUtil
import tqdm
from datetime import datetime
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_ollama import OllamaEmbeddings
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# Use PyTorch's AdamW instead of transformers'
from torch.optim import AdamW


# ======== Helper function to check library installation ========
def is_library_installed(lib_name):
    try:
        __import__(lib_name)
        return True
    except ImportError:
        return False


# ================== CONFIGURATION ==================
class Config:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = "nomic-embed-text"
        self.epochs = 3
        self.batch_size = 8
        self.learning_rate = 5e-5
        self.hidden_size = 512
        self.embedding_dim = 256
        self.rag_top_k = 3
        self.final_model_dir = "final_dialogue_model"
        self.vocab_size = 50260  # GPT-2 tokenizer size
        self.max_seq_length = 256
        self.model_file = "dialogue_model.pth"
        self.knowledge_base_dir = "knowledge_base"
        self.use_gpt2 = True  # Use GPT-2 as base model
        self.save_every = 500  # Save model every 500 steps
        self.log_every = 100  # Log every 100 steps

        # Automatically detect GPT-2 paths
        self.gpt2_model_path = self.detect_gpt2_path("model")
        self.gpt2_tokenizer_path = self.detect_gpt2_path("tokenizer")

        print(f"Using GPT-2 model path: {self.gpt2_model_path}")
        print(f"Using GPT-2 tokenizer path: {self.gpt2_tokenizer_path}")

    def detect_gpt2_path(self, path_type):
        """Intelligent GPT-2 path detection handling both files and directories"""
        # 1. Check environment variables
        env_var = f"GPT2_{path_type.upper()}_PATH"
        if env_var in os.environ:
            env_path = os.environ[env_var]
            if self.is_valid_gpt2_path(env_path, path_type):
                return env_path

        # 2. Check common locations
        common_paths = [
            os.path.join(os.getcwd(), "gpt2", path_type),
            os.path.join(os.path.expanduser("~"), "gpt2", path_type),
            os.path.join(os.path.expanduser("~"), "models", "gpt2", path_type),
            f"/usr/share/gpt2/{path_type}",
            f"/opt/gpt2/{path_type}",
            f"C:\\gpt2\\{path_type}",
            f"D:\\gpt2\\{path_type}",
            f"E:\\gpt2\\{path_type}",
            "gpt2_model.bin",
            "gpt2_tokenizer.bin",
            "pytorch_model.bin",
            "tokenizer.json"
        ]

        for path in common_paths:
            if self.is_valid_gpt2_path(path, path_type):
                return path

        # 3. Scan all drives for GPT-2 files
        scan_result = self.scan_drives_for_gpt2(path_type)
        if scan_result:
            return scan_result

        # 4. Create default directory as fallback
        default_path = os.path.join(os.getcwd(), f"gpt2_{path_type}")
        os.makedirs(default_path, exist_ok=True)
        print(f"Warning: Using default {path_type} path: {default_path}")
        return default_path

    def is_valid_gpt2_path(self, path, path_type):
        """Validate if path is a valid GPT-2 file or directory"""
        if not os.path.exists(path):
            return False

        # Define required files for each type
        model_files = {"pytorch_model.bin", "config.json"}
        tokenizer_files = {"tokenizer.json", "vocab.json", "merges.txt"}

        if os.path.isfile(path):
            # Handle single file case
            if path_type == "model":
                return path.endswith((".bin", ".pt", ".pth", ".ckpt"))
            else:  # tokenizer
                return path.endswith((".json", ".txt", ".model"))
        else:
            # Handle directory case
            dir_files = set(os.listdir(path))
            if path_type == "model":
                return model_files.issubset(dir_files)
            else:  # tokenizer
                return tokenizer_files.issubset(dir_files)

    def scan_drives_for_gpt2(self, path_type):
        """Efficiently scan all drives for GPT-2 files and directories"""
        # Get all drives
        drives = self.get_all_drives()

        # Define search patterns for both directories and files
        dir_patterns = ["gpt2", "gpt-2", "gpt_2", "transformers"]
        file_patterns = {
            "model": ["pytorch_model.bin", "gpt2_model", ".bin", ".pt", ".pth", ".ckpt"],
            "tokenizer": ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model"]
        }

        for drive in drives:
            print(f"Scanning {drive} for GPT-2 {path_type}...")
            try:
                for root, dirs, files in os.walk(drive):
                    # Skip system directories
                    if any(x in root.lower() for x in ["windows", "program files", "system32", "$recycle.bin"]):
                        continue

                    # Check directories
                    if any(term in root.lower() for term in dir_patterns):
                        if self.is_valid_gpt2_path(root, path_type):
                            print(f"Found valid GPT-2 {path_type} directory: {root}")
                            return root

                    # Check individual files
                    for file in files:
                        if any(pattern in file.lower() for pattern in file_patterns[path_type]):
                            file_path = os.path.join(root, file)
                            if self.is_valid_gpt2_path(file_path, path_type):
                                print(f"Found valid GPT-2 {path_type} file: {file_path}")
                                return file_path
            except Exception as e:
                print(f"Scan interrupted on {drive}: {str(e)}")

        return None

    def get_all_drives(self):
        """Get all available drives in the system"""
        drives = []
        if os.name == 'nt':  # Windows
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
        else:  # Linux/Mac
            drives = ["/"]  # Start from root directory
            for mount in os.listdir("/mnt"):
                drives.append(os.path.join("/mnt", mount))
            for mount in os.listdir("/media"):
                drives.append(os.path.join("/media", mount))
            for mount in os.listdir("/Volumes"):
                drives.append(os.path.join("/Volumes", mount))
        return drives


# ================== AUTOMATIC MODEL DETECTION ==================
def detect_ollama_models():
    """Detect all locally available Ollama models with retry logic"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
        except Exception as e:
            print(f"Model detection error (attempt {attempt + 1}): {str(e)}")
        time.sleep(2)
    return []


# ================== KNOWLEDGE BASE & RAG ==================
class KnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.chunks = []
        self.vector_store = None

        # Load tokenizer from detected path
        print(f"Loading tokenizer from: {config.gpt2_tokenizer_path}")

        # Handle both file and directory paths
        if os.path.isfile(config.gpt2_tokenizer_path):
            # If it's a single file, load from file
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                os.path.dirname(config.gpt2_tokenizer_path),
                tokenizer_file=os.path.basename(config.gpt2_tokenizer_path)
            )
        else:
            # If it's a directory
            self.tokenizer = GPT2Tokenizer.from_pretrained(config.gpt2_tokenizer_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded successfully")

    def load_files(self, directory_path):
        """Load and process multiple knowledge files"""
        if not os.path.exists(directory_path):
            return False, f"Directory not found: {directory_path}"

        try:
            # Load all text files in directory
            loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()

            if not documents:
                return False, "No text files found in directory"

            # Split text
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = splitter.split_documents(documents)
            self.chunks = [doc.page_content for doc in split_docs]

            # Create vector store
            embeddings = OllamaEmbeddings(model=self.config.embedding_model)
            self.vector_store = FAISS.from_documents(split_docs, embeddings)

            return True, f"Loaded {len(self.chunks)} chunks from {len(documents)} files"
        except Exception as e:
            return False, f"Error loading files: {str(e)}"

    def retrieve_context(self, query, k=None):
        """Retrieve relevant context using RAG"""
        if k is None:
            k = self.config.rag_top_k

        if not self.vector_store:
            return ""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join([r.page_content for r in results])
        except:
            return ""

    def encode_text(self, text):
        """Convert text to token IDs using GPT-2 tokenizer"""
        return self.tokenizer.encode(text, max_length=self.config.max_seq_length, truncation=True)


# ================== ADVANCED DIALOGUE MODEL ==================
class DialogueModel(nn.Module):
    def __init__(self, config):
        super(DialogueModel, self).__init__()
        self.config = config

        # Load GPT-2 model from detected path
        print(f"Loading GPT-2 model from: {config.gpt2_model_path}")

        # Handle both file and directory paths
        if os.path.isfile(config.gpt2_model_path):
            # If it's a single file, load using torch
            model_state = torch.load(config.gpt2_model_path)
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
            self.gpt2.load_state_dict(model_state)
        else:
            # If it's a directory
            self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path)

        self.gpt2.resize_token_embeddings(config.vocab_size)
        print("GPT-2 model loaded successfully")

        # Knowledge fusion layer
        self.knowledge_projection = nn.Linear(768, self.gpt2.config.n_embd)
        self.learning_rate = config.learning_rate

    def forward(self, input_ids, attention_mask=None, knowledge_embeddings=None):
        # If knowledge embeddings exist, fuse with input
        if knowledge_embeddings is not None:
            projected_knowledge = self.knowledge_projection(knowledge_embeddings)
            if projected_knowledge.dim() == 2:
                projected_knowledge = projected_knowledge.unsqueeze(1).repeat(1, input_ids.size(1), 1)
            inputs_embeds = self.gpt2.transformer.wte(input_ids)
            inputs_embeds = inputs_embeds + projected_knowledge
            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            return self.gpt2(input_ids, attention_mask=attention_mask)


# ================== DIALOGUE DATASET ==================
class DialogueDataset(Dataset):
    def __init__(self, knowledge_base, config):
        self.knowledge_base = knowledge_base
        self.config = config
        self.examples = []
        self.tokenizer = knowledge_base.tokenizer
        self.add_knowledge_dialogue_examples()

    def add_knowledge_dialogue_examples(self):
        """Create dialogue examples from knowledge base"""
        for chunk in self.knowledge_base.chunks:
            questions = [
                f"What is this about: {chunk[:50]}...?",
                f"Can you explain: {chunk[:50]}...?",
                f"Summarize this: {chunk[:50]}...",
                f"Tell me more about: {chunk[:50]}...",
                f"What does this mean: {chunk[:50]}...?"
            ]

            answers = [
                f"Based on my knowledge: {chunk[:200]}",
                f"I found this information: {chunk[:200]}",
                f"Here's what I know: {chunk[:200]}",
                f"According to the knowledge base: {chunk[:200]}",
                f"The relevant information is: {chunk[:200]}"
            ]

            for q, a in zip(questions, answers):
                self.examples.append({
                    "input": q,
                    "output": a,
                    "knowledge": chunk
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = self.tokenizer.encode(
            example["input"], max_length=self.config.max_seq_length, truncation=True
        )
        output_ids = self.tokenizer.encode(
            example["output"], max_length=self.config.max_seq_length, truncation=True
        )
        knowledge_ids = self.tokenizer.encode(
            example["knowledge"], max_length=self.config.max_seq_length, truncation=True
        ) if example["knowledge"] else []

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "output_ids": torch.tensor(output_ids, dtype=torch.long),
            "knowledge_ids": torch.tensor(knowledge_ids, dtype=torch.long)
        }


# ================== COLLATE FUNCTION ==================
def collate_fn(batch):
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    output_ids = pad_sequence([item["output_ids"] for item in batch], batch_first=True, padding_value=0)
    knowledge_ids = pad_sequence([item["knowledge_ids"] for item in batch], batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).int(),
        "output_ids": output_ids,
        "knowledge_ids": knowledge_ids
    }


# ================== UNIFIED MODEL TRAINER ==================
class UnifiedModelTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base
        self.model = DialogueModel(config)
        self.monitor = ResourceMonitor()
        self.training_metrics = {"loss": [], "epoch_times": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Prepare dataset
        self.dataset = DialogueDataset(knowledge_base, config)
        train_size = int(0.9 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size,
            shuffle=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size,
            collate_fn=collate_fn
        )

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader) * config.epochs
        )

    def train(self):
        os.makedirs(self.config.final_model_dir, exist_ok=True)
        start_time = time.time()
        global_step = 0
        best_val_loss = float('inf')

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            total_loss = 0
            total_tokens = 0
            self.model.train()

            train_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]")
            for batch in train_bar:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                outputs = batch["output_ids"].to(self.device)
                knowledge = batch["knowledge_ids"].to(self.device)

                model_output = self.model(input_ids=inputs, attention_mask=masks, knowledge_embeddings=knowledge)

                shift_logits = model_output.logits[..., :-1, :].contiguous()
                shift_labels = outputs[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                current_lr = self.scheduler.get_last_lr()[0]
                tokens_processed = inputs.nelement()
                total_loss += loss.item()
                total_tokens += tokens_processed

                self.monitor.record(tokens_processed, loss.item(), current_lr)
                global_step += 1

                train_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

                if global_step % self.config.log_every == 0:
                    self.update_training_plot(ax1, ax2, epoch, global_step)
                if global_step % self.config.save_every == 0:
                    self.save_checkpoint(global_step, epoch)

            val_loss = self.validate()
            self.training_metrics["loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")

            epoch_time = time.time() - epoch_start
            self.training_metrics["epoch_times"].append(epoch_time)
            print(f"Epoch {epoch + 1}/{self.config.epochs} completed: "
                  f"Train Loss={total_loss / len(self.train_loader):.4f}, "
                  f"Val Loss={val_loss:.4f}, Time={epoch_time:.2f}s")

        training_time = time.time() - start_time
        total_tokens = sum(self.monitor.token_counts)
        self.save_model("final_model")

        metadata = {
            "trained_at": datetime.now().isoformat(),
            "training_time": training_time,
            "total_tokens": total_tokens,
            "resource_report": self.monitor.generate_report(training_time, total_tokens),
            "training_metrics": self.training_metrics,
            "config": self.config.__dict__,
            "model_file": os.path.join(self.config.final_model_dir, self.config.model_file)
        }

        with open(os.path.join(self.config.final_model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self.generate_final_plots()
        plt.ioff()
        plt.close()

        return self.config.final_model_dir, metadata

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_bar = tqdm.tqdm(self.val_loader, desc="Validation")
            for batch in val_bar:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)
                outputs = batch["output_ids"].to(self.device)
                knowledge = batch["knowledge_ids"].to(self.device)

                model_output = self.model(input_ids=inputs, attention_mask=masks, knowledge_embeddings=knowledge)

                shift_logits = model_output.logits[..., :-1, :].contiguous()
                shift_labels = outputs[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss(ignore_index=0)(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                total_val_loss += loss.item()
                val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        return total_val_loss / len(self.val_loader)

    def update_training_plot(self, ax1, ax2, epoch, step):
        if not self.monitor.loss_values: return

        ax1.clear()
        ax2.clear()
        steps = range(1, len(self.monitor.loss_values) + 1)

        ax1.plot(steps, self.monitor.loss_values, 'b-', label='Training Loss')
        ax1.set_title(f"Epoch {epoch + 1} - Step {step}")
        ax1.set_xlabel("Training Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        if self.monitor.learning_rates:
            ax2.plot(steps, self.monitor.learning_rates, 'r-', label='Learning Rate')
            ax2.set_xlabel("Training Steps")
            ax2.set_ylabel("Learning Rate")
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def save_checkpoint(self, step, epoch):
        checkpoint_path = os.path.join(self.config.final_model_dir, f"checkpoint_epoch{epoch}_step{step}.pth")
        torch.save({
            'step': step, 'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.monitor.loss_values[-1] if self.monitor.loss_values else 0
        }, checkpoint_path)

    def save_model(self, prefix):
        model_path = os.path.join(self.config.final_model_dir, f"{prefix}_{self.config.model_file}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.knowledge_base.tokenizer,
            'config': self.config.__dict__
        }, model_path)

    def generate_final_plots(self):
        if not self.monitor.loss_values: return

        plt.figure(figsize=(10, 6))
        steps = range(1, len(self.monitor.loss_values) + 1)
        plt.plot(steps, self.monitor.loss_values, 'b-', label='Training Loss')
        plt.title("Training Loss", fontsize=14)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.final_model_dir, "training_loss.png"))
        plt.close()

        if self.monitor.learning_rates:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, self.monitor.learning_rates, 'r-', label='Learning Rate')
            plt.title("Learning Rate Schedule", fontsize=14)
            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.final_model_dir, "learning_rate.png"))
            plt.close()


# ================== RESOURCE MONITOR ==================
class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.token_counts = []
        self.loss_values = []
        self.learning_rates = []

    def record(self, tokens_processed=0, loss_value=None, lr=None):
        self.timestamps.append(time.time())
        self.cpu_usage.append(psutil.cpu_percent())
        self.token_counts.append(tokens_processed)
        if loss_value is not None:
            self.loss_values.append(loss_value)
        if lr is not None:
            self.learning_rates.append(lr)

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(gpus[0].load * 100)
                self.gpu_memory.append(gpus[0].memoryUsed)
            else:
                self.gpu_usage.append(0)
                self.gpu_memory.append(0)
        except:
            self.gpu_usage.append(0)
            self.gpu_memory.append(0)

    def generate_report(self, training_time, total_tokens):
        return {
            "training_time": training_time,
            "total_tokens": total_tokens,
            "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_gpu": np.mean(self.gpu_usage) if self.gpu_usage else 0,
            "max_gpu": np.max(self.gpu_usage) if self.gpu_usage else 0,
            "max_gpu_mem": np.max(self.gpu_memory) if self.gpu_memory else 0,
            "min_loss": np.min(self.loss_values) if self.loss_values else 0,
            "final_loss": self.loss_values[-1] if self.loss_values else 0
        }


# ================== DIALOGUE ENGINE ==================
class DialogueEngine:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        config = Config()
        for key, value in checkpoint['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.tokenizer = checkpoint['tokenizer']
        self.model = DialogueModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, query, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
        input_ids = self.tokenizer.encode(query, return_tensors="pt").to(self.device)
        output = self.model.gpt2.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(query):].strip() if response.startswith(query) else response


# ================== MAIN PROCESS ==================
def main():
    print("=============================================")
    print("Advanced Unified Dialogue Model Trainer")
    print("=============================================")

    config = Config()

    # Step 1: Detect Ollama models
    print("\n[1/4] Detecting local Ollama models...")
    available_models = detect_ollama_models()
    print(f"Detected {len(available_models)} models" if available_models else "No Ollama models detected")

    # Step 2: Load knowledge base
    print("\n[2/4] Loading knowledge base...")
    dir_path = input("Enter path to knowledge directory: ").strip()
    knowledge_base = KnowledgeBase(config)
    success, message = knowledge_base.load_files(dir_path)

    if not success:
        print(f"Error: {message}")
        print("\nTroubleshooting tips:")
        print("1. Verify directory exists and contains text files")
        print("2. Ensure Ollama is running: ollama serve")
        print("3. Install embedding model: ollama pull nomic-embed-text")
        return

    print(f"Success: {message}")

    # Step 3: Train model
    print("\n[3/4] Training unified dialogue model...")
    trainer = UnifiedModelTrainer(config, knowledge_base)
    model_dir, metadata = trainer.train()
    model_file = os.path.join(model_dir, "final_model_dialogue_model.pth")

    # Step 4: Initialize dialogue engine
    print("\n[4/4] Initializing dialogue engine...")
    engine = DialogueEngine(model_file)

    print("\n" + "=" * 60)
    print("✅ Advanced Dialogue Model Ready!")
    print("=" * 60)
    print(f"Model file: {os.path.abspath(model_file)}")

    # Interactive dialogue
    print("\nStarting conversation (type 'exit' to end)...\n")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']: break

            context = knowledge_base.retrieve_context(query)
            start_time = time.time()
            response = engine.generate_response(query, temperature=0.7, top_p=0.85, max_length=150)
            response_time = time.time() - start_time

            print(f"\nAssistant: {response}")
            if context: print(f"\n[Knowledge Context]: {context[:200]}...")
            print(f"Response time: {response_time:.2f}s\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    required_libs = ['transformers', 'torch', 'langchain', 'tqdm']
    missing_libs = [lib for lib in required_libs if not is_library_installed(lib)]

    if missing_libs:
        print(f"Please install required libraries: {', '.join(missing_libs)}")
        print("You can install them with: pip install transformers torch langchain tqdm")
        exit(1)

    main()