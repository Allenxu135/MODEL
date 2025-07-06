import os
import time
import json
import subprocess
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import psutil
import GPUtil
from datetime import datetime
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings


# ================== CONFIGURATION ==================
class Config:
    def __init__(self):
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = "nomic-embed-text"
        self.epochs = 3
        self.batch_size = 8
        self.learning_rate = 0.001
        self.hidden_size = 256
        self.embedding_dim = 128
        self.rag_top_k = 3
        self.final_model_dir = "final_dialogue_model"
        self.vocab_size = 10000
        self.max_seq_length = 100


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
        self.tokenizer = None

    def load_file(self, file_path):
        """Load and process knowledge file"""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        try:
            loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            split_docs = splitter.split_documents(documents)
            self.chunks = [doc.page_content for doc in split_docs]

            # Create vocabulary
            self._build_vocabulary()

            # Create vector store for RAG
            embeddings = OllamaEmbeddings(model=self.config.embedding_model)
            self.vector_store = FAISS.from_documents(split_docs, embeddings)

            return True, f"Loaded {len(self.chunks)} chunks from {os.path.basename(file_path)}"
        except Exception as e:
            return False, f"Error loading file: {str(e)}"

    def _build_vocabulary(self):
        """Build vocabulary from knowledge base"""
        word_counts = Counter()
        for chunk in self.chunks:
            words = chunk.split()
            word_counts.update(words)

        # Get most common words
        most_common = word_counts.most_common(self.config.vocab_size - 2)

        # Create vocabulary dictionary
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for idx, (word, count) in enumerate(most_common):
            self.vocab[word] = idx + 2

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
        """Convert text to token IDs"""
        words = text.split()
        token_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        return token_ids


# ================== RESOURCE MONITOR ==================
class ResourceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.timestamps = []
        self.token_counts = []

    def record(self, tokens_processed=0):
        """Record current resource usage"""
        self.timestamps.append(time.time())
        self.cpu_usage.append(psutil.cpu_percent())
        self.token_counts.append(tokens_processed)

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
        """Generate resource usage report"""
        return {
            "training_time": training_time,
            "total_tokens": total_tokens,
            "avg_cpu": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu": np.max(self.cpu_usage) if self.cpu_usage else 0,
            "avg_gpu": np.mean(self.gpu_usage) if self.gpu_usage else 0,
            "max_gpu": np.max(self.gpu_usage) if self.gpu_usage else 0,
            "max_gpu_mem": np.max(self.gpu_memory) if self.gpu_memory else 0
        }


# ================== CUSTOM LSTM MODEL ==================
class DialogueLSTM(nn.Module):
    def __init__(self, config, vocab_size):
        super(DialogueLSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # Embedding layer
        embedded = self.embedding(x)

        # LSTM layer
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Fully connected layer
        output = self.fc(lstm_out)
        return output, hidden


# ================== DIALOGUE DATASET ==================
class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, knowledge_base, config):
        self.knowledge_base = knowledge_base
        self.config = config
        self.examples = []

        # Create dialogue examples
        for chunk in knowledge_base.chunks:
            # Create a simple Q&A format
            self.examples.append(f"Knowledge: {chunk}\n\nQuestion: What is this about?\nAnswer: {chunk[:200]}")
            self.examples.append(f"Knowledge: {chunk}\n\nQuestion: Summarize this\nAnswer: {chunk[:150]}...")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        token_ids = self.knowledge_base.encode_text(text)

        # Pad or truncate sequence
        if len(token_ids) > self.config.max_seq_length:
            token_ids = token_ids[:self.config.max_seq_length]
        else:
            token_ids = token_ids + [0] * (self.config.max_seq_length - len(token_ids))

        # Convert to PyTorch tensors
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_ids[1:], dtype=torch.long)

        return input_ids, target_ids


# ================== UNIFIED MODEL TRAINER ==================
class UnifiedModelTrainer:
    def __init__(self, config, knowledge_base):
        self.config = config
        self.knowledge_base = knowledge_base
        self.vocab_size = len(knowledge_base.vocab)
        self.model = DialogueLSTM(config, self.vocab_size)
        self.monitor = ResourceMonitor()
        self.training_metrics = {
            "loss": [],
            "epoch_times": []
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        """Train a unified dialogue model with RAG-enhanced knowledge"""
        # Prepare dataset
        dataset = DialogueDataset(self.knowledge_base, self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Training loop
        print(f"Training unified dialogue model with {len(dataset)} examples...")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            total_loss = 0
            total_tokens = 0

            self.model.train()
            for inputs, targets in dataloader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs, _ = self.model(inputs)

                # Calculate loss
                loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                tokens_processed = inputs.nelement()
                total_tokens += tokens_processed
                self.monitor.record(tokens_processed)

            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / len(dataloader)

            # Record metrics
            self.training_metrics["loss"].append(avg_loss)
            self.training_metrics["epoch_times"].append(epoch_time)

            print(f"Epoch {epoch + 1}/{self.config.epochs}: "
                  f"Loss={avg_loss:.4f}, Time={epoch_time:.2f}s, "
                  f"Tokens={total_tokens}")

        # Finalize training
        training_time = time.time() - start_time
        total_tokens = sum(self.monitor.token_counts)

        # Save final model
        os.makedirs(self.config.final_model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.final_model_dir, "model_weights.pth"))

        # Save vocabulary and metadata
        with open(os.path.join(self.config.final_model_dir, "vocab.json"), "w") as f:
            json.dump(self.knowledge_base.vocab, f)

        metadata = {
            "trained_at": datetime.now().isoformat(),
            "training_time": training_time,
            "total_tokens": total_tokens,
            "resource_report": self.monitor.generate_report(training_time, total_tokens),
            "training_metrics": self.training_metrics,
            "config": {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.config.embedding_dim,
                "hidden_size": self.config.hidden_size
            }
        }

        with open(os.path.join(self.config.final_model_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Generate training curves
        self.generate_training_curves()
        self.generate_resource_plots()

        return self.config.final_model_dir, metadata

    def generate_training_curves(self):
        """Generate training loss curve"""
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.training_metrics["loss"]) + 1)

        plt.plot(epochs, self.training_metrics["loss"], 'b-o')
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.final_model_dir, "training_loss.png"))
        plt.close()

    def generate_resource_plots(self):
        """Generate resource usage plots"""
        if not self.monitor.timestamps:
            return

        plt.figure(figsize=(12, 10))
        timestamps = [t - self.monitor.timestamps[0] for t in self.monitor.timestamps]

        # CPU Usage
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, self.monitor.cpu_usage, 'r-')
        plt.title("CPU Usage (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU %")
        plt.grid(True)

        # GPU Usage
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, self.monitor.gpu_usage, 'b-')
        plt.title("GPU Utilization (%)")
        plt.xlabel("Time (s)")
        plt.ylabel("GPU %")
        plt.grid(True)

        # Token Processing
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, self.monitor.token_counts, 'g-')
        plt.title("Tokens Processed")
        plt.xlabel("Time (s)")
        plt.ylabel("Tokens")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.final_model_dir, "resource_usage.png"))
        plt.close()


# ================== DIALOGUE ENGINE ==================
class DialogueEngine:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load vocabulary
        with open(os.path.join(model_dir, "vocab.json"), "r") as f:
            self.vocab = json.load(f)

        # Reverse vocabulary for decoding
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}

        # Load model config
        with open(os.path.join(model_dir, "training_metadata.json"), "r") as f:
            metadata = json.load(f)
            config = metadata["config"]

        # Initialize model
        self.model = DialogueLSTM(
            config=Config(),
            vocab_size=config["vocab_size"]
        )
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pth")))
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, query, max_length=50, temperature=0.7):
        """Generate response using knowledge-enhanced dialogue"""
        # Encode query
        token_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in query.split()]

        # Convert to tensor
        inputs = torch.tensor([token_ids], dtype=torch.long).to(self.device)

        # Generate response
        hidden = None
        response_tokens = []

        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                output, hidden = self.model(inputs, hidden)

                # Get last token prediction
                logits = output[0, -1, :]

                # Apply temperature
                logits = logits / temperature

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, 1).item()

                # Stop if end token is generated
                if next_token == self.vocab["<PAD>"]:
                    break

                # Add token to response
                response_tokens.append(next_token)
                inputs = torch.tensor([[next_token]], dtype=torch.long).to(self.device)

        # Convert tokens to words
        response_words = [self.idx_to_word.get(token, "<UNK>") for token in response_tokens]
        return " ".join(response_words)


# ================== MAIN PROCESS ==================
def main():
    print("=============================================")
    print("Unified Dialogue Model Trainer")
    print("=============================================")

    # Initialize configuration
    config = Config()

    # Step 1: Detect Ollama models (for RAG)
    print("\n[1/4] Detecting local Ollama models for RAG...")
    available_models = detect_ollama_models()

    if available_models:
        print(f"✅ Detected {len(available_models)} models")
    else:
        print("⚠️ No Ollama models detected - RAG will be limited")

    # Step 2: Load knowledge base
    print("\n[2/4] Loading knowledge base...")
    file_path = input("Enter path to knowledge file: ").strip()
    knowledge_base = KnowledgeBase(config)
    success, message = knowledge_base.load_file(file_path)

    if not success:
        print(f"❌ {message}")
        print("\nTroubleshooting tips:")
        print("1. Verify file exists and is readable")
        print("2. Check file encoding (try UTF-8)")
        print("3. Ensure Ollama is running: ollama serve")
        print("4. Install embedding model: ollama pull nomic-embed-text")
        return

    print(f"✅ {message}")

    # Step 3: Train unified dialogue model
    print("\n[3/4] Training unified dialogue model...")
    trainer = UnifiedModelTrainer(config, knowledge_base)
    model_dir, metadata = trainer.train()

    print(f"✅ Training completed! Model saved at: {model_dir}")
    print(f"⏱️ Training time: {metadata['training_time']:.2f} seconds")
    print(f"🔤 Tokens processed: {metadata['total_tokens']}")

    # Step 4: Initialize dialogue engine
    print("\n[4/4] Initializing dialogue engine...")
    engine = DialogueEngine(model_dir)

    # Completion message
    print("\n" + "=" * 60)
    print("✅ Dialogue Model Ready!")
    print("=" * 60)
    print(f"Model path: {os.path.abspath(model_dir)}")
    print("You can now have conversations with the trained model.")

    # Interactive conversation
    print("\nStarting conversation (type 'exit' to end)...\n")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ['exit', 'quit']:
                break

            start_time = time.time()
            response = engine.generate_response(query)
            response_time = time.time() - start_time

            print(f"\nAssistant: {response}")
            print(f"Response time: {response_time:.2f}s\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Verify langchain-ollama is installed
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        print("Please install langchain-ollama: pip install langchain-ollama")
        exit(1)

    # Run main process
    main()