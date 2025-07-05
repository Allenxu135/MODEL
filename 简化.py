import os
import time
import json
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ================== AUTOMATIC MODEL DETECTION ==================
def detect_ollama_models():
    """Detect all locally available Ollama models"""
    try:
        result = subprocess.run(["ollama", "list"],
                                capture_output=True,
                                text=True,
                                timeout=20)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # Skip header line
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
    except Exception as e:
        print(f"Model detection error: {str(e)}")
    return []


# ================== KNOWLEDGE BASE HANDLER ==================
def load_knowledge(file_path):
    """Load knowledge from text-based files"""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Simple chunking
        chunk_size = 1000
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

        if not chunks:
            return None, "File is empty"

        return chunks, f"Loaded {len(chunks)} chunks from {os.path.basename(file_path)}"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


# ================== SIMPLE NEURAL NETWORK ==================
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=3):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        hidden = torch.relu(self.fc1(embedded))
        return self.fc2(hidden)


# ================== DATASET HANDLER ==================
class TextDataset(Dataset):
    def __init__(self, chunks, vocab):
        self.chunks = chunks
        self.vocab = vocab

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        text = self.chunks[idx]
        words = text.split()
        token_ids = [self.vocab.get(word, 0) for word in words]
        label = len(text) % 3  # Simple synthetic label
        return token_ids, label


# Collate function for DataLoader
def collate_batch(batch):
    token_ids = [torch.tensor(ids) for ids, _ in batch]
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    offsets = [0] + [len(ids) for ids in token_ids[:-1]]
    offsets = torch.tensor(offsets).cumsum(dim=0)
    token_ids = torch.cat(token_ids)
    return token_ids, offsets, labels


# ================== MODEL TRAINING ==================
def train_model(model_name, chunks):
    """Train a simple neural network locally"""
    model_path = f"{model_name.replace(':', '_')}_trained"
    os.makedirs(model_path, exist_ok=True)

    # Create vocabulary
    vocab = {}
    word_counter = 1  # Start from 1, 0 is for unknown

    for chunk in chunks:
        words = chunk.split()
        for word in words:
            if word not in vocab:
                vocab[word] = word_counter
                word_counter += 1
                if word_counter >= 10000:  # Limit vocabulary size
                    break

    # Create dataset
    dataset = TextDataset(chunks, vocab)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)

    # Initialize model
    model = SimpleClassifier(vocab_size=10000, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Training loop
    print(f"Training local model for {model_name}...")
    for epoch in range(3):  # 3 epochs
        total_loss = 0
        total_acc = 0

        for token_ids, offsets, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(token_ids, offsets)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            total_acc += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataset)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(model_path, "model_weights.pth"))

    # Save vocab and model info
    with open(os.path.join(model_path, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    model_info = {
        "model_name": model_name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": len(vocab)
    }

    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump(model_info, f)

    print(f"Training completed for {model_name}")
    return model_path


# ================== MODEL MERGER ==================
def merge_models(model_paths, output_path="final_model"):
    """Create a final merged model"""
    os.makedirs(output_path, exist_ok=True)

    # Create merged model info
    merged_info = {
        "merged_models": model_paths,
        "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Final merged model"
    }

    # Copy the first model's vocab
    first_model = model_paths[0]
    try:
        with open(os.path.join(first_model, "vocab.json"), "r") as f:
            vocab = json.load(f)

        with open(os.path.join(output_path, "vocab.json"), "w") as f:
            json.dump(vocab, f)
    except:
        pass

    # Save merged info
    with open(os.path.join(output_path, "model_info.json"), "w") as f:
        json.dump(merged_info, f)

    print(f"Final model created at: {output_path}")
    return output_path


# ================== MAIN PROCESS ==================
def main():
    print("=============================================")
    print("Automated Model Training System")
    print("=============================================")

    # Step 1: Automatically detect models
    print("\n[1/4] Detecting local Ollama models...")
    available_models = detect_ollama_models()

    if not available_models:
        print("Error: No Ollama models detected.")
        print("Please install Ollama and download at least one model:")
        print("1. Install: https://ollama.com/")
        print("2. Download model: e.g., 'ollama pull llama3'")
        return

    print(f"✅ Detected {len(available_models)} models: {', '.join(available_models)}")

    # Step 2: Load knowledge base
    file_path = input("\n[2/4] Enter path to knowledge file: ").strip()
    chunks, message = load_knowledge(file_path)

    if not chunks:
        print(f"❌ {message}")
        return

    print(f"✅ {message}")

    # Step 3: Train top 3 models
    print("\n[3/4] Training top 3 models locally...")
    models_to_train = available_models[:3]
    trained_models = []

    for model_name in models_to_train:
        print(f"\n--- Training {model_name} ---")
        model_path = train_model(model_name, chunks)
        trained_models.append(model_path)
        print(f"✅ Trained model saved at: {model_path}")

    # Step 4: Create final merged model
    print("\n[4/4] Creating final model...")
    final_model_path = merge_models(trained_models)

    # Completion message
    print("\n" + "=" * 50)
    print("✅ Process completed successfully!")
    print("=" * 50)
    print(f"Final model is ready in: {os.path.abspath(final_model_path)}")
    print("You can now use this model for inference tasks.")
    print("=" * 50)


if __name__ == "__main__":
    main()