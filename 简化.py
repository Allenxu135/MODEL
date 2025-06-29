import os
import time
import numpy as np
import torch
import multiprocessing as mp
import psutil
import subprocess
import ollama
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import matplotlib.pyplot as plt
from datetime import datetime

# ================== GLOBAL SETTINGS ==================
APP_STRINGS = {
    "no_kb_msg": "Please import data first",
    "model_trained": "Model trained:",
    "training_completed": "Training completed! Model saved at:",
    "error_importing": "Error importing data:",
    "kb_imported": "Data imported successfully: {} text chunks",
    "training_model": "Training model: {}",
    "training_status": "Training status: Epoch {}/{} | Loss: {:.4f}",
    "model_loaded": "Model loaded",
    "no_models": "No models available",
    "install_models": "Please install models in Ollama",
    "ollama_unavailable": "Ollama service unavailable",
    "ollama_running": "Ollama: Running",
    "ollama_stopped": "Ollama: Not Running",
    "scanning_models": "Scanning local models...",
    "models_found": "Found {} models",
    "model_scan_failed": "Failed to scan models",
}


# ================== TEXT GENERATION MODEL ==================
class TextGenerationModel:
    def __init__(self):
        self.client = ollama.Client()
        self.available_models = []
        self.model_name = None
        self.generation_config = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        self.scan_local_models()

    def scan_local_models(self):
        """Scan for locally installed Ollama models"""
        self.available_models = []
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            self.available_models.append(model_name)

                if self.available_models:
                    self.model_name = self.available_models[0]

            return True
        except Exception as e:
            print(f"Error scanning local models: {str(e)}")
            return False

    def set_model(self, model_name):
        if model_name in self.available_models:
            self.model_name = model_name
            return True
        return False

    def load_model(self):
        if not self.model_name:
            return False
        return True

    def generate_text(self, prompt, context=None):
        if not self.model_name:
            return "No model available"

        full_prompt = context + "\n\n" + prompt if context else prompt

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options=self.generation_config
            )
            return response['response']
        except Exception as e:
            error_message = str(e)

        for model in self.available_models:
            if model == self.model_name:
                continue
            try:
                response = self.client.generate(
                    model=model,
                    prompt=full_prompt,
                    options=self.generation_config
                )
                self.model_name = model
                return response['response']
            except Exception:
                continue

        return f"Error: {error_message} | Available models: {', '.join(self.available_models)}"


# ================== CPU OPTIMIZATION SETTINGS ==================
class CPUSettings:
    def __init__(self):
        self.enable_binding = False
        self.cpu_cores = []
        self.use_half_precision = False
        self.cpu_threads = max(1, mp.cpu_count() - 2)
        self.enable_async = True
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.max_parallel_training = max(1, mp.cpu_count() // 2)

    def apply_cpu_binding(self):
        if self.enable_binding and self.cpu_cores:
            try:
                p = psutil.Process()
                p.cpu_affinity(self.cpu_cores)
            except Exception:
                pass


# ================== TRAINING MONITOR ==================
class TrainingMonitor:
    """Real-time training metrics monitoring and visualization"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f'Training Metrics - {model_name}')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.train_line, = self.ax.plot([], [], 'b-', label='Train Loss')
        self.val_line, = self.ax.plot([], [], 'r-', label='Validation Loss')
        self.ax.legend()
        plt.tight_layout()

        # Create output directory if needed
        os.makedirs("training_plots", exist_ok=True)

    def update_metrics(self, epoch, train_loss, val_loss=None):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)

        if val_loss is not None:
            self.val_loss.append(val_loss)

        # Update plot data
        self.train_line.set_data(self.epochs, self.train_loss)

        if val_loss is not None:
            self.val_line.set_data(self.epochs, self.val_loss)

        # Adjust axes limits
        self.ax.relim()
        self.ax.autoscale_view()

        # Save intermediate plot
        self.save_plot()

        # Display current plot
        plt.pause(0.01)

    def save_plot(self, final=False):
        filename = f"training_plots/{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if final:
            filename += "_final.png"
        else:
            filename += f"_epoch{len(self.epochs)}.png"

        self.fig.savefig(filename)
        print(f"Saved training plot: {filename}")

    def finalize(self):
        self.save_plot(final=True)
        plt.close(self.fig)


# ================== KNOWLEDGE BASE ==================
class KnowledgeBase:
    def __init__(self, cpu_settings):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.text_chunks = []
        self.cpu_settings = cpu_settings

    def import_data(self, file_paths):
        if not file_paths:
            return 0

        documents = []
        for path in file_paths:
            try:
                if path.endswith('.txt'):
                    loader = TextLoader(path, encoding='utf-8')
                    docs = loader.load()
                    if docs:
                        documents.extend(docs)
                elif path.endswith('.csv'):
                    loader = CSVLoader(path)
                    docs = loader.load()
                    if docs:
                        documents.extend(docs)
            except Exception as e:
                print(f"Error loading file {path}: {str(e)}")

        if not documents:
            return 0

        chunks = []
        for doc in documents:
            try:
                split_docs = self.text_splitter.split_documents([doc])
                if split_docs:
                    chunks.extend(split_docs)
            except Exception as e:
                print(f"Error splitting document: {str(e)}")

        self.text_chunks = [chunk.page_content for chunk in chunks]

        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            return len(chunks)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return 0

    def retrieve_context(self, query, k=5):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)


# ================== MODEL TRAINING ==================
class ModelTrainer:
    def __init__(self, model_name, knowledge_base, cpu_settings, epochs=5):
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.epochs = epochs
        self.cpu_settings = cpu_settings
        self.monitor = TrainingMonitor(model_name)
        self.cpu_settings.apply_cpu_binding()

    def train(self):
        try:
            if not self.knowledge_base.text_chunks:
                print(APP_STRINGS["no_kb_msg"])
                return None

            texts = self.knowledge_base.text_chunks
            labels = np.array([len(text) % 4 for text in texts], dtype=np.int64)

            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            torch_dtype = torch.float16 if self.cpu_settings.use_half_precision else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4,
                torch_dtype=torch_dtype
            )

            # Tokenization
            encodings = tokenizer(texts, truncation=True, padding=True,
                                  max_length=512, return_tensors="pt")

            class TextDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

                def __len__(self):
                    return len(self.labels)

            dataset = TextDataset(encodings, labels)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=self.epochs,
                per_device_train_batch_size=self.cpu_settings.batch_size,
                per_device_eval_batch_size=32,
                learning_rate=self.cpu_settings.learning_rate,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                dataloader_num_workers=self.cpu_settings.cpu_threads,
                fp16=self.cpu_settings.use_half_precision,
                no_cuda=True,
                logging_strategy="epoch",
                report_to="none"  # Disable external logging
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            # Training loop with progress monitoring
            print(f"Starting training for {self.model_name}...")
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}/{self.epochs}")

                # Train for one epoch
                train_metrics = trainer.train()
                train_loss = train_metrics.metrics.get("train_loss", 0)

                # Evaluate
                eval_metrics = trainer.evaluate()
                val_loss = eval_metrics.get("eval_loss", 0)

                # Update monitor
                self.monitor.update_metrics(epoch + 1, train_loss, val_loss)

                print(APP_STRINGS["training_status"].format(
                    self.model_name, epoch + 1, self.epochs, train_loss
                ))

            # Finalize training
            self.monitor.finalize()
            saved_model_path = self.save_model(model, tokenizer)
            print(APP_STRINGS["training_completed"].format(saved_model_path))
            return saved_model_path

        except Exception as e:
            print(f"Training error: {str(e)}")
            return None

    def save_model(self, model, tokenizer):
        if model and tokenizer:
            model_dir = f"./saved_models/{self.model_name}_{int(time.time())}"
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            return model_dir
        return ""


# ================== MAIN APPLICATION ==================
class DeepLearningStudio:
    def __init__(self):
        self.cpu_settings = CPUSettings()
        self.knowledge_base = KnowledgeBase(self.cpu_settings)
        self.text_generator = TextGenerationModel()
        self.trained_models = {}

    def check_ollama_service(self):
        try:
            self.text_generator.client.list()
            print(APP_STRINGS["ollama_running"])
            return True
        except Exception:
            print(APP_STRINGS["ollama_stopped"])
            return False

    def scan_local_models(self):
        print(APP_STRINGS["scanning_models"])
        if self.text_generator.scan_local_models():
            models = self.text_generator.available_models
            if models:
                print(APP_STRINGS["models_found"].format(len(models)))
                print("Available models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                return models
            else:
                print(APP_STRINGS["no_models"])
                print(APP_STRINGS["install_models"])
        else:
            print(APP_STRINGS["model_scan_failed"])
        return []

    def import_knowledge(self, file_paths):
        if not file_paths:
            return False

        chunk_count = self.knowledge_base.import_data(file_paths)
        if chunk_count > 0:
            print(APP_STRINGS["kb_imported"].format(chunk_count))
            return True
        else:
            print(APP_STRINGS["error_importing"])
            return False

    def train_model(self, model_name, epochs=5):
        if not self.knowledge_base.vector_store:
            print(APP_STRINGS["no_kb_msg"])
            return False

        trainer = ModelTrainer(model_name, self.knowledge_base, self.cpu_settings, epochs)
        model_path = trainer.train()
        if model_path:
            self.trained_models[model_name] = model_path
            return True
        return False

    def interact_with_model(self, model_name, prompt):
        if not self.text_generator.set_model(model_name):
            print(f"Model '{model_name}' not available")
            return

        context = None
        if self.knowledge_base.vector_store:
            context_chunks = self.knowledge_base.retrieve_context(prompt, k=3)
            context = "\n".join([c.page_content for c in context_chunks])

        response = self.text_generator.generate_text(prompt, context)
        print("\n=== Model Response ===")
        print(response)
        print("======================")


# ================== COMMAND LINE INTERFACE ==================
def main_menu(studio):
    while True:
        print("\n===== Deep Learning Studio =====")
        print("1. Scan for available models")
        print("2. Import knowledge base data")
        print("3. Train a model")
        print("4. Interact with a model")
        print("5. Check Ollama service status")
        print("6. Exit")

        choice = input("Select an option: ").strip()

        if choice == "1":
            studio.scan_local_models()

        elif choice == "2":
            paths = input("Enter file paths (comma separated): ").split(',')
            cleaned_paths = [p.strip() for p in paths if p.strip()]
            if cleaned_paths:
                studio.import_knowledge(cleaned_paths)
            else:
                print("No valid paths provided")

        elif choice == "3":
            model_name = input("Enter model name to train: ").strip()
            if model_name:
                try:
                    epochs = int(input("Number of epochs (default 5): ") or 5)
                except ValueError:
                    epochs = 5
                studio.train_model(model_name, epochs)
            else:
                print("Model name required")

        elif choice == "4":
            model_name = input("Enter model name: ").strip()
            if model_name:
                prompt = input("Enter your prompt: ").strip()
                if prompt:
                    studio.interact_with_model(model_name, prompt)
                else:
                    print("Prompt cannot be empty")
            else:
                print("Model name required")

        elif choice == "5":
            studio.check_ollama_service()

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice, please try again")


if __name__ == "__main__":
    print("Initializing Deep Learning Studio...")
    studio = DeepLearningStudio()

    # Check Ollama status on startup
    studio.check_ollama_service()

    # Start main menu
    main_menu(studio)