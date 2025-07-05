import os
import time
import torch
import numpy as np
import ollama
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ================== GLOBAL SETTINGS ==================
DEFAULT_PARAMS = {
    "epochs": 3,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "max_length": 512,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "temperature": 0.7,
}


# ================== KNOWLEDGE BASE HANDLER ==================
class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.text_chunks = []

    def import_data(self, file_path):
        """Import data from various file formats (txt, csv, pdf, docx)"""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"

        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                return False, "Unsupported file format"

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_PARAMS["chunk_size"],
                chunk_overlap=DEFAULT_PARAMS["chunk_overlap"]
            )
            chunks = splitter.split_documents(docs)
            self.text_chunks = [chunk.page_content for chunk in chunks]

            # Create vector store for context retrieval
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vector_store = FAISS.from_documents(chunks, embeddings)

            return True, f"Imported {len(chunks)} chunks from {os.path.basename(file_path)}"

        except Exception as e:
            return False, f"Import error: {str(e)}"

    def get_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        if not self.vector_store:
            return ""
        results = self.vector_store.similarity_search(query, k=k)
        return "\n".join([r.page_content for r in results])


# ================== MODEL DETECTOR ==================
class ModelDetector:
    @staticmethod
    def get_available_models():
        """Detect all locally available Ollama models"""
        try:
            result = ollama.list()
            return [model['name'] for model in result['models']]
        except:
            return []


# ================== MODEL TRAINER ==================
class ModelTrainer:
    def __init__(self, model_name, knowledge_base):
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.model_path = f"trained_models/{model_name.replace(':', '_')}_{int(time.time())}"

    def train(self, params):
        """Train a model with configurable parameters"""
        os.makedirs(self.model_path, exist_ok=True)

        # Prepare dataset
        texts = self.knowledge_base.text_chunks
        if len(texts) < 10:
            return None, "Insufficient data for training (min 10 chunks required)"

        # Create synthetic labels for demonstration (real use would have real labels)
        labels = np.array([hash(text) % 3 for text in texts], dtype=np.int64)

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3
        )

        # Tokenize text
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=params["max_length"],
            return_tensors="pt"
        )

        # Create dataset
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

        # Training configuration
        training_args = TrainingArguments(
            output_dir=self.model_path,
            num_train_epochs=params["epochs"],
            per_device_train_batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            save_strategy="epoch",
            no_cuda=True,  # CPU-only training
            report_to="none"
        )

        # Start training
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        print(f"Training {self.model_name} with {len(texts)} samples...")
        trainer.train()

        # Save final model
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)

        return self.model_path, "Training successful"


# ================== MAIN APPLICATION ==================
class DeepLearningStudio:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.trained_models = {}
        self.available_models = ModelDetector.get_available_models()

    def import_knowledge(self, file_path):
        return self.knowledge_base.import_data(file_path)

    def train_all_models(self, params=DEFAULT_PARAMS):
        """Train all detected models"""
        if not self.available_models:
            return False, "No models available. Install Ollama models first."

        if not self.knowledge_base.text_chunks:
            return False, "Import knowledge first"

        print(f"\nTraining {len(self.available_models)} models with parameters:")
        print(f"• Epochs: {params['epochs']} (more epochs = better accuracy but longer training)")
        print(f"• Learning rate: {params['learning_rate']} (lower = slower but more precise learning)")
        print(f"• Batch size: {params['batch_size']} (larger = faster training but more memory)")
        print(f"• Context length: {params['max_length']} tokens\n")

        # Train all detected models
        for model_name in self.available_models:
            print(f"Starting training for {model_name}...")
            trainer = ModelTrainer(model_name, self.knowledge_base)
            model_path, message = trainer.train(params)
            if model_path:
                self.trained_models[model_name] = model_path
                print(f"Trained {model_name} saved at: {model_path}")
            else:
                print(f"Failed to train {model_name}: {message}")

        return True, f"Completed training for {len(self.available_models)} models"

    def query_model(self, model_name, question):
        """Query a trained model with context from knowledge base"""
        if model_name not in self.trained_models:
            return f"Model {model_name} not trained yet"

        # Retrieve relevant context
        context = self.knowledge_base.get_context(question)
        full_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

        try:
            response = ollama.generate(
                model=model_name,
                prompt=full_prompt,
                options={
                    "temperature": DEFAULT_PARAMS["temperature"],
                    "num_predict": 256
                }
            )
            return response['response']
        except Exception as e:
            return f"Error: {str(e)}"


# ================== COMMAND LINE INTERFACE ==================
def print_params_explanation():
    print("\nParameter Explanations:")
    print("1. Epochs: Number of complete passes through training data")
    print("   - Higher = Better accuracy but longer training")
    print("2. Learning Rate: Step size for weight updates")
    print("   - Lower = More precise but slower convergence")
    print("3. Batch Size: Number of samples processed per update")
    print("   - Larger = Faster training but more memory required")
    print("4. Max Length: Maximum context length (in tokens)")
    print("   - Longer = More context but slower processing")
    print("5. Temperature: Output randomness (0-1 range)")
    print("   - Lower = More focused and deterministic answers")


def print_available_models(models):
    print("\nAvailable Models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")


def main():
    studio = DeepLearningStudio()

    # Create necessary directories
    os.makedirs("trained_models", exist_ok=True)

    # Check Ollama service
    try:
        ollama.list()
        print("Ollama service: ACTIVE")
        print(f"Detected {len(studio.available_models)} models")
    except:
        print("Ollama service: INACTIVE - Install and run Ollama first")
        studio.available_models = []

    while True:
        print("\n==== Deep Learning Studio ====")
        print("1. Import Knowledge Base")
        print("2. Train All Detected Models")
        print("3. Query a Model")
        print("4. Rescan for Models")
        print("5. Exit")

        choice = input("Select: ").strip()

        if choice == "1":
            file_path = input("File path: ").strip()
            success, message = studio.import_knowledge(file_path)
            print(f"\n{message}")

        elif choice == "2":
            if not studio.available_models:
                print("No models available. Rescan first.")
                continue

            print("Use default parameters? (Y/n)")
            if input().strip().lower() in ('', 'y', 'yes'):
                params = DEFAULT_PARAMS
            else:
                print_params_explanation()
                params = DEFAULT_PARAMS.copy()
                try:
                    params["epochs"] = int(input("\nEpochs (3): ") or 3)
                    params["learning_rate"] = float(input("Learning Rate (2e-5): ") or 2e-5)
                    params["batch_size"] = int(input("Batch Size (8): ") or 8)
                    params["max_length"] = int(input("Max Length (512): ") or 512)
                    params["temperature"] = float(input("Temperature (0.7): ") or 0.7)
                except ValueError:
                    print("Invalid input. Using defaults")

            success, message = studio.train_all_models(params)
            print(f"\n{message}")

        elif choice == "3":
            if not studio.trained_models:
                print("No trained models available. Train models first.")
                continue

            print_available_models(studio.trained_models.keys())
            try:
                model_idx = int(input("Select model number: ")) - 1
                model_name = list(studio.trained_models.keys())[model_idx]
                question = input("\nYour question: ").strip()
                response = studio.query_model(model_name, question)

                print("\n=== Response ===")
                print(response)
                print("================")
            except (ValueError, IndexError):
                print("Invalid selection")

        elif choice == "4":
            studio.available_models = ModelDetector.get_available_models()
            print(f"Rescanned models. Found {len(studio.available_models)} models")
            if studio.available_models:
                print_available_models(studio.available_models)

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()