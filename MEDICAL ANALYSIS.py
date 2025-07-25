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

# Force CUDA usage
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


# PDF processing module
def pdf_to_text(file_path):
    """Process PDF files using PyPDF2"""
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
        # Hardware configuration
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        print(f"Using device: {self.device}")

        if self.cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # GPT-2 model path (project root directory only)
        self.gpt2_model_path = os.path.join(os.getcwd(), "gpt2")

        # Check if model exists
        if os.path.exists(self.gpt2_model_path) and any(
                os.path.exists(os.path.join(self.gpt2_model_path, f))
                for f in ["pytorch_model.bin", "model.safetensors"]
        ):
            print(f"Found GPT-2 model at: {self.gpt2_model_path}")
            self.gpt2_exists = True
        else:
            print(f"GPT-2 model not found at: {self.gpt2_model_path}")
            self.gpt2_exists = False

        # Ollama model path
        self.ollama_model_path = os.path.join(os.getcwd(), "ollama_models")
        os.makedirs(self.ollama_model_path, exist_ok=True)
        print(f"Ollama models stored at: {self.ollama_model_path}")

        # Knowledge base path (project root directory)
        self.knowledge_paths = self.setup_knowledge_paths()

        # Training parameters
        self.epochs = 30
        self.batch_size = 2
        self.learning_rate = 1e-5
        self.max_seq_length = 512

        # File processing
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Output configuration
        self.model_name = "MEDICAL_ANALYSIS"
        self.output_dir = "MEDICAL_ANALYSIS_MODEL"
        os.makedirs(self.output_dir, exist_ok=True)

        # Multi-language support
        self.languages = ["CN", "EN"]
        self.current_lang = "EN"

        # Load local models
        self.tokenizer = self.load_tokenizer()
        self.embedding_model = self.load_ollama_embeddings()

        print("\n=== MEDICAL ANALYSIS CONFIGURATION ===")
        print(f"GPT-2 Model Exists: {self.gpt2_exists}")
        print(f"Knowledge Paths: {self.knowledge_paths}")
        print(f"Using GPU: {self.cuda_available}")
        print("=====================================")

    def set_language(self, lang):
        """Set current language for responses"""
        if lang in self.languages:
            self.current_lang = lang
        else:
            print(f"Unsupported language: {lang}")

    def setup_knowledge_paths(self):
        """Set knowledge base paths to 'knowledge_base' folder in project root"""
        knowledge_dir = os.path.join(os.getcwd(), "knowledge_base")

        # Create directory if it doesn't exist
        if not os.path.exists(knowledge_dir):
            print(f"Knowledge base directory not found. Creating at: {knowledge_dir}")
            os.makedirs(knowledge_dir)
            # Add example file
            with open(os.path.join(knowledge_dir, "example.txt"), "w", encoding="utf-8") as f:
                f.write(
                    "Disease: Common Cold\nSymptoms: Cough, Fever, Runny Nose\nTreatment: Rest, Drink Water, Take Antipyretics")

        return [knowledge_dir]

    def load_tokenizer(self):
        """Load tokenizer"""
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

            # Add special tokens
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
        """Load Ollama embedding model (offline)"""
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

        # Load knowledge
        self.load_knowledge()

        # Create vector store
        self.create_vector_store()
        print(
            f"Knowledge base loaded. Diseases: {len(self.disease_info)}, Symptoms: {len(self.symptom_info)}, Chunks: {len(self.chunks)}")

    def load_file(self, file_path):
        """Load a single knowledge file with improved error handling"""
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
        """Flexible medical information extraction"""
        try:
            # Disease information extraction (flexible rules)
            disease_pattern = re.compile(r'(?:disease|condition|illness)[\s:]*([^\n]+)', re.IGNORECASE)
            symptom_pattern = re.compile(r'(?:symptoms|signs)[\s:]*([^\n]+)', re.IGNORECASE)
            treatment_pattern = re.compile(r'(?:treatments|therapy|management)[\s:]*([^\n]+)', re.IGNORECASE)
            medication_pattern = re.compile(r'(?:medications|drugs|prescriptions)[\s:]*([^\n]+)', re.IGNORECASE)

            # Extract disease information
            disease_match = disease_pattern.search(text)
            if disease_match:
                disease_name = disease_match.group(1).strip().split('\n')[0].split(',')[0].strip()

                # Extract related symptoms
                symptoms = []
                symptom_match = symptom_pattern.search(text)
                if symptom_match:
                    symptoms = [s.strip() for s in symptom_match.group(1).split(',')]

                # Extract treatments
                treatments = []
                treatment_match = treatment_pattern.search(text)
                if treatment_match:
                    treatments = [t.strip() for t in treatment_match.group(1).split(',')]

                # Extract medications
                medications = []
                medication_match = medication_pattern.search(text)
                if medication_match:
                    medications = [m.strip() for m in medication_match.group(1).split(',')]

                # Save disease information
                self.disease_info[disease_name] = {
                    "symptoms": symptoms,
                    "treatments": treatments,
                    "medications": medications
                }

            # Symptom information extraction (flexible rules)
            symptom_names = set()
            for line in text.split('\n'):
                if "symptom" in line.lower() or "sign" in line.lower():
                    parts = re.split(r'[:]', line, maxsplit=1)
                    if len(parts) > 1:
                        symptoms = [s.strip() for s in parts[1].split(',')]
                        symptom_names.update(symptoms)
                    else:
                        # Try to extract symptoms without explicit label
                        possible_symptoms = re.findall(r'\b\w+\s+\w+\b|\b\w+\b', line)
                        symptom_names.update(possible_symptoms)

            # Save symptom information
            for symptom in symptom_names:
                if symptom and symptom not in self.symptom_info:
                    self.symptom_info[symptom] = {
                        "description": "",
                        "possible_diseases": []
                    }

            return True
        except Exception as e:
            print(f"Error extracting medical info: {str(e)}")
            return False

    def load_knowledge(self):
        """Load all knowledge base files - with flexible text processing"""
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
                            # Load file content
                            contents = self.load_file(file_path)

                            for content in contents:
                                # Extract medical information
                                self.extract_medical_info(content)

                                # Split text regardless of extraction success
                                chunks = splitter.split_text(content)
                                self.chunks.extend(chunks)
                                self.total_chunks += len(chunks)

                            file_count += 1
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")

            print(f"Processed {file_count} files in {path}")

        # If no diseases were extracted, create some defaults
        if not self.disease_info:
            print("No diseases extracted, creating default knowledge")
            self.disease_info = {
                "Common Cold": {
                    "symptoms": ["Cough", "Fever", "Runny Nose", "Sore Throat"],
                    "treatments": ["Rest", "Drink Fluids", "Over-the-counter Cold Medicine"],
                    "medications": ["Acetaminophen", "Ibuprofen", "Decongestants"]
                },
                "Influenza": {
                    "symptoms": ["High Fever", "Body Aches", "Fatigue", "Headache"],
                    "treatments": ["Rest", "Hydration", "Antiviral Medication"],
                    "medications": ["Oseltamivir (Tamiflu)", "Zanamivir (Relenza)"]
                }
            }

        # If no symptoms were extracted, create some defaults
        if not self.symptom_info:
            print("No symptoms extracted, creating default knowledge")
            self.symptom_info = {
                "Cough": {"description": "Reflex to clear airways", "possible_diseases": ["Common Cold", "Bronchitis"]},
                "Fever": {"description": "Elevated body temperature", "possible_diseases": ["Infection", "Influenza"]}
            }

    def create_vector_store(self):
        """Create vector store with FAISS"""
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
        """Diagnose disease based on symptoms - with FAISS similarity search"""
        possible_diseases = {}

        # If no symptoms, return empty
        if not symptoms:
            return []

        # First try simple matching
        for symptom in symptoms:
            for disease, info in self.disease_info.items():
                if any(symptom.lower() in s.lower() for s in info["symptoms"]):
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1
                elif symptom.lower() in disease.lower():
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1

        # If simple matching found results, use those
        if possible_diseases:
            # Calculate probability
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

            # Sort by probability
            return sorted(results, key=lambda x: x["probability"], reverse=True)

        # If no matches found, try FAISS similarity search
        if self.vector_store:
            try:
                print("Using FAISS for similarity search...")
                # Combine symptoms into a query
                query = "Symptoms: " + ", ".join(symptoms)

                # Search for similar documents
                docs = self.vector_store.similarity_search(query, k=3)

                # Extract diseases from similar documents
                for doc in docs:
                    content = doc.page_content
                    # Try to find disease names in the content
                    disease_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                    for disease in disease_matches:
                        if disease in self.disease_info:
                            possible_diseases[disease] = possible_diseases.get(disease, 0) + 1

                # If we found matches through similarity search
                if possible_diseases:
                    results = []
                    for disease, count in possible_diseases.items():
                        results.append({
                            "disease": disease,
                            "probability": min(count * 30, 80),  # Max 80% for similarity-based
                            "matched_symptoms": 0,
                            "total_symptoms": 0,
                            "method": "similarity"
                        })
                    return sorted(results, key=lambda x: x["probability"], reverse=True)

            except Exception as e:
                print(f"Error in FAISS similarity search: {str(e)}")

        # If all methods fail, return empty
        return []

    def get_treatment_plan(self, disease):
        """Get treatment plan for a disease"""
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

        # Load GPT-2 model
        try:
            if config.gpt2_exists:
                print(f"Loading GPT-2 model from: {config.gpt2_model_path}")
                self.gpt2 = GPT2LMHeadModel.from_pretrained(config.gpt2_model_path)
            else:
                print("Using base GPT-2 model")
                self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        except Exception as e:
            print(f"Error loading GPT-2 model: {str(e)}")
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

        # Adjust token embeddings size
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
        """Generate response with GPU support"""
        generation_config = {
            "max_length": max_length,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": self.config.tokenizer.eos_token_id,
            **kwargs
        }

        # Move tensors to device
        input_ids = input_ids.to(self.config.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)

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

        # Prepare model
        self.model = MedicalAnalysisModel(config).to(config.device)
        print(f"Model placed on {config.device}")

        # Prepare dataset
        dataset = MedicalDataset(knowledge_base, config)
        self.train_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )

        # Ensure we have training data
        if len(dataset) == 0:
            print("No training data available")
            return

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        # Learning rate scheduler
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Loss function
        self.criterion = CrossEntropyLoss(ignore_index=-100)

        # Training monitoring
        self.loss_history = []

        # Mixed precision training
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
            # Ensure all tensors are on the correct device
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["labels"].to(self.config.device)

            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    # Calculate loss
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.criterion(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )

                # Backpropagation
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Calculate loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
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
        """Save model"""
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

        # Generate training data
        self.generate_training_data()
        print(f"Created {len(self.examples)} training examples")

    def generate_training_data(self):
        """Generate medical training data"""
        # 1. Disease diagnosis examples
        for disease, info in self.knowledge_base.disease_info.items():
            symptoms = info["symptoms"]
            treatments = info["treatments"]
            medications = info["medications"]

            if symptoms:
                # English example
                self.examples.append({
                    "input": f"Patient symptoms: {', '.join(symptoms[:3])}",
                    "output": f"<BOS>Based on the symptoms, the patient may have {disease}. Recommended treatments: {', '.join(treatments[:2])}. Medications: {', '.join(medications[:1])}.<EOS>",
                    "lang": "EN"
                })

        # 2. Symptom analysis examples
        for symptom in self.knowledge_base.symptom_info.keys():
            # English example
            self.examples.append({
                "input": f"Patient complains: {symptom}",
                "output": f"<BOS>{symptom} may be related to various diseases. Detailed examination is recommended.<EOS>",
                "lang": "EN"
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Encode input
        input_enc = self.tokenizer(
            example["input"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Encode output
        output_enc = self.tokenizer(
            example["output"],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels (ignore padding)
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

        # Load model
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            self.model = None
            return

        try:
            self.model = MedicalAnalysisModel(config)
            self.model.load_state_dict(torch.load(model_path, map_location=config.device))
            self.model.to(config.device)
            self.model.eval()
            print(f"Medical assistant model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def analyze_symptoms(self, symptoms_text, lang=None):
        """Analyze symptoms and provide medical advice"""
        if lang:
            self.config.set_language(lang)

        # Save user input
        self.conversation_context.append(symptoms_text)

        # Extract symptom keywords
        symptoms = self.extract_symptoms(symptoms_text)

        # Try to diagnose
        diagnoses = self.knowledge_base.diagnose(symptoms)

        # Response logic
        if not diagnoses:
            if len(self.conversation_context) > 2:  # After multiple attempts
                return self.response(
                    "Based on the current knowledge base, the cause cannot be determined. Please update the knowledge base.",
                    "Based on the current knowledge base, the cause cannot be determined. Please update the knowledge base."
                )
            else:  # First attempt
                return self.response(
                    "Please describe your symptoms in more detail.",
                    "Please describe your symptoms in more detail."
                )

        # Get highest probability disease
        top_diagnosis = diagnoses[0]
        disease = top_diagnosis["disease"]
        treatment = self.knowledge_base.get_treatment_plan(disease)

        # Generate response
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
        """Extract symptom keywords from text"""
        found_symptoms = []

        # Iterate through all symptoms in knowledge base
        for symptom in self.knowledge_base.symptom_info.keys():
            if symptom.lower() in text.lower():
                found_symptoms.append(symptom)

        # If nothing found, try fuzzy matching
        if not found_symptoms:
            symptom_words = set()
            for word in text.split():
                # Common symptom keywords list
                symptom_keywords = ["pain", "ache", "cough", "fever", "vomit", "dizzy", "bleed", "itch", "swell",
                                    "discomfort"]
                if any(keyword in word for keyword in symptom_words) or any(
                        keyword in word for keyword in symptom_keywords):
                    symptom_words.add(word)

            found_symptoms = list(symptom_words)

        return found_symptoms

    def format_list(self, items):
        """Format a list of items for display"""
        if not items:
            return "None"
        return "\n".join([f"- {item}" for item in items])

    def response(self, cn_text, en_text):
        """Return response based on current language"""
        if self.config.current_lang == "CN":
            return cn_text
        return en_text

    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = []


# ========== MAIN ==========
def main():
    try:
        # Initialize configuration
        config = MedicalConfig()

        # Load knowledge base
        print("\n[1/3] Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # Check if knowledge base is empty
        if not knowledge_base.chunks:
            print("Knowledge base is empty, creating default knowledge")
            # Create default knowledge after reloading
            time.sleep(1)
            knowledge_base = MedicalKnowledgeBase(config)

        # Train model
        print("\n[2/3] Training Medical Analysis Model...")
        trainer = MedicalTrainer(config, knowledge_base)

        best_loss = float('inf')
        best_epoch = -1

        for epoch in range(config.epochs):
            current_loss = trainer.train_epoch(epoch)

            # Save the best model
            if current_loss < best_loss and current_loss > 0:
                best_loss = current_loss
                best_epoch = epoch
                trainer.save_model(epoch)

        # Save final model
        if best_epoch >= 0:
            model_path = trainer.save_model(best_epoch)
        else:
            model_path = os.path.join(config.output_dir, f"{config.model_name}_default.pth")
            print(f"Using default model path: {model_path}")

        # Initialize medical assistant
        print(f"\n[3/3] Starting Medical Assistant (model: {model_path})")
        assistant = MedicalAssistant(model_path, knowledge_base, config)

        # Interactive interface
        print("\n=== MEDICAL ANALYSIS ASSISTANT ===")
        print("Commands: lang [CN/EN], clear, exit")

        while True:
            try:
                # Get user input
                user_input = input("\nPatient Symptoms: ").strip()

                # Process commands
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
                        print("Conversation context cleared")
                    continue

                # Analyze symptoms
                if assistant and assistant.model:
                    start_time = time.time()
                    response = assistant.analyze_symptoms(user_input)
                    response_time = time.time() - start_time

                    # Display response
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