import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import csv
import faiss
from langchain_community.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json
import docx
from PyPDF2 import PdfReader
import logging
import langid
from deep_translator import GoogleTranslator
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset


# ========== LOGGER SETUP ==========
def setup_logger():
    """Set up logger"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medical_diagnosis_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    # Add console output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger('MedicalDiagnosis')


# Initialize logger
logger = setup_logger()


# ========== CONFIGURATION ==========
class MedicalConfig:
    def __init__(self):
        # Knowledge base paths
        self.knowledge_paths = self.setup_knowledge_paths()

        # Chart output directory
        self.chart_dir = "sci_charts"
        os.makedirs(self.chart_dir, exist_ok=True)
        logger.info(f"Chart output directory: {self.chart_dir}")

        # Model directory
        self.model_dir = "trained_models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model directory: {self.model_dir}")

        # File processing
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Ollama model configuration
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_model = "llama3"
        self.generation_temp = 0.7
        self.generation_top_p = 0.9
        self.diagnosis_threshold = 0.95  # 95% similarity threshold
        self.max_attempts = 3  # Maximum inquiry attempts

        # DDD configuration
        self.ddd_threshold = 0.8  # High DDD value threshold

        # Training configuration
        self.epochs = 5
        self.batch_size = 4
        self.learning_rate = 2e-5

        logger.info("\n=== MEDICAL ANALYSIS CONFIGURATION ===")
        logger.info(f"Knowledge Paths: {self.knowledge_paths}")
        logger.info(f"Ollama Model: {self.ollama_model}")
        logger.info(f"Diagnosis Threshold: {self.diagnosis_threshold * 100}%")
        logger.info(f"Max Inquiry Attempts: {self.max_attempts}")
        logger.info(f"Training Epochs: {self.epochs}")
        logger.info("=====================================")

    def setup_knowledge_paths(self):
        """Set knowledge base paths to 'knowledge_base' folder"""
        knowledge_dir = os.path.join(os.getcwd(), "knowledge_base")
        os.makedirs(knowledge_dir, exist_ok=True)
        logger.info(f"Knowledge path: {knowledge_dir}")
        return [knowledge_dir]

    def get_ollama_generator(self):
        """Create Ollama generator"""
        return Ollama(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
            temperature=self.generation_temp,
            top_p=self.generation_top_p
        )

    def translate_to_english(self, text):
        """Translate text to English"""
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def translate_to_chinese(self, text):
        """Translate text to Chinese"""
        try:
            return GoogleTranslator(source='auto', target='zh-CN').translate(text)
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text


# ========== KNOWLEDGE BASE ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.chunks = []
        self.disease_info = {}
        self.symptom_info = {}
        self.total_chunks = 0
        self.faiss_index = None
        self.symptom_embeddings = {}
        self.disease_embeddings = {}
        self.ollama_generator = config.get_ollama_generator()
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

        # DDD storage
        self.medication_ddd_info = defaultdict(dict)  # {medication: {ddd_value, unit, route}}
        self.medication_unit_conversion = self.setup_unit_conversion()

        # Record learning process
        self.learning_stats = {
            "files_processed": 0,
            "diseases_extracted": 0,
            "symptoms_extracted": 0,
            "chunks_created": 0,
            "ddds_extracted": 0,
            "training_examples": 0
        }

        # Load knowledge
        self.load_knowledge()

        # Create vector store
        self.create_vector_store()

        # Train diagnostic model
        self.train_diagnostic_model()

        # Generate learning charts
        self.generate_learning_charts()

        logger.info(
            f"Knowledge base loaded. Diseases: {len(self.disease_info)}, Symptoms: {len(self.symptom_info)}, Chunks: {len(self.chunks)}")

    def setup_unit_conversion(self):
        """Set up medication unit conversion system"""
        return {
            'mg': 1,
            'milligram': 1,
            'g': 1000,
            'gram': 1000,
            'mcg': 0.001,
            'microgram': 0.001,
            'μg': 0.001,
            'IU': {'Vit D': 0.025, 'Insulin': 1},  # Special cases
            'units': 1,
            'ml': 1  # Assuming 1mg/ml for liquids for simplicity
        }

    def extract_ddd_info(self, text):
        """Intelligently extract DDD information from text"""
        try:
            # Common patterns for DDD information
            patterns = [
                r'DDD:\s*([\d.]+)\s*(\w+)',  # DDD: 100 mg
                r'daily\s+dose.*?([\d.]+)\s*(\w+)',  # daily dose: 100 mg
                r'recommended\s+dose.*?([\d.]+)\s*(\w+)',  # recommended dose: 100 mg
                r'standard\s+dosage.*?([\d.]+)\s*(\w+)',  # standard dosage: 100 mg
                r'剂量:\s*([\d.]+)\s*(\w+)',  # Chinese pattern
                r'每日剂量.*?([\d.]+)\s*(\w+)'  # Chinese pattern
            ]

            all_medications = []
            for med_list in self.disease_info.values():
                all_medications.extend(med_list.get('medications', []))

            extracted_count = 0

            for medication in set(all_medications):
                medication_lower = medication.lower()
                # Try to find medication in text
                if medication_lower in text.lower():
                    for pattern in patterns:
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            ddd_value = float(match.group(1))
                            unit = match.group(2).lower()

                            # Standardize unit
                            if unit in ['mg', 'milligram', 'mg/d']:
                                unit = 'mg'
                            elif unit in ['g', 'gram']:
                                unit = 'g'

                            # Route detection (simplified)
                            route = 'oral'  # Default to oral
                            if 'iv' in text.lower() or 'intravenous' in text.lower():
                                route = 'iv'
                            elif 'injection' in text.lower() or '注射' in text.lower():
                                route = 'im'

                            self.medication_ddd_info[medication] = {
                                'ddd_value': ddd_value,
                                'unit': unit,
                                'route': route
                            }
                            extracted_count += 1
                            break

            logger.info(f"Extracted DDD info for {extracted_count} medications")
            self.learning_stats["ddds_extracted"] += extracted_count

            return extracted_count > 0
        except Exception as e:
            logger.error(f"Error extracting DDD info: {str(e)}")
            return False

    def calculate_ddd(self, medication, dosage, unit, frequency):
        """Calculate DDD values based on medication information"""
        try:
            # Check if we have learned DDD information
            if medication in self.medication_ddd_info:
                ddd_info = self.medication_ddd_info[medication]
                ddd_value = ddd_info['ddd_value']
                ddd_unit = ddd_info['unit']
                route = ddd_info['route']

                # Convert dosage to mg
                dose_in_mg = self.convert_to_mg(dosage, unit)
                ddd_in_mg = self.convert_to_mg(ddd_value, ddd_unit)

                # Calculate DDD ratio
                return dose_in_mg / ddd_in_mg
            else:
                # Use Ollama to predict DDD
                return self.predict_ddd_with_ollama(medication, dosage, unit, frequency)
        except Exception as e:
            logger.error(f"Error calculating DDD for {medication}: {str(e)}")
            return 0.0

    def convert_to_mg(self, value, unit):
        """Convert medication unit to milligrams"""
        unit = unit.lower()
        conv = self.medication_unit_conversion

        # Special handling for IU (International Units)
        if unit == 'iu':
            # Attempt to determine IU type
            if 'insulin' in unit:
                return value * conv['IU']['Insulin']
            return value * conv['IU']['Vit D']  # Default to Vit D conversion

        # Standard conversion
        if unit in conv:
            if isinstance(conv[unit], dict):
                # Can't convert without knowing type
                return 0.0
            return value * conv[unit]

        # Try to find similar units
        for known_unit, factor in conv.items():
            if known_unit in unit:
                return value * factor

        logger.warning(f"Unknown medication unit: {unit}")
        return 0.0

    def predict_ddd_with_ollama(self, medication, dosage, unit, frequency):
        """Use Ollama to predict DDD when not in knowledge base"""
        prompt = f"""
        As a medical expert, predict the Defined Daily Dose (DDD) for:
        Medication: {medication}
        Daily dosage: {dosage} {unit} (frequency: {frequency})

        Required:
        1. Calculate DDD ratio (0.0 - 2.0) based on standard medical practice
        2. Output format: JSON with keys: "ddd_ratio", "certainty"
        """

        try:
            response = self.ollama_generator.invoke(prompt)
            logger.info(f"DDD prediction for {medication}: {response}")

            # Parse JSON response
            try:
                ddd_data = json.loads(response)
                ddd_ratio = float(ddd_data.get('ddd_ratio', 0))
                return max(0, min(ddd_ratio, 2.0))  # Constrain between 0-2.0
            except:
                # Fallback to pattern matching if JSON parse fails
                num_match = re.search(r'[\d.]+', response)
                if num_match:
                    return float(num_match.group(0))
                return 0.0
        except Exception as e:
            logger.error(f"Error predicting DDD: {str(e)}")
            return 0.0

    def extract_medical_info(self, text):
        """Comprehensive medical information extraction with DDD support"""
        try:
            # Extract all disease information
            disease_matches = re.findall(r'(?:disease|condition|illness|diagnosis)[\s:]*([^\n]+)', text, re.IGNORECASE)
            for match in disease_matches:
                disease_name = match.strip().split('\n')[0].split(',')[0].strip()

                # Extract related symptoms
                symptoms = []
                symptom_matches = re.findall(r'(?:symptoms|signs|complaint)[\s:]*([^\n]+)', text, re.IGNORECASE)
                for sm in symptom_matches:
                    symptoms.extend([s.strip() for s in sm.split(',')])

                # Extract treatments
                treatments = []
                treatment_matches = re.findall(r'(?:treatments|therapy|management|plan)[\s:]*([^\n]+)', text,
                                               re.IGNORECASE)
                for tm in treatment_matches:
                    treatments.extend([t.strip() for t in tm.split(',')])

                # Extract medications with dosing information
                medications = []
                medication_matches = re.findall(
                    r'(?:medications|drugs|prescriptions|剂量|药物)[\s:]*([^\n]+(?:\n[^\n]*)?)', text, re.IGNORECASE)
                for mm in medication_matches:
                    # Extract medication lines
                    med_lines = mm.split('\n')
                    for line in med_lines:
                        # Match medication with possible dosing info
                        med_match = re.search(
                            r'([a-zA-Z\u4e00-\u9fff]+\s*[a-zA-Z\u4e00-\u9fff]*)[\s(]*([\d.]+)?\s*([a-zA-Z]+)?[\s]*[/\s]*([\d.]+)?\s*(\w+)?',
                            line)
                        if med_match:
                            name = med_match.group(1).strip()
                            dosage = med_match.group(2) if med_match.group(2) else None
                            unit = med_match.group(3) if med_match.group(3) else None

                            medications.append({
                                'name': name,
                                'dosage': dosage,
                                'unit': unit
                            })

                # Save disease information
                if disease_name not in self.disease_info:
                    self.disease_info[disease_name] = {
                        "symptoms": symptoms,
                        "treatments": treatments,
                        "medications": medications
                    }
                    self.learning_stats["diseases_extracted"] += 1

            # Extract all symptom information
            symptom_names = set()
            symptom_matches = re.findall(r'(?:symptom|sign|complaint)[\s:]*([^\n]+)', text, re.IGNORECASE)
            for sm in symptom_matches:
                symptom_names.update([s.strip() for s in sm.split(',')])

            # Save symptom information
            for symptom in symptom_names:
                if symptom and symptom not in self.symptom_info:
                    self.symptom_info[symptom] = {
                        "description": "",
                        "possible_diseases": []
                    }
                    self.learning_stats["symptoms_extracted"] += 1

            # Extract DDD information from text
            self.extract_ddd_info(text)

            return True
        except Exception as e:
            logger.error(f"Error extracting medical info: {str(e)}")
            return False

    def load_knowledge(self):
        """Load all knowledge base files"""
        logger.info("Loading medical knowledge...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        for path in self.config.knowledge_paths:
            if not os.path.exists(path):
                logger.warning(f"Knowledge path not found: {path}")
                continue

            logger.info(f"Processing directory: {path}")
            file_count = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file_path.endswith(ext) for ext in ('.txt', '.csv', '.json', '.docx', '.pdf')):
                        logger.info(f"Processing file: {file_path}")
                        try:
                            # Load file content
                            contents = self.load_file(file_path)

                            for content in contents:
                                # Extract medical information
                                self.extract_medical_info(content)

                                # Split text
                                chunks = splitter.split_text(content)
                                self.chunks.extend(chunks)
                                self.learning_stats["chunks_created"] += len(chunks)

                            file_count += 1
                            self.learning_stats["files_processed"] += 1
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")

            logger.info(f"Processed {file_count} files in {path}")

        # If no diseases extracted, create default knowledge
        if not self.disease_info:
            logger.info("No diseases extracted, creating default knowledge")
            self.disease_info = {
                "Common Cold": {
                    "symptoms": ["Cough", "Fever", "Runny Nose"],
                    "treatments": ["Rest", "Drink Fluids"],
                    "medications": [{"name": "Acetaminophen", "dosage": "500", "unit": "mg"}]
                }
            }
            self.learning_stats["diseases_extracted"] += 1

        # Create embeddings for all symptoms and diseases
        self.create_embeddings()

    def create_embeddings(self):
        """Create embeddings for all symptoms and diseases"""
        logger.info("Creating symptom and disease embeddings...")

        # Symptom embeddings
        for symptom in self.symptom_info.keys():
            self.symptom_embeddings[symptom] = self.embedding_model.embed_query(symptom)

        # Disease embeddings
        for disease, info in self.disease_info.items():
            # Create embedding using disease name and symptoms
            disease_text = f"{disease}: {', '.join(info['symptoms'])}"
            self.disease_embeddings[disease] = self.embedding_model.embed_query(disease_text)

        logger.info(
            f"Created {len(self.symptom_embeddings)} symptom embeddings and {len(self.disease_embeddings)} disease embeddings")

    def create_faiss_index(self, embeddings):
        """Create Faiss index"""
        logger.info("Creating Faiss index...")
        dim = embeddings.shape[1]

        # Create base index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        self.faiss_index = index
        logger.info("Faiss index created on CPU")

    def create_vector_store(self):
        """Create vector store"""
        if not self.chunks:
            logger.warning("No knowledge chunks, skipping vector store creation")
            return

        logger.info(f"Creating vector store with {len(self.chunks)} chunks...")
        try:
            # Create FAISS store
            self.vector_store = FAISS.from_texts(self.chunks, self.embedding_model)
            logger.info("FAISS vector store created successfully")

            # Create Faiss index for hybrid search
            embeddings = np.array([self.embedding_model.embed_query(chunk) for chunk in self.chunks])
            self.create_faiss_index(embeddings.astype('float32'))

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")

    def calculate_similarity(self, symptom1, symptom2):
        """Calculate cosine similarity between two symptoms"""
        emb1 = self.symptom_embeddings.get(symptom1)
        emb2 = self.symptom_embeddings.get(symptom2)

        if emb1 is None or emb2 is None:
            return 0.0

        return cosine_similarity([emb1], [emb2])[0][0]

    def prepare_training_data(self):
        """Prepare training data for diagnostic model"""
        logger.info("Preparing training data for diagnostic model...")

        training_data = []

        # Create training examples from disease info
        for disease, info in self.disease_info.items():
            symptoms = ", ".join(info["symptoms"])
            medications = ", ".join([med['name'] for med in info["medications"]])

            # Create multiple variations of symptoms
            for i in range(3):
                shuffled_symptoms = list(info["symptoms"])
                np.random.shuffle(shuffled_symptoms)
                symptoms_text = ", ".join(shuffled_symptoms[:len(shuffled_symptoms) - i])

                training_data.append({
                    "symptoms": symptoms_text,
                    "disease": disease,
                    "medications": medications
                })

        # Create negative examples
        diseases = list(self.disease_info.keys())
        for i in range(min(100, len(diseases) * 2)):
            disease1, disease2 = np.random.choice(diseases, 2, replace=False)
            symptoms = ", ".join(np.random.choice(self.disease_info[disease1]["symptoms"],
                                                  min(3, len(self.disease_info[disease1]["symptoms"])),
                                                  replace=False))

            training_data.append({
                "symptoms": symptoms,
                "disease": disease2,
                "medications": ""
            })

        self.learning_stats["training_examples"] = len(training_data)
        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data

    def train_diagnostic_model(self):
        """Train diagnostic model using knowledge base"""
        logger.info("Training diagnostic model...")

        # Prepare training data
        training_data = self.prepare_training_data()

        if not training_data:
            logger.warning("No training data available, skipping model training")
            return

        # Split into train and validation sets
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)

        # Convert to Hugging Face dataset
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

        # Tokenizer and model
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples["symptoms"], padding="max_length", truncation=True, max_length=128)

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)

        # Create label mappings
        all_diseases = list(set([item["disease"] for item in training_data]))
        disease_to_id = {disease: idx for idx, disease in enumerate(all_diseases)}
        id_to_disease = {idx: disease for disease, idx in disease_to_id.items()}

        # Add labels
        def add_labels(examples):
            examples["label"] = [disease_to_id[d] for d in examples["disease"]]
            return examples

        tokenized_train = tokenized_train.map(add_labels)
        tokenized_val = tokenized_val.map(add_labels)

        # Model setup
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(all_diseases),
            id2label=id_to_disease,
            label2id=disease_to_id
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_dir, "diagnostic_model"),
            evaluation_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(self.config.model_dir, "logs"),
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        # Train model
        trainer.train()

        # Save model
        model.save_pretrained(os.path.join(self.config.model_dir, "diagnostic_model"))
        tokenizer.save_pretrained(os.path.join(self.config.model_dir, "diagnostic_model"))

        # Save disease mappings
        with open(os.path.join(self.config.model_dir, "disease_mappings.json"), "w") as f:
            json.dump({"disease_to_id": disease_to_id, "id_to_disease": id_to_disease}, f)

        logger.info("Diagnostic model trained and saved successfully")

        # Load model for immediate use
        self.diagnostic_model = model
        self.diagnostic_tokenizer = tokenizer
        self.disease_mappings = {"disease_to_id": disease_to_id, "id_to_disease": id_to_disease}

    def diagnose_with_model(self, symptoms):
        """Diagnose using trained diagnostic model"""
        if not hasattr(self, 'diagnostic_model'):
            logger.error("Diagnostic model not trained")
            return None

        # Tokenize input
        inputs = self.diagnostic_tokenizer(
            symptoms,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Predict
        with torch.no_grad():
            outputs = self.diagnostic_model(**inputs)

        # Get predicted disease
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        disease = self.disease_mappings["id_to_disease"][predicted_class_idx]

        # Get confidence
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0][predicted_class_idx].item()

        # Get medications
        medications = self.disease_info.get(disease, {}).get("medications", [])

        return {
            "disease": disease,
            "confidence": confidence,
            "medications": medications
        }

    def recommend_tests(self, symptoms):
        """Recommend tests based on knowledge base training results"""
        logger.info("Recommending tests based on knowledge base training...")

        # Build prompt using knowledge base content
        prompt = f"""
        As a medical expert, recommend diagnostic tests based on knowledge base content and the following symptoms:
        Symptoms: {', '.join(symptoms)}

        Requirements:
        1. List 3-5 key tests
        2. Prioritize by importance
        3. Explain the purpose of each test
        4. Base recommendations on knowledge base content
        """

        # Add knowledge base content as context
        context = "\n".join(self.chunks[:3])  # Use first 3 chunks as context
        prompt = f"Knowledge Base Context:\n{context}\n\n{prompt}"

        # Use Ollama to generate recommendations
        response = self.ollama_generator.invoke(prompt)
        return response

    def generate_learning_charts(self):
        """Generate learning charts for SCI papers with DDD stats"""
        logger.info("Generating learning charts for SCI papers...")

        try:
            # Create disease-symptom count chart
            diseases = list(self.disease_info.keys())
            symptom_counts = [len(info["symptoms"]) for info in self.disease_info.values()]

            plt.figure(figsize=(12, 8))
            sns.barplot(x=diseases, y=symptom_counts, palette="viridis")
            plt.title("Diseases and Their Symptom Counts", fontsize=16)
            plt.xlabel("Disease", fontsize=14)
            plt.ylabel("Number of Symptoms", fontsize=14)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.chart_dir, "disease_symptom_counts.png"), dpi=300)
            plt.close()

            # Create learning statistics chart
            stats_df = pd.DataFrame({
                "Metric": ["Files Processed", "Diseases Extracted", "Symptoms Extracted",
                           "Chunks Created", "DDDs Extracted", "Training Examples"],
                "Count": [
                    self.learning_stats["files_processed"],
                    self.learning_stats["diseases_extracted"],
                    self.learning_stats["symptoms_extracted"],
                    self.learning_stats["chunks_created"],
                    self.learning_stats["ddds_extracted"],
                    self.learning_stats["training_examples"]
                ]
            })

            plt.figure(figsize=(10, 6))
            sns.barplot(data=stats_df, x="Metric", y="Count", palette="muted")
            plt.title("Knowledge Base Learning Statistics", fontsize=16)
            plt.xlabel("Metric", fontsize=14)
            plt.ylabel("Count", fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.chart_dir, "learning_statistics.png"), dpi=300)
            plt.close()

            # Create medication coverage chart
            if self.medication_ddd_info:
                meds = list(self.medication_ddd_info.keys())
                ddd_values = [info['ddd_value'] for info in self.medication_ddd_info.values()]

                plt.figure(figsize=(12, 8))
                sns.barplot(x=meds, y=ddd_values, palette="rocket")
                plt.title("Medications and Their DDD Values", fontsize=16)
                plt.xlabel("Medication", fontsize=14)
                plt.ylabel("DDD Value", fontsize=14)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.chart_dir, "medication_ddd_values.png"), dpi=300)
                plt.close()

            # Create model training chart
            if hasattr(self, 'training_history'):
                history = self.training_history
                plt.figure(figsize=(10, 6))
                plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
                plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
                plt.title("Model Training Progress", fontsize=16)
                plt.xlabel("Epoch", fontsize=14)
                plt.ylabel("Loss", fontsize=14)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.chart_dir, "training_progress.png"), dpi=300)
                plt.close()

            logger.info("Learning charts generated successfully")
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")

    def load_file(self, file_path):
        """Load a single knowledge file"""
        try:
            content = ""
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    content = "\n".join([",".join(row) for row in reader])
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, ensure_ascii=False)
            elif file_path.endswith('.docx'):
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []

            return [content]
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []


# ========== MEDICAL ASSISTANT ==========
class MedicalAssistant:
    def __init__(self, knowledge_base, config):
        self.config = config
        self.knowledge_base = knowledge_base
        self.ollama_generator = config.get_ollama_generator()
        self.current_symptoms = []
        self.attempt_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.doctor_ddd_history = defaultdict(list)  # Track doctor's prescription DDD values

    def analyze_symptoms(self, symptoms_text):
        """Analyze symptoms and provide medical advice (for doctors)"""
        # Extract symptom keywords (supports Chinese and English)
        symptoms = self.extract_symptoms(symptoms_text)
        self.current_symptoms.extend(symptoms)
        self.attempt_count += 1

        # Log current symptoms
        logger.info(f"Inquiry #{self.attempt_count}: Symptoms - {', '.join(symptoms)}")
        logger.info(f"Current symptoms list: {', '.join(self.current_symptoms)}")

        # Use trained diagnostic model for diagnosis
        diagnosis = self.knowledge_base.diagnose_with_model(", ".join(self.current_symptoms))

        if not diagnosis:
            logger.error("Diagnosis failed")
            return "Diagnosis failed, please try again."

        confidence = diagnosis["confidence"]

        # Check if diagnosis threshold is reached
        if confidence >= self.config.diagnosis_threshold:
            # Threshold reached, provide diagnosis and treatment
            response = self.build_diagnosis_response(diagnosis)
            self.reset_session()
            return response
        elif self.attempt_count >= self.config.max_attempts:
            # Reached max attempts without diagnosis
            response = self.build_test_recommendation()
            self.reset_session()
            return response
        else:
            # Request more symptom information
            return self.request_more_symptoms(confidence)

    def extract_symptoms(self, text):
        """Extract symptom keywords from text (supports Chinese and English)"""
        # Detect input language
        lang, confidence = langid.classify(text)
        is_chinese = lang == 'zh'

        # Use Ollama to extract standardized symptom terms
        prompt = f"""
        As a medical expert, extract standardized symptom terms from the following text:
        Text: {text}

        Requirements:
        1. List all mentioned symptoms
        2. Use standard medical terminology
        3. Format as comma-separated list
        """

        response = self.ollama_generator.invoke(prompt)

        # Parse response
        symptoms = []
        if ":" in response:
            parts = response.split(":", 1)
            if len(parts) > 1:
                symptoms = [s.strip() for s in parts[1].split(",")]
        else:
            symptoms = [s.strip() for s in response.split(",")]

        # Translate to English if input was Chinese
        if is_chinese:
            return [self.config.translate_to_english(s) for s in symptoms]
        return symptoms

    def request_more_symptoms(self, confidence):
        """Request more symptom information"""
        # Use Ollama to generate relevant symptom suggestions
        prompt = f"""
        As a medical expert, recommend additional symptoms to investigate based on:
        Current symptoms: {', '.join(self.current_symptoms)}
        Current confidence: {confidence * 100:.1f}%

        Requirements:
        1. List 3-5 key symptoms
        2. Focus on symptoms that would help differentiate possible diagnoses
        3. Use professional medical terminology
        """

        suggested_symptoms = self.ollama_generator.invoke(prompt)

        # Build bilingual response
        response_en = (
            f"Current symptoms: {', '.join(self.current_symptoms)}\n"
            f"Current confidence: {confidence * 100:.1f}% (threshold: {self.config.diagnosis_threshold * 100}%)\n\n"
            "No definitive diagnosis yet. Please provide additional information:\n\n"
            f"Suggested symptoms to investigate:\n{suggested_symptoms}"
        )

        response_cn = self.config.translate_to_chinese(response_en)

        return f"{response_en}\n\n{response_cn}"

    def build_diagnosis_response(self, diagnosis):
        """Build diagnosis response (bilingual) with DDD calculation"""
        disease = diagnosis["disease"]
        confidence = diagnosis["confidence"]
        medications = diagnosis["medications"]

        # Build English response
        response_en = (
            f"Diagnosis: {disease}\n"
            f"Confidence: {confidence * 100:.1f}%\n\n"
            "Medication Recommendations:\n"
        )

        # Build Chinese response header
        response_cn = (
            f"诊断: {self.config.translate_to_chinese(disease)}\n"
            f"置信度: {confidence * 100:.1f}%\n\n"
            "药物推荐:\n"
        )

        # Process medications with DDD calculation
        med_text_en = ""
        med_text_cn = ""
        for med_info in medications:
            name = med_info["name"]
            dosage = med_info.get("dosage", "")
            unit = med_info.get("unit", "")
            frequency = "daily"  # Default frequency

            # Calculate DDD if possible
            ddd_ratio = 0.0
            if dosage and unit:
                ddd_ratio = self.knowledge_base.calculate_ddd(name, float(dosage), unit, frequency)

                # Add DDD info to text
                med_info_text_en = f"- {name}: {dosage}{unit} daily (DDD ratio: {ddd_ratio:.2f})"
                med_info_text_cn = f"- {self.config.translate_to_chinese(name)}: {dosage}{unit} 每日 (DDD率: {ddd_ratio:.2f})"

                # Add warning for high DDD
                if ddd_ratio > self.config.ddd_threshold:
                    med_info_text_en += " [Warning: High DDD value]"
                    med_info_text_cn += " [警告: DDD值过高]"

                # Track doctor's prescription patterns
                self.doctor_ddd_history[name].append(ddd_ratio)
            else:
                med_info_text_en = f"- {name}"
                med_info_text_cn = f"- {self.config.translate_to_chinese(name)}"

            med_text_en += med_info_text_en + "\n"
            med_text_cn += med_info_text_cn + "\n"

        # Add medication info to responses
        response_en += med_text_en
        response_cn += med_text_cn

        return f"{response_en}\n\n{response_cn}"

    def build_test_recommendation(self):
        """Build test recommendation response (bilingual)"""
        # Generate test recommendations based on knowledge base training
        test_recommendation = self.knowledge_base.recommend_tests(self.current_symptoms)

        # Build English response
        response_en = (
            f"After {self.config.max_attempts} inquiries, no definitive diagnosis reached.\n"
            f"Current symptoms: {', '.join(self.current_symptoms)}\n\n"
            "Recommended diagnostic tests:\n"
            f"{test_recommendation}"
        )

        # Build Chinese response
        response_cn = self.config.translate_to_chinese(response_en)

        return f"{response_en}\n\n{response_cn}"

    def reset_session(self):
        """Reset session state"""
        self.current_symptoms = []
        self.attempt_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")


# ========== MAIN ==========
def main():
    try:
        # Initialize configuration
        config = MedicalConfig()

        # Load knowledge base
        logger.info("\n[1/2] Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # Initialize medical assistant
        logger.info("\n[2/2] Starting Medical Assistant")
        assistant = MedicalAssistant(knowledge_base, config)

        # Interactive interface
        logger.info("\n=== MEDICAL DIAGNOSTIC ASSISTANT (FOR PHYSICIANS) ===")
        logger.info("Enter patient symptoms for diagnosis or 'exit' to quit")
        logger.info(f"Diagnosis threshold: {config.diagnosis_threshold * 100}%")
        logger.info(f"Max inquiry attempts: {config.max_attempts}")
        logger.info("Supports Chinese and English input")

        while True:
            user_input = input("\nEnter symptoms: ").strip()

            if user_input.lower() == "exit":
                break

            response = assistant.analyze_symptoms(user_input)
            print(f"\nAssistant: {response}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()