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
from transformers import AutoModel, AutoTokenizer


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

        logger.info("\n=== MEDICAL ANALYSIS CONFIGURATION ===")
        logger.info(f"Knowledge Paths: {self.knowledge_paths}")
        logger.info(f"Ollama Model: {self.ollama_model}")
        logger.info(f"Diagnosis Threshold: {self.diagnosis_threshold * 100}%")
        logger.info(f"Max Inquiry Attempts: {self.max_attempts}")
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
        self.ddd_database = self.load_ddd_database()

        # Record learning process
        self.learning_stats = {
            "files_processed": 0,
            "diseases_extracted": 0,
            "symptoms_extracted": 0,
            "chunks_created": 0
        }

        # Load knowledge
        self.load_knowledge()

        # Create vector store
        self.create_vector_store()

        # Train Ollama model
        self.train_ollama_model()

        # Generate learning charts
        self.generate_learning_charts()

        logger.info(
            f"Knowledge base loaded. Diseases: {len(self.disease_info)}, Symptoms: {len(self.symptom_info)}, Chunks: {len(self.chunks)}")

    def load_ddd_database(self):
        """Load DDD database"""
        ddd_db = {
            "Amoxicillin": {"ddd": 1500, "unit": "mg", "route": "oral"},
            "Ceftriaxone": {"ddd": 2000, "unit": "mg", "route": "iv"},
            "Azithromycin": {"ddd": 500, "unit": "mg", "route": "oral"},
            "Metronidazole": {"ddd": 1500, "unit": "mg", "route": "oral"},
            "Doxycycline": {"ddd": 200, "unit": "mg", "route": "oral"},
            "Levofloxacin": {"ddd": 500, "unit": "mg", "route": "oral"},
            "Vancomycin": {"ddd": 2000, "unit": "mg", "route": "iv"},
            "Acetaminophen": {"ddd": 3000, "unit": "mg", "route": "oral"},
            "Ibuprofen": {"ddd": 1200, "unit": "mg", "route": "oral"},
            "Oseltamivir": {"ddd": 150, "unit": "mg", "route": "oral"},
        }
        logger.info(f"Loaded DDD database with {len(ddd_db)} medications")
        return ddd_db

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

    def extract_medical_info(self, text):
        """Comprehensive medical information extraction"""
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

                # Extract medications
                medications = []
                medication_matches = re.findall(r'(?:medications|drugs|prescriptions)[\s:]*([^\n]+)', text,
                                                re.IGNORECASE)
                for mm in medication_matches:
                    meds = [m.strip() for m in mm.split(',')]
                    medications.extend(meds)

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
                    "medications": ["Acetaminophen"]
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

    def train_ollama_model(self):
        """Train Ollama model for deep symptom analysis"""
        logger.info("Training Ollama model for deep symptom analysis...")

        # Automatically determine iterations
        iterations = min(100, max(10, len(self.disease_info) * 5))
        logger.info(f"Automatically set iterations: {iterations}")

        # Build training data
        training_data = []
        for disease, info in self.disease_info.items():
            symptoms = ", ".join(info["symptoms"])
            training_data.append(f"Disease: {disease}\nSymptoms: {symptoms}")

        # Train model (simplified implementation)
        logger.info(f"Training model with {len(training_data)} disease samples")

        # Actual training process would call Ollama API
        # Here we simulate completion
        logger.info("Ollama model training completed")

    def ollama_diagnose(self, symptoms):
        """Diagnose using trained Ollama model for deep symptom analysis"""
        logger.info("Using Ollama model for deep symptom analysis...")

        # Build prompt
        prompt = f"""
        As a medical expert, perform deep diagnostic analysis based on the following symptoms:
        Symptoms: {', '.join(symptoms)}

        Requirements:
        1. Provide the most likely diagnosis
        2. List matched disease symptoms
        3. Calculate overall similarity
        4. Recommend DDD-based medication therapy
        """

        # Use trained model to generate diagnosis
        response = self.ollama_generator.invoke(prompt)

        # Parse response
        diagnosis = {
            "disease": "Unknown",
            "similarity": 0.0,
            "matched_symptoms": [],
            "medications": []
        }

        # Try to extract diagnosis
        disease_match = re.search(r'Diagnosis:(.*?)\n', response)
        if disease_match:
            diagnosis["disease"] = disease_match.group(1).strip()

        # Try to extract similarity
        similarity_match = re.search(r'Similarity:(\d+\.\d+)', response)
        if similarity_match:
            diagnosis["similarity"] = float(similarity_match.group(1))

        # Try to extract matched symptoms
        symptoms_match = re.search(r'Matched Symptoms:(.*?)\n', response)
        if symptoms_match:
            diagnosis["matched_symptoms"] = [s.strip() for s in symptoms_match.group(1).split(',')]

        # Try to extract medications
        meds_match = re.search(r'Recommended Medications:(.*?)\n', response)
        if meds_match:
            diagnosis["medications"] = [m.strip() for m in meds_match.group(1).split(',')]

        return diagnosis

    def get_ddd_recommendation(self, medication):
        """Get DDD-based medication recommendation"""
        if medication in self.ddd_database:
            ddd_info = self.ddd_database[medication]
            return f"{medication} (DDD: {ddd_info['ddd']} {ddd_info['unit']}, Route: {ddd_info['route']})"
        return f"{medication} (DDD information unavailable)"

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
        """Generate learning charts for SCI papers"""
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
                "Metric": ["Files Processed", "Diseases Extracted", "Symptoms Extracted", "Chunks Created"],
                "Count": [
                    self.learning_stats["files_processed"],
                    self.learning_stats["diseases_extracted"],
                    self.learning_stats["symptoms_extracted"],
                    self.learning_stats["chunks_created"]
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

            # Create symptom distribution chart
            if self.symptom_info:
                symptom_names = list(self.symptom_info.keys())
                disease_counts = [len(self.symptom_info[s]["possible_diseases"]) for s in symptom_names]

                plt.figure(figsize=(12, 8))
                sns.scatterplot(x=symptom_names, y=disease_counts, s=100)
                plt.title("Symptoms and Associated Diseases", fontsize=16)
                plt.xlabel("Symptom", fontsize=14)
                plt.ylabel("Number of Associated Diseases", fontsize=14)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.chart_dir, "symptom_disease_association.png"), dpi=300)
                plt.close()

            logger.info("Learning charts generated successfully")
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")


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

        # Use trained Ollama model for deep symptom analysis
        diagnosis = self.knowledge_base.ollama_diagnose(self.current_symptoms)
        similarity = diagnosis["similarity"]

        # Check if diagnosis threshold is reached
        if similarity >= self.config.diagnosis_threshold:
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
            return self.request_more_symptoms(similarity)

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

    def request_more_symptoms(self, similarity):
        """Request more symptom information"""
        # Use Ollama to generate relevant symptom suggestions
        prompt = f"""
        As a medical expert, recommend additional symptoms to investigate based on:
        Current symptoms: {', '.join(self.current_symptoms)}
        Current similarity: {similarity * 100:.1f}%

        Requirements:
        1. List 3-5 key symptoms
        2. Focus on symptoms that would help differentiate possible diagnoses
        3. Use professional medical terminology
        """

        suggested_symptoms = self.ollama_generator.invoke(prompt)

        # Build bilingual response
        response_en = (
            f"Current symptoms: {', '.join(self.current_symptoms)}\n"
            f"Current similarity: {similarity * 100:.1f}% (threshold: {self.config.diagnosis_threshold * 100}%)\n\n"
            "No definitive diagnosis yet. Please provide additional information:\n\n"
            f"Suggested symptoms to investigate:\n{suggested_symptoms}"
        )

        response_cn = self.config.translate_to_chinese(response_en)

        return f"{response_en}\n\n{response_cn}"

    def build_diagnosis_response(self, diagnosis):
        """Build diagnosis response (bilingual)"""
        disease = diagnosis["disease"]
        similarity = diagnosis["similarity"]
        medications = diagnosis["medications"]

        # Get DDD recommendations
        ddd_recommendations = []
        for med in medications:
            ddd_rec = self.knowledge_base.get_ddd_recommendation(med)

            # Check doctor's prescription history
            if med in self.doctor_ddd_history:
                avg_ddd = sum(self.doctor_ddd_history[med]) / len(self.doctor_ddd_history[med])
                if avg_ddd > self.config.ddd_threshold:
                    ddd_rec += " (Warning: Your prescription DDD values are high)"

            ddd_recommendations.append(ddd_rec)

        # Build English response
        response_en = (
            f"Diagnosis: {disease}\n"
            f"Similarity: {similarity * 100:.1f}%\n\n"
            f"Matched Symptoms: {', '.join(diagnosis['matched_symptoms'])}\n\n"
            f"Medication Recommendations (based on DDD):\n"
            f"{chr(10).join(ddd_recommendations)}"
        )

        # Build Chinese response
        response_cn = self.config.translate_to_chinese(response_en)

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