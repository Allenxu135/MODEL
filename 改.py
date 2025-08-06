import os
import time
import json
import re
import numpy as np
import psutil
from tqdm import tqdm
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_ollama import OllamaEmbeddings
import docx
import csv
import GPUtil
import subprocess
import faiss
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from langdetect import detect, LangDetectException
from langchain_community.llms import Ollama


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
        # Set Faiss to use CPU-GPU hybrid mode
        self.faiss_gpu = True if torch.cuda.is_available() else False
        if self.faiss_gpu:
            print("Enabling Faiss GPU acceleration")
            self.faiss_res = faiss.StandardGpuResources()
        else:
            print("Using Faiss CPU only")

        # Ollama model path
        self.ollama_model_path = os.path.join(os.getcwd(), "ollama_models")
        os.makedirs(self.ollama_model_path, exist_ok=True)
        print(f"Ollama models stored at: {self.ollama_model_path}")

        # Knowledge base path
        self.knowledge_paths = self.setup_knowledge_paths()

        # File processing
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Output configuration
        self.model_name = "MEDICAL_ANALYSIS"
        self.output_dir = "MEDICAL_ANALYSIS_MODEL"
        os.makedirs(self.output_dir, exist_ok=True)

        # Multi-language support
        self.supported_languages = ["en", "es", "fr", "de", "zh", "ru", "ar", "hi"]
        self.current_lang = "en"

        # Ollama model configuration
        self.ollama_generation_models = self.detect_ollama_models()
        self.active_generation_model = "llama3" if "llama3" in self.ollama_generation_models else \
            self.ollama_generation_models[0] if self.ollama_generation_models else None
        self.generation_temp = 0.7
        self.generation_top_p = 0.9
        self.context_window = 4096
        self.enable_knowledge_distillation = True

        # Hybrid parameters
        self.hybrid_training = True
        self.faiss_parallel = min(multiprocessing.cpu_count(), 8)

        # Load models
        self.embedding_model = self.load_ollama_embeddings()
        self.generator = OllamaGenerator(self)

        print("\n=== MEDICAL ANALYSIS CONFIGURATION ===")
        print(f"Knowledge Paths: {self.knowledge_paths}")
        print(f"Supported Languages: {self.supported_languages}")
        print(f"Hybrid Processing: {self.hybrid_training}")
        print(f"Faiss Parallel: {self.faiss_parallel} threads")
        print(f"Available OLLAMA Models: {self.ollama_generation_models}")
        print(f"Active Generation Model: {self.active_generation_model}")
        print("=====================================")

    def detect_ollama_models(self):
        """Detect locally available OLLAMA models"""
        print("Detecting local OLLAMA models...")
        available_models = []

        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            lines = result.stdout.split('\n')

            for line in lines[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        model_name = parts[0]
                        available_models.append(model_name)
                        print(f"Found OLLAMA model: {model_name}")

            if not available_models:
                print("No OLLAMA models found. Install models with 'ollama pull <model>'")
        except Exception as e:
            print(f"Error detecting OLLAMA models: {str(e)}")
            print("Ensure OLLAMA is installed and running")

        return available_models

    def set_language(self, lang):
        """Set current language for responses"""
        if lang in self.supported_languages:
            self.current_lang = lang
            print(f"Language set to: {lang}")
        else:
            print(f"Unsupported language: {lang}. Keeping current: {self.current_lang}")

    def setup_knowledge_paths(self):
        """Set knowledge base paths to language-specific folders"""
        knowledge_base = os.path.join(os.getcwd(), "knowledge_base")
        language_paths = {}

        # Create language-specific folders if they don't exist
        for lang in self.supported_languages:
            lang_path = os.path.join(knowledge_base, lang)
            os.makedirs(lang_path, exist_ok=True)
            language_paths[lang] = lang_path
            print(f"Knowledge path for {lang}: {lang_path}")

        return language_paths

    def load_ollama_embeddings(self):
        """Load Ollama embedding model"""
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

    def detect_language(self, text):
        """Detect language of text"""
        try:
            return detect(text)
        except LangDetectException:
            return self.current_lang


# ========== OLLAMA GENERATOR ==========
class OllamaGenerator:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.load_models()

    def load_models(self):
        """Load all OLLAMA generation models"""
        for model_name in self.config.ollama_generation_models:
            try:
                self.models[model_name] = Ollama(
                    model=model_name,
                    base_url=self.config.ollama_model_path,
                    temperature=self.config.generation_temp,
                    top_p=self.config.generation_top_p
                )
                print(f"Loaded OLLAMA model: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {str(e)}")

    def generate(self, prompt, model_name=None, target_lang=None):
        """Generate text using specified model in target language"""
        model = model_name or self.config.active_generation_model
        if model not in self.models:
            print(f"Model {model} not available. Using default.")
            model = self.config.active_generation_model

        try:
            # Add language instruction if target language is specified
            if target_lang and target_lang in self.config.supported_languages:
                prompt = f"Respond in {target_lang} language.\n\n{prompt}"

            response = self.models[model].invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ""

    def set_active_model(self, model_name):
        """Set current active model"""
        if model_name in self.models:
            self.config.active_generation_model = model_name
            print(f"Active model set to: {model_name}")
            return True
        return False

    def translate(self, text, source_lang, target_lang):
        """Translate text between languages"""
        if source_lang == target_lang:
            return text

        prompt = f"Translate the following medical text from {source_lang} to {target_lang}:\n\n{text}"
        return self.generate(prompt, target_lang=target_lang)


# ========== KNOWLEDGE BASE ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.vector_store = None
        self.chunks = []
        self.disease_info = {}
        self.symptom_info = {}
        self.total_chunks = 0
        self.generator = config.generator
        self.faiss_index = None
        self.language_index = {}  # Index of chunks by language

        # DDD database for medications
        self.ddd_database = self.load_ddd_database()
        print(f"Loaded DDD database with {len(self.ddd_database)} medications")

        # Load knowledge
        self.load_knowledge()

        # Create vector store
        self.create_vector_store()
        print(
            f"Knowledge base loaded. Diseases: {len(self.disease_info)}, Symptoms: {len(self.symptom_info)}, Chunks: {len(self.chunks)}")

    def load_ddd_database(self):
        """Load DDD database for medications"""
        ddd_db = {}
        return ddd_db

    def get_ddd_info(self, medication_name):
        """Get DDD information for a medication"""
        for name, info in self.ddd_database.items():
            if name.lower() == medication_name.lower():
                return info

        for name, info in self.ddd_database.items():
            if medication_name.lower() in name.lower():
                return info

        return {"ddd": None, "route": "oral", "unit": "mg"}

    def load_file(self, file_path):
        """Load a single knowledge file with language detection"""
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

            # Detect language from file path or content
            lang = self.detect_language_from_path(file_path) or self.config.detect_language(content)
            return [{"content": content, "language": lang}]
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    def detect_language_from_path(self, file_path):
        """Detect language from file path"""
        for lang in self.config.supported_languages:
            if f"/{lang}/" in file_path.replace("\\", "/"):
                return lang
        return None

    def extract_medical_info(self, text, lang):
        """Extract medical information from text"""
        try:
            # Disease information extraction
            disease_pattern = re.compile(r'(?:disease|condition|illness)[\s:]*([^\n]+)', re.IGNORECASE)
            symptom_pattern = re.compile(r'(?:symptoms|signs)[\s:]*([^\n]+)', re.IGNORECASE)
            treatment_pattern = re.compile(r'(?:treatments|therapy|management)[\s:]*([^\n]+)', re.IGNORECASE)
            medication_pattern = re.compile(r'(?:medications|drugs|prescriptions)[\s:]*([^\n]+)', re.IGNORECASE)

            # Extract disease information
            disease_match = disease_pattern.search(text)
            if disease_match:
                disease_name = disease_match.group(1).strip().split('\n')[0].split(',')[0].strip()

                # Extract symptoms
                symptoms = []
                symptom_match = symptom_pattern.search(text)
                if symptom_match:
                    symptoms = [s.strip() for s in symptom_match.group(1).split(',')]

                # Extract treatments
                treatments = []
                treatment_match = treatment_pattern.search(text)
                if treatment_match:
                    treatments = [t.strip() for t in treatment_match.group(1).split(',')]

                # Extract medications with DDD info
                medications = []
                medication_match = medication_pattern.search(text)
                if medication_match:
                    meds = [m.strip() for m in medication_match.group(1).split(',')]
                    for med in meds:
                        ddd_info = self.get_ddd_info(med)
                        medications.append({
                            "name": med,
                            "ddd": ddd_info["ddd"],
                            "route": ddd_info["route"],
                            "unit": ddd_info["unit"]
                        })

                # Save disease information with language
                if lang not in self.disease_info:
                    self.disease_info[lang] = {}
                self.disease_info[lang][disease_name] = {
                    "symptoms": symptoms,
                    "treatments": treatments,
                    "medications": medications
                }

            # Symptom information extraction
            symptom_names = set()
            for line in text.split('\n'):
                if "symptom" in line.lower() or "sign" in line.lower():
                    parts = re.split(r'[:]', line, maxsplit=1)
                    if len(parts) > 1:
                        symptoms = [s.strip() for s in parts[1].split(',')]
                        symptom_names.update(symptoms)
                    else:
                        possible_symptoms = re.findall(r'\b\w+\s+\w+\b|\b\w+\b', line)
                        symptom_names.update(possible_symptoms)

            # Save symptom information with language
            if lang not in self.symptom_info:
                self.symptom_info[lang] = {}
            for symptom in symptom_names:
                if symptom and symptom not in self.symptom_info[lang]:
                    self.symptom_info[lang][symptom] = {
                        "description": "",
                        "possible_diseases": []
                    }

            return True
        except Exception as e:
            print(f"Error extracting medical info: {str(e)}")
            return False

    def enhance_knowledge(self):
        """Enhance knowledge base using OLLAMA models"""
        if not self.config.enable_knowledge_distillation or not self.generator.models:
            print("Knowledge distillation disabled or no models available")
            return

        print("Enhancing knowledge with OLLAMA models...")
        enhanced_chunks = []

        for chunk in tqdm(self.chunks[:5], desc="Enhancing knowledge"):
            lang = chunk["language"]
            content = chunk["content"]

            prompt = f"""
            As a medical expert, generate a detailed and structured medical knowledge description 
            based on the following fragment:

            Original content: {content}

            Requirements:
            1. Describe disease, symptoms, and treatments in a clear structure
            2. Add relevant medical background knowledge
            3. Use professional medical language
            4. Include DDD information for medications if available
            5. Respond in {lang} language
            """

            enhanced = self.generator.generate(prompt, target_lang=lang)
            if enhanced:
                enhanced_chunks.append({"content": enhanced, "language": lang})

        # Add enhanced content to knowledge base
        self.chunks.extend(enhanced_chunks)
        print(f"Added {len(enhanced_chunks)} enhanced knowledge chunks")

    def load_knowledge(self):
        """Load all knowledge base files with language support"""
        print("Loading medical knowledge...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        for lang, path in self.config.knowledge_paths.items():
            if not os.path.exists(path):
                print(f"Knowledge path not found: {path}")
                continue

            print(f"Processing directory: {path} for language: {lang}")
            file_count = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file_path.endswith(ext) for ext in ('.txt', '.csv', '.pdf', '.docx', '.json')):
                        print(f"Processing file: {file_path}")
                        try:
                            file_contents = self.load_file(file_path)

                            for content_info in file_contents:
                                content = content_info["content"]
                                file_lang = content_info["language"]

                                self.extract_medical_info(content, file_lang)
                                chunks = splitter.split_text(content)

                                for chunk in chunks:
                                    self.chunks.append({
                                        "content": chunk,
                                        "language": file_lang
                                    })
                                    self.total_chunks += 1

                                # Index by language
                                if file_lang not in self.language_index:
                                    self.language_index[file_lang] = []
                                self.language_index[file_lang].extend(chunks)

                            file_count += 1
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")

            print(f"Processed {file_count} files in {path} for language {lang}")

        # Enhance knowledge with OLLAMA models
        self.enhance_knowledge()

    def create_faiss_index(self, embeddings):
        """Create Faiss index for hybrid CPU-GPU search"""
        print("Creating Faiss index for hybrid search...")
        dim = embeddings.shape[1]

        if len(embeddings) < 10000:
            index = faiss.IndexFlatL2(dim)
        else:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, min(100, len(embeddings) // 10), faiss.METRIC_L2)
            index.train(embeddings)

        if self.config.faiss_gpu:
            gpu_index = faiss.index_cpu_to_gpu(self.config.faiss_res, 0, index)
            gpu_index.add(embeddings)
            self.faiss_index = gpu_index
            print("Faiss index created on GPU")
        else:
            faiss.omp_set_num_threads(self.config.faiss_parallel)
            index.add(embeddings)
            self.faiss_index = index
            print(f"Faiss index created on CPU with {self.config.faiss_parallel} threads")

    def create_vector_store(self):
        """Create vector store with FAISS and Faiss hybrid"""
        if not self.chunks or not self.config.embedding_model:
            print("No knowledge chunks or embedding model, skipping vector store")
            return

        print(f"Creating vector store with {len(self.chunks)} chunks...")
        try:
            # Create FAISS store from text content only
            texts = [chunk["content"] for chunk in self.chunks]
            self.vector_store = FAISS.from_texts(texts, self.config.embedding_model)
            print("FAISS vector store created")

            embeddings = np.array([self.config.embedding_model.embed_query(chunk["content"]) for chunk in
                                   tqdm(self.chunks, desc="Embedding chunks")])
            self.create_faiss_index(embeddings.astype('float32'))

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            self.vector_store = None
            self.faiss_index = None

    def hybrid_search(self, query, k=3, lang=None):
        """Perform hybrid CPU-GPU search using Faiss with language filtering"""
        if not self.faiss_index:
            print("Faiss index not available, using simple search")
            return []

        try:
            query_embedding = np.array([self.config.embedding_model.embed_query(query)])
            query_embedding = query_embedding.astype('float32')

            distances, indices = self.faiss_index.search(query_embedding, k * 3)  # Get more results for filtering

            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    chunk = self.chunks[idx]
                    # Apply language filter if specified
                    if not lang or chunk["language"] == lang:
                        results.append({
                            "content": chunk["content"],
                            "language": chunk["language"],
                            "distance": distances[0][i]
                        })
                    if len(results) >= k:
                        break

            return results[:k]
        except Exception as e:
            print(f"Hybrid search error: {str(e)}")
            return []

    def diagnose(self, symptoms, lang=None):
        """Diagnose disease based on symptoms with language support"""
        possible_diseases = {}

        if not symptoms:
            return []

        # First try simple matching in the specified language
        target_lang = lang or self.config.current_lang
        disease_info = self.disease_info.get(target_lang, {})

        for symptom in symptoms:
            for disease, info in disease_info.items():
                if any(symptom.lower() in s.lower() for s in info["symptoms"]):
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1
                elif symptom.lower() in disease.lower():
                    possible_diseases[disease] = possible_diseases.get(disease, 0) + 1

        if possible_diseases:
            results = []
            for disease, count in possible_diseases.items():
                total_symptoms = len(disease_info[disease]["symptoms"])
                if total_symptoms == 0:
                    total_symptoms = 1
                probability = min(count / total_symptoms, 1.0) * 100
                results.append({
                    "disease": disease,
                    "probability": round(probability, 1),
                    "matched_symptoms": count,
                    "total_symptoms": total_symptoms,
                    "language": target_lang
                })

            return sorted(results, key=lambda x: x["probability"], reverse=True)

        # If no matches found, try hybrid search with language filtering
        if self.faiss_index:
            try:
                print("Using hybrid CPU-GPU search with language filtering...")
                query = "Symptoms: " + ", ".join(symptoms)
                results = self.hybrid_search(query, k=3, lang=target_lang)

                for result in results:
                    content = result["content"]
                    # Try to find disease names in the content
                    disease_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                    for disease in disease_matches:
                        if disease in disease_info:
                            possible_diseases[disease] = possible_diseases.get(disease, 0) + 1

                if possible_diseases:
                    results = []
                    for disease, count in possible_diseases.items():
                        results.append({
                            "disease": disease,
                            "probability": min(count * 30, 80),
                            "matched_symptoms": 0,
                            "total_symptoms": 0,
                            "method": "hybrid",
                            "language": target_lang
                        })
                    return sorted(results, key=lambda x: x["probability"], reverse=True)

            except Exception as e:
                print(f"Error in hybrid search: {str(e)}")

        return []

    def get_treatment_plan(self, disease, lang=None):
        """Get treatment plan for a disease in specified language"""
        target_lang = lang or self.config.current_lang
        disease_info = self.disease_info.get(target_lang, {})

        if disease in disease_info:
            return {
                "treatments": disease_info[disease]["treatments"],
                "medications": disease_info[disease]["medications"],
                "language": target_lang
            }
        return {"treatments": [], "medications": [], "language": target_lang}


# ========== MEDICAL ASSISTANT ==========
class MedicalAssistant:
    def __init__(self, knowledge_base, config):
        self.config = config
        self.knowledge_base = knowledge_base
        self.conversation_context = []
        self.dialog_history = []
        self.max_history = 3
        self.patient_weight = None

    def calculate_ddd_recommendation(self, medication, duration_days=5):
        """Calculate medication recommendation based on DDD"""
        if not medication or not medication.get("ddd"):
            return "Standard dosing", ""

        ddd = medication["ddd"]
        unit = medication["unit"]
        route = medication["route"]

        if self.patient_weight:
            adjusted_ddd = ddd * (self.patient_weight / 70)
            daily_dose = f"{adjusted_ddd:.1f} {unit}"
            explanation = f"Adjusted for patient weight ({self.patient_weight}kg)"
        else:
            daily_dose = f"{ddd} {unit}"
            explanation = "Standard DDD (based on 70kg adult)"

        total_dose = ddd * duration_days
        if self.patient_weight:
            total_dose = total_dose * (self.patient_weight / 70)

        recommendation = (
            f"- Daily dose: {daily_dose} ({route})\n"
            f"- Course duration: {duration_days} days\n"
            f"- Total course: {total_dose:.1f} {unit}\n"
            f"Note: {explanation}"
        )

        return recommendation

    def extract_weight(self, text):
        """Extract weight information from text"""
        patterns = [
            r'(\d+)\s*kgs?\b',
            r'(\d+)\s*kg\b',
            r'(\d+)\s*kilos?\b',
            r'weight\s*[:]?\s*(\d+)\s*(kg|kgs|kilo|kilos)',
            r'(\d+)\s*lbs?\b',
            r'(\d+)\s*pounds?\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                weight = float(match.group(1))

                if "lb" in match.group(0).lower() or "pound" in match.group(0).lower():
                    weight = weight * 0.453592

                return round(weight, 1)

        return None

    def analyze_symptoms(self, symptoms_text, lang=None):
        """Analyze symptoms and provide medical advice in specified language"""
        target_lang = lang or self.config.current_lang
        self.config.set_language(target_lang)

        self.conversation_context.append(symptoms_text)
        self.patient_weight = self.extract_weight(symptoms_text)
        symptoms = self.extract_symptoms(symptoms_text)
        diagnoses = self.knowledge_base.diagnose(symptoms, lang=target_lang)

        if not diagnoses:
            if len(self.conversation_context) > 2:
                return self.generator.translate(
                    "Based on current knowledge, diagnosis cannot be determined. Please update knowledge base.",
                    "en", target_lang
                )
            else:
                return self.generator.translate(
                    "Please describe symptoms in more detail.",
                    "en", target_lang
                )

        top_diagnosis = diagnoses[0]
        disease = top_diagnosis["disease"]
        treatment = self.knowledge_base.get_treatment_plan(disease, lang=target_lang)

        # Build response in target language
        response = self.generator.translate(
            f"Based on symptom analysis, possible diagnosis: {disease} (confidence {top_diagnosis['probability']}%).",
            "en", target_lang
        )

        response += "\n\n" + self.generator.translate(
            "Recommended treatments:",
            "en", target_lang
        ) + "\n" + self.format_list(treatment["treatments"], target_lang) + "\n\n"

        if treatment["medications"]:
            response += self.generator.translate(
                "Medication recommendations (WHO DDD guidelines):",
                "en", target_lang
            ) + "\n"

            for med in treatment["medications"]:
                med_name = med["name"]
                ddd_rec = self.calculate_ddd_recommendation(med)

                # Translate medication recommendation
                ddd_rec_translated = self.generator.translate(ddd_rec, "en", target_lang)

                if med.get("ddd"):
                    ddd_info = f" (WHO DDD: {med['ddd']} {med['unit']}/{med['route']})"
                else:
                    ddd_info = ""

                response += f"- {med_name}{ddd_info}:\n{ddd_rec_translated}\n\n"

        if not self.patient_weight:
            response += "\n" + self.generator.translate(
                "Note: Doses calculated for standard 70kg adult. Provide weight for personalized dosing.",
                "en", target_lang
            )

        response += "\n" + self.generator.translate(
            "Note: AI-generated advice. Consult medical professional.",
            "en", target_lang
        )

        return response

    def chat(self, user_input, lang=None):
        """Handle multi-turn dialog in specified language"""
        target_lang = lang or self.config.current_lang
        self.config.set_language(target_lang)

        self.dialog_history.append(f"User: {user_input}")
        if len(self.dialog_history) > self.max_history * 2:
            self.dialog_history = self.dialog_history[-self.max_history * 2:]

        context = "\n".join(self.dialog_history) + "\nAssistant:"

        # Generate response in target language
        response = self.config.generator.generate(context, target_lang=target_lang)

        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        self.dialog_history.append(f"Assistant: {response}")
        return response

    def extract_symptoms(self, text):
        """Extract symptom keywords from text"""
        found_symptoms = []

        # Get symptoms for current language
        target_lang = self.config.current_lang
        symptom_info = self.knowledge_base.symptom_info.get(target_lang, {})

        for symptom in symptom_info.keys():
            if symptom.lower() in text.lower():
                found_symptoms.append(symptom)

        if not found_symptoms:
            symptom_words = set()
            for word in text.split():
                symptom_keywords = ["pain", "ache", "cough", "fever", "vomit", "dizzy", "bleed", "itch", "swell"]
                if any(keyword in word for keyword in symptom_words) or any(
                        keyword in word for keyword in symptom_keywords):
                    symptom_words.add(word)

            found_symptoms = list(symptom_words)

        return found_symptoms

    def format_list(self, items, lang=None):
        """Format a list of items for display in specified language"""
        if not items:
            return self.generator.translate("None", "en", lang or self.config.current_lang)

        formatted = "\n".join([f"- {item}" for item in items])
        return formatted

    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = []
        self.dialog_history = []
        self.patient_weight = None


# ========== MAIN ==========
def main():
    try:
        # Initialize configuration
        config = MedicalConfig()

        # Load knowledge base
        print("\n[1/2] Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # Initialize medical assistant
        print("\n[2/2] Starting Medical Assistant")
        assistant = MedicalAssistant(knowledge_base, config)

        # Interactive interface
        print("\n=== MEDICAL ANALYSIS ASSISTANT ===")
        print("Commands: lang [en/es/fr/de/zh/ru/ar/hi], clear, exit, /model [name], /set [param] [value]")
        print(f"Available OLLAMA models: {config.ollama_generation_models}")
        print(f"Supported languages: {config.supported_languages}")

        while True:
            try:
                user_input = input("\nPatient: ").strip()

                if user_input.lower() == "exit":
                    break
                elif user_input.lower().startswith("lang "):
                    lang = user_input.split()[1].lower()
                    config.set_language(lang)
                    continue
                elif user_input.lower() == "clear":
                    assistant.clear_context()
                    print("Conversation context cleared")
                    continue
                elif user_input.startswith("/model "):
                    model_name = user_input.split()[1]
                    if config.generator.set_active_model(model_name):
                        print(f"Model switched to: {model_name}")
                    else:
                        print(f"Model not available. Options: {config.ollama_generation_models}")
                    continue
                elif user_input.startswith("/set "):
                    try:
                        parts = user_input.split()
                        param = parts[1]
                        value = parts[2]

                        if param == "temp":
                            config.generation_temp = float(value)
                            print(f"Temperature set to {value}")
                        elif param == "topp":
                            config.generation_top_p = float(value)
                            print(f"Top-p set to {value}")
                        else:
                            print("Invalid parameter. Options: temp, topp")
                    except:
                        print("Invalid command. Usage: /set [param] [value]")
                    continue

                start_time = time.time()

                # Detect input language
                try:
                    input_lang = detect(user_input)
                    if input_lang not in config.supported_languages:
                        input_lang = config.current_lang
                except:
                    input_lang = config.current_lang

                if "symptom" in user_input.lower() or "pain" in user_input.lower() or "feel" in user_input.lower():
                    response = assistant.analyze_symptoms(user_input, lang=input_lang)
                else:
                    response = assistant.chat(user_input, lang=input_lang)

                response_time = time.time() - start_time

                print(f"\nAssistant: {response}")
                print(f"Response time: {response_time:.2f}s")

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