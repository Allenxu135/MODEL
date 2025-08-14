import os
import re
import json
import logging
import difflib
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from googletrans import Translator
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
import Levenshtein


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

        # Model directory
        self.model_dir = "trained_models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model directory: {self.model_dir}")

        # Ollama model configuration
        self.ollama_base_url = "http://localhost:11434"
        self.ollama_model = "llama3"
        self.generation_temp = 0.7
        self.generation_top_p = 0.9
        self.diagnosis_threshold = 0.95  # 95% confidence threshold
        self.max_attempts = 2  # Maximum inquiry attempts

        # Training configuration
        self.epochs = 4
        self.batch_size = 4
        self.learning_rate = 2e-5
        self.ddd_threshold = 1.0  # High DDD value threshold

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

    def translate_to_english(self, text):
        """Translate text to English"""
        try:
            # Skip translation if already in English
            if self.is_english(text):
                return text
            return Translator().translate(text, dest='en').text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def translate_to_chinese(self, text):
        """Translate text to Chinese"""
        try:
            # Skip translation if already in Chinese
            if self.is_chinese(text):
                return text
            return Translator().translate(text, dest='zh-cn').text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return text

    def is_english(self, text):
        """Check if text is English"""
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def is_chinese(self, text):
        """Check if text is Chinese"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)


# ========== KNOWLEDGE BASE ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.disease_info = {}
        self.symptom_info = {}
        self.medication_ddd_info = {}
        self.learning_stats = {
            "files_processed": 0,
            "diseases_extracted": 0,
            "symptoms_extracted": 0,
            "medications_extracted": 0,
            "tests_extracted": 0
        }

        # Load knowledge
        self.load_knowledge()
        logger.info(f"知识库加载完成: {len(self.disease_info)}种疾病, {len(self.symptom_info)}种症状")

    def extract_medical_info(self, text):
        """从文本中提取医疗信息 (支持多语言)"""
        try:
            # 疾病提取 (支持中英文)
            disease_pattern = r'(?:disease|condition|illness|diagnosis|疾病|病症|诊断)[\s:：]*([^\n]+)'
            disease_matches = re.findall(disease_pattern, text, re.IGNORECASE)

            for match in disease_matches:
                disease_name = match.strip().split('\n')[0].split(',')[0].strip()

                # 症状提取 (支持中英文)
                symptoms = []
                symptom_pattern = r'(?:symptoms|signs|complaint|症状|体征|不适)[\s:：]*([^\n]+)'
                symptom_matches = re.findall(symptom_pattern, text, re.IGNORECASE)
                for sm in symptom_matches:
                    symptoms.extend([s.strip() for s in re.split(r'[,，、]', sm)])

                # 药物提取 (支持中英文)
                medications = []
                medication_pattern = r'(?:medications|drugs|prescriptions|剂量|药物)[\s:：]*([^\n]+)'
                medication_matches = re.findall(medication_pattern, text, re.IGNORECASE)

                for mm in medication_matches:
                    for line in mm.split('\n'):
                        # 支持多种格式的药物描述
                        med_match = re.search(
                            r'([a-zA-Z\u4e00-\u9fff]+[\s\-]*[a-zA-Z\u4e00-\u9fff]*\d*)[\s(]*([\d.]+)?\s*([a-zA-Z\u4e00-\u9fff]*)?',
                            line)
                        if med_match:
                            name = med_match.group(1).strip()
                            dosage = med_match.group(2) if med_match.group(2) else ""
                            unit = med_match.group(3) if med_match.group(3) else ""

                            medications.append({
                                'name': name,
                                'dosage': dosage,
                                'unit': unit
                            })

                # 检查提取 (支持中英文)
                tests = []
                test_pattern = r'(?:tests|examinations|diagnostic procedures|检查|检验|检测)[\s:：]*([^\n]+)'
                test_matches = re.findall(test_pattern, text, re.IGNORECASE)
                for tm in test_matches:
                    tests.extend([t.strip() for t in re.split(r'[,，、]', tm)])

                # 保存疾病信息
                if disease_name and disease_name not in self.disease_info:
                    self.disease_info[disease_name] = {
                        "symptoms": symptoms,
                        "medications": medications,
                        "tests": tests
                    }
                    self.learning_stats["diseases_extracted"] += 1
                    self.learning_stats["medications_extracted"] += len(medications)
                    self.learning_stats["tests_extracted"] += len(tests)

            # 提取症状信息
            symptom_names = set()
            for symptom_list in [info["symptoms"] for info in self.disease_info.values()]:
                symptom_names.update(symptom_list)

            for symptom in symptom_names:
                if symptom and symptom not in self.symptom_info:
                    self.symptom_info[symptom] = {
                        "description": "",
                        "related_tests": []
                    }
                    self.learning_stats["symptoms_extracted"] += 1

            return True
        except Exception as e:
            logger.error(f"医疗信息提取错误: {str(e)}")
            return False

    def calculate_ddd(self, medication, dosage, unit, frequency):
        """计算DDD值 - 简化版"""
        # 1. 首先检查知识库中是否有该药物的DDD信息
        if medication in self.medication_ddd_info:
            ddd_info = self.medication_ddd_info[medication]
            return self._perform_ddd_calculation(ddd_info, dosage, unit), None

        # 2. 知识库中没有则尝试寻找替代药物
        alternatives = self.find_alternative_medications(medication)
        if alternatives:
            return None, alternatives  # 返回None表示需要换药

        # 3. 最后尝试预测DDD
        return self.predict_ddd_with_ollama(medication, dosage, unit, frequency), None

    def find_alternative_medications(self, medication):
        """在知识库中寻找替代药物"""
        alternatives = []
        for disease, info in self.disease_info.items():
            for med in info.get("medications", []):
                med_name = med["name"]
                # 相似药物匹配 (支持多语言)
                if self.is_similar_medication(medication, med_name) and med_name != medication:
                    alternatives.append(med_name)
        return list(set(alternatives))  # 去重

    def is_similar_medication(self, med1, med2):
        """检查药物是否相似 (支持多语言)"""
        # 翻译为英文后比较
        med1_en = self.config.translate_to_english(med1).lower()
        med2_en = self.config.translate_to_english(med2).lower()
        return SequenceMatcher(None, med1_en, med2_en).ratio() > 0.7

    def _perform_ddd_calculation(self, ddd_info, dosage, unit):
        """执行DDD计算 (简化版)"""
        try:
            # 实际应用中这里会有更复杂的计算逻辑
            return float(ddd_info.get("ddd_value", 0))
        except:
            return 0.0

    def predict_ddd_with_ollama(self, medication, dosage, unit, frequency):
        """当没有DDD信息时尝试预测"""
        # 在实际应用中，这里会调用Ollama进行预测
        # 简化版：返回0.0
        return 0.0, None

    def load_knowledge(self):
        """Load all knowledge base files from knowledge_base folder"""
        logger.info("Loading medical knowledge from local knowledge_base folder...")

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
                            content = self.load_file(file_path)

                            # Extract medical information
                            self.extract_medical_info(content)

                            file_count += 1
                            self.learning_stats["files_processed"] += 1
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {str(e)}")

            logger.info(f"Processed {file_count} files in {path}")

        # 完全移除任何默认知识添加
        if not self.disease_info:
            logger.warning("No diseases extracted from knowledge base files")
        if not self.symptom_info:
            logger.warning("No symptoms extracted from knowledge base files")

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
                import docx
                doc = docx.Document(file_path)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif file_path.endswith('.pdf'):
                from PyPDF2 import PdfReader
                with open(file_path, 'rb') as f:
                    pdf_reader = PdfReader(f)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return ""

            return content
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return ""


# ========== MEDICAL ASSISTANT ==========
class MedicalAssistant:
    def __init__(self, knowledge_base, config):
        self.knowledge_base = knowledge_base
        self.config = config
        self.thought_process = []  # 记录思考过程
        self.current_symptoms = []
        self.attempt_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    def diagnose(self, chief_complaint):
        """诊断流程 (类人脑思考过程)"""
        self.thought_process = [f"患者主诉: {chief_complaint}"]

        # 步骤1: 初步诊断
        diagnosis = self.initial_diagnosis(chief_complaint)
        self.thought_process.append(f"初步诊断: {diagnosis['disease']} (置信度: {diagnosis['confidence'] * 100:.1f}%)")

        # 步骤2: 知识库验证
        kb_match = self.check_knowledge_base_match(diagnosis['disease'])
        self.thought_process.append(f"知识库匹配: {kb_match['match']} (相似度: {kb_match['similarity'] * 100:.1f}%)")

        if kb_match['similarity'] < self.config.diagnosis_threshold:
            return self.handle_low_confidence(kb_match)

        # 步骤3: 用药推荐
        medication_response = self.recommend_medication(diagnosis['disease'])

        # 步骤4: 检查建议
        test_recommendation = self.recommend_tests(diagnosis['disease'])

        # 步骤5: 生成最终响应
        return self.generate_final_response(
            diagnosis,
            kb_match,
            medication_response,
            test_recommendation
        )

    def initial_diagnosis(self, chief_complaint):
        """初步诊断 (深度思考)"""
        # 使用Ollama进行初步诊断思考
        prompt = f"""
        作为医疗辅助模型，根据患者主诉进行诊断思考:
        主诉: {chief_complaint}

        思考步骤:
        1. 分析关键症状
        2. 考虑可能的鉴别诊断
        3. 评估最可能的疾病
        4. 输出JSON格式: {{"disease": "疾病名称", "confidence": 0.0-1.0}}
        """

        # 在实际应用中，这里会调用Ollama
        # 简化版：模拟响应
        simulated_response = '{"disease": "上呼吸道感染", "confidence": 0.92}'

        try:
            return json.loads(simulated_response)
        except:
            # 如果JSON解析失败，使用简单提取
            disease_match = re.search(r'[a-zA-Z\u4e00-\u9fff]+', simulated_response)
            return {
                "disease": disease_match.group(0) if disease_match else "未知疾病",
                "confidence": 0.8
            }

    def check_knowledge_base_match(self, disease):
        """检查知识库匹配度 (深度思考)"""
        # 获取知识库中所有疾病
        kb_diseases = list(self.knowledge_base.disease_info.keys())

        if not kb_diseases:
            return {"match": "知识库中无相关疾病信息", "similarity": 0.0}

        # 计算相似度
        best_match = ""
        best_similarity = 0.0

        for kb_disease in kb_diseases:
            # 使用多语言相似度计算
            similarity = self.calculate_multilingual_similarity(disease, kb_disease)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = kb_disease

        return {
            "match": best_match if best_similarity > 0.7 else "无匹配疾病",
            "similarity": best_similarity
        }

    def calculate_multilingual_similarity(self, text1, text2):
        """多语言相似度计算"""
        # 翻译为英文后比较
        text1_en = self.config.translate_to_english(text1)
        text2_en = self.config.translate_to_english(text2)

        # 使用编辑距离计算相似度
        return 1 - (Levenshtein.distance(text1_en, text2_en) / max(len(text1_en), len(text2_en)))

    def handle_low_confidence(self, kb_match):
        """处理低置信度情况"""
        # 在实际应用中，这里会请求更多症状信息
        # 简化版：直接返回信息
        self.thought_process.append("诊断置信度过低，请求更多症状信息")

        # 英文响应
        en_response = f"Diagnosis confidence is below threshold ({self.config.diagnosis_threshold * 100}%)\n"
        en_response += f"Best match in knowledge base: {kb_match['match']} (Similarity: {kb_match['similarity'] * 100:.1f}%)\n"
        en_response += "Please provide more symptom details."

        # 中文响应
        cn_response = f"诊断置信度低于阈值 ({self.config.diagnosis_threshold * 100}%)\n"
        cn_response += f"知识库最佳匹配: {self.config.translate_to_chinese(kb_match['match'])} (相似度: {kb_match['similarity'] * 100:.1f}%)\n"
        cn_response += "请提供更多症状细节。"

        # 添加思考过程
        thought_header = "\n\n=== THINKING PROCESS ===\n" + "\n".join(self.thought_process)
        thought_header_cn = "\n\n=== 思考过程 ===\n" + "\n".join(
            [self.config.translate_to_chinese(t) for t in self.thought_process])

        return f"{en_response}\n\n{cn_response}{thought_header}{thought_header_cn}"

    def recommend_medication(self, disease):
        """推荐药物 (类人脑思考过程)"""
        self.thought_process.append(f"为 {disease} 推荐药物...")

        # 获取知识库中的药物
        medications = self.knowledge_base.disease_info.get(disease, {}).get("medications", [])

        if not medications:
            return {"status": "no_medication", "message": "知识库中无相关药物信息"}

        # 计算DDD值
        results = []
        total_ddd = 0.0

        for med in medications:
            ddd_value, alternatives = self.knowledge_base.calculate_ddd(
                med["name"], med["dosage"], med["unit"], "daily"
            )

            if ddd_value is None:  # 需要换药
                if alternatives:
                    alt_text = ", ".join(alternatives[:3])
                    results.append({
                        "medication": med["name"],
                        "status": "need_alternative",
                        "message": f"无法计算DDD，建议换药: {alt_text}"
                    })
                else:
                    results.append({
                        "medication": med["name"],
                        "status": "no_ddd",
                        "message": "无法计算DDD且无替代药物"
                    })
            else:
                results.append({
                    "medication": med["name"],
                    "dosage": med["dosage"],
                    "unit": med["unit"],
                    "ddd": ddd_value,
                    "status": "success"
                })
                total_ddd += ddd_value

        return {
            "status": "success" if any(r["status"] == "success" for r in results) else "partial",
            "medications": results,
            "total_ddd": total_ddd
        }

    def recommend_tests(self, disease):
        """推荐检查 (基于知识库的深度思考)"""
        self.thought_process.append(f"为 {disease} 分析检查需求...")

        # 首先检查知识库中是否有该疾病的相关信息
        disease_info = self.knowledge_base.disease_info.get(disease, {})

        if not disease_info:
            # 知识库中没有该疾病信息
            self.thought_process.append(f"知识库中没有关于 {disease} 的信息")
            return None

        # 检查知识库中是否有明确的检查建议
        if "tests" in disease_info and disease_info["tests"]:
            tests = disease_info["tests"]
            self.thought_process.append(f"从知识库中找到 {len(tests)} 项检查建议")
            return tests

        # 知识库中没有明确的检查建议，尝试从症状中推断
        symptoms = disease_info.get("symptoms", [])
        inferred_tests = self.infer_tests_from_symptoms(symptoms)

        if inferred_tests:
            self.thought_process.append(f"从 {len(symptoms)} 个症状推断出 {len(inferred_tests)} 项检查")
            return inferred_tests

        # 没有任何可用的检查建议
        self.thought_process.append(f"无法为 {disease} 推荐任何检查")
        return None

    def infer_tests_from_symptoms(self, symptoms):
        """从症状推断检查项目 (基于知识库的深度思考)"""
        if not symptoms:
            return []

        # 从知识库中收集所有症状相关的检查
        symptom_test_mapping = {}
        for symptom, info in self.knowledge_base.symptom_info.items():
            if "related_tests" in info:
                symptom_test_mapping[symptom] = info["related_tests"]

        # 找出当前症状相关的检查
        recommended_tests = []
        for symptom in symptoms:
            # 在知识库中查找最匹配的症状
            best_match = symptom
            max_similarity = 0
            for kb_symptom in symptom_test_mapping.keys():
                similarity = self.calculate_symptom_similarity(symptom, kb_symptom)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = kb_symptom

            # 如果找到匹配的症状且有相关检查，则添加
            if max_similarity > 0.7 and best_match in symptom_test_mapping:
                recommended_tests.extend(symptom_test_mapping[best_match])

        # 去重并限制数量
        return list(set(recommended_tests))[:5]  # 最多返回5项

    def calculate_symptom_similarity(self, symptom1, symptom2):
        """计算症状相似度 (支持多语言)"""
        # 翻译为英文后比较
        symptom1_en = self.config.translate_to_english(symptom1)
        symptom2_en = self.config.translate_to_english(symptom2)

        # 使用编辑距离计算相似度
        return 1 - (Levenshtein.distance(symptom1_en, symptom2_en) / max(len(symptom1_en), len(symptom2_en)))

    def generate_final_response(self, diagnosis, kb_match, medication, tests):
        """生成最终双语响应"""
        # 英文部分
        en_response = f"Diagnosis: {diagnosis['disease']}\n"
        en_response += f"Confidence: {diagnosis['confidence'] * 100:.1f}%\n"
        en_response += f"Knowledge Base Match: {kb_match['match']} (Similarity: {kb_match['similarity'] * 100:.1f}%)\n\n"

        en_response += "Medication Recommendations:\n"
        if medication["status"] == "no_medication":
            en_response += "No medication information found in knowledge base\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    en_response += f"- {med['medication']}: {med['dosage']}{med['unit']} (DDD: {med['ddd']:.2f})\n"
                else:
                    en_response += f"- {med['medication']}: {med['message']}\n"

            if medication["total_ddd"] > 0:
                en_response += f"Total DDD: {medication['total_ddd']:.2f}\n"

        en_response += "\nRecommended Tests:\n"
        if tests:
            en_tests = "\n".join([f"- {test}" for test in tests])
            en_response += en_tests
        else:
            en_response += "No specific tests recommended based on current knowledge"

        # 中文部分 (翻译英文内容)
        cn_response = f"诊断: {self.config.translate_to_chinese(diagnosis['disease'])}\n"
        cn_response += f"置信度: {diagnosis['confidence'] * 100:.1f}%\n"
        cn_response += f"知识库匹配: {self.config.translate_to_chinese(kb_match['match'])} (相似度: {kb_match['similarity'] * 100:.1f}%)\n\n"

        cn_response += "药物推荐:\n"
        if medication["status"] == "no_medication":
            cn_response += "知识库中未找到药物信息\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    cn_response += f"- {self.config.translate_to_chinese(med['medication'])}: {med['dosage']}{self.config.translate_to_chinese(med['unit'])} (DDD值: {med['ddd']:.2f})\n"
                else:
                    cn_response += f"- {self.config.translate_to_chinese(med['medication'])}: {self.config.translate_to_chinese(med['message'])}\n"

            if medication["total_ddd"] > 0:
                cn_response += f"总DDD值: {medication['total_ddd']:.2f}\n"

        cn_response += "\n推荐检查:\n"
        if tests:
            cn_tests = "\n".join([f"- {self.config.translate_to_chinese(test)}" for test in tests])
            cn_response += cn_tests
        else:
            cn_response += "基于当前知识库，暂无特定检查建议"

        # 添加思考过程
        thought_header = "\n\n=== THINKING PROCESS ===\n" + "\n".join(self.thought_process)
        thought_header_cn = "\n\n=== 思考过程 ===\n" + "\n".join(
            [self.config.translate_to_chinese(t) for t in self.thought_process])

        return f"{en_response}\n\n{cn_response}{thought_header}{thought_header_cn}"


# ========== MAIN FUNCTION ==========
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

            response = assistant.diagnose(user_input)
            print(f"\nAssistant: {response}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"系统错误: {str(e)}")


if __name__ == "__main__":
    main()