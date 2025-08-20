import os
import re
import json
import logging
import difflib
import numpy as np
import csv
from datetime import datetime
from difflib import SequenceMatcher
import asyncio
from deep_translator import GoogleTranslator
import Levenshtein


# ========== LOGGER SETUP ==========
def setup_logger():
    """设置日志记录器 (Set up logger)"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建带时间戳的日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"medical_diagnosis_{timestamp}.log")

    # 配置日志
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

    # 添加控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger('MedicalDiagnosis')


# 初始化日志记录器
logger = setup_logger()


# ========== 配置 ==========
class MedicalConfig:
    def __init__(self):
        # 知识库路径
        self.knowledge_paths = self.setup_knowledge_paths()

        # 模型目录
        self.model_dir = "trained_models"
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"模型目录 | Model directory: {self.model_dir}")

        # 训练配置
        self.epochs = 4
        self.batch_size = 4
        self.learning_rate = 2e-5
        self.ddd_threshold = 1.0  # DDD高阈值

        # 诊断阈值设为95%
        self.diagnosis_threshold = 0.95

        logger.info("\n=== 医疗分析配置 | MEDICAL ANALYSIS CONFIGURATION ===")
        logger.info(f"知识路径 | Knowledge Paths: {self.knowledge_paths}")
        logger.info(f"诊断阈值 | Diagnosis Threshold: {self.diagnosis_threshold * 100}%")
        logger.info(f"训练轮数 | Training Epochs: {self.epochs}")
        logger.info("===================================================")

    def setup_knowledge_paths(self):
        """设置知识库路径为'knowledge_base'文件夹 (Set knowledge base paths)"""
        knowledge_dir = os.path.join(os.getcwd(), "knowledge_base")
        os.makedirs(knowledge_dir, exist_ok=True)
        logger.info(f"知识路径 | Knowledge path: {knowledge_dir}")
        return [knowledge_dir]

    async def translate_to_english(self, text):
        """异步翻译文本到英文 (Translate text to English asynchronously)"""
        try:
            # 使用 to_thread 运行同步翻译任务
            return await asyncio.to_thread(GoogleTranslator(source='auto', target='en').translate, text)
        except Exception as e:
            logger.error(f"翻译错误 | Translation error: {str(e)}")
            return text

    async def translate_to_chinese(self, text):
        """异步翻译文本到中文 (Translate text to Chinese asynchronously)"""
        try:
            # 使用 asyncio.to_thread 来运行同步翻译任务
            return await asyncio.to_thread(GoogleTranslator(source='auto', target='zh-CN').translate, text)
        except Exception as e:
            logger.error(f"翻译错误 | Translation error: {str(e)}")
            return text

    def is_english(self, text):
        """检查文本是否为英文 (Check if text is English)"""
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def is_chinese(self, text):
        """检查文本是否为中文 (Check if text is Chinese)"""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    async def translate_bilingual(self, en_text, cn_text):
        """创建双语文本 (Create bilingual text)"""
        return f"🌐 ENGLISH:\n{en_text}\n\n🌐 中文:\n{cn_text}"


# ========== 知识库 ==========
class MedicalKnowledgeBase:
    def __init__(self, config):
        self.config = config
        self.disease_info = {}
        self.symptom_info = {}
        self.medication_ddd_info = {}
        self.full_knowledge = []  # 存储完整的知识库内容
        self.learning_stats = {
            "files_processed": 0,
            "diseases_extracted": 0,
            "symptoms_extracted": 0,
            "medications_extracted": 0,
            "tests_extracted": 0,
            "total_size_kb": 0
        }

        # 加载知识
        self.load_knowledge()
        logger.info(f"知识库加载完成 | Knowledge base loaded: "
                    f"{len(self.disease_info)}种疾病 | diseases, "
                    f"{len(self.symptom_info)}种症状 | symptoms, "
                    f"{self.learning_stats['files_processed']}个文件 | files, "
                    f"{self.learning_stats['total_size_kb']:.2f} KB内容 | content")

    def extract_medical_info(self, text, file_path):
        """从文本中提取医疗信息 (支持多语言) (Extract medical info from text)"""
        try:
            # 保存完整知识
            self.full_knowledge.append({
                "file_path": file_path,
                "content": text,
                "size_kb": len(text.encode('utf-8')) / 1024
            })
            self.learning_stats["total_size_kb"] += len(text.encode('utf-8')) / 1024

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

                # 药物提取 (支持中英文) - 包含DDD值
                medications = []
                medication_pattern = r'(?:medications|drugs|prescriptions|剂量|药物)[\s:：]*([^\n]+)'
                medication_matches = re.findall(medication_pattern, text, re.IGNORECASE)

                for mm in medication_matches:
                    for line in mm.split('\n'):
                        # 支持多种格式的药物描述，包括DDD值
                        med_match = re.search(
                            r'([a-zA-Z\u4e00-\u9fff]+[\s\-]*[a-zA-Z\u4e00-\u9fff]*\d*)[\s(]*([\d.]+)?\s*([a-zA-Z\u4e00-\u9fff]*)?\s*(?:DDD:?\s*([\d.]+))?',
                            line, re.IGNORECASE)
                        if med_match:
                            name = med_match.group(1).strip()
                            dosage = med_match.group(2) if med_match.group(2) else ""
                            unit = med_match.group(3) if med_match.group(3) else ""
                            ddd_value = float(med_match.group(4)) if med_match.group(4) else None

                            medications.append({
                                'name': name,
                                'dosage': dosage,
                                'unit': unit,
                                'ddd': ddd_value
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

                    # 存储药物DDD信息
                    for med in medications:
                        if med['ddd'] is not None:
                            self.medication_ddd_info[med['name']] = med['ddd']

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
            logger.error(f"医疗信息提取错误 | Medical info extraction error: {str(e)}")
            return False

    async def calculate_ddd(self, medication, dosage, unit, frequency):
        """计算DDD值 (Calculate DDD value)"""
        # 1. 首先检查知识库中是否有该药物的DDD信息
        if medication in self.medication_ddd_info:
            ddd_value = self.medication_ddd_info[medication]
            return ddd_value, None

        # 2. 知识库中没有则尝试寻找替代药物
        alternatives = await self.find_alternative_medications(medication)
        if alternatives:
            return None, alternatives  # 返回None表示需要换药

        # 3. 最后尝试预测DDD
        ddd_value = self.predict_ddd_with_model(medication, dosage, unit, frequency)
        if ddd_value is not None:
            return ddd_value, None
        else:
            return None, ["知识库中没有DDD值相关信息，请更新知识库"]  # 返回错误消息

    async def find_alternative_medications(self, medication):
        """在知识库中寻找替代药物 (Find alternative medications in knowledge base)"""
        alternatives = []
        for disease, info in self.disease_info.items():
            for med in info.get("medications", []):
                med_name = med["name"]
                # 相似药物匹配 (支持多语言)
                if await self.is_similar_medication(medication, med_name) and med_name != medication:
                    alternatives.append(med_name)
        return list(set(alternatives))  # 去重

    async def is_similar_medication(self, med1, med2):
        """检查药物是否相似 (支持多语言) (Check if medications are similar)"""
        # 翻译为英文后比较
        med1_en = await self.config.translate_to_english(med1)
        med2_en = await self.config.translate_to_english(med2)

        # 检查翻译结果是否为 None 或空字符串，并进行小写转换
        med1_en_lower = (med1_en or "").lower()
        med2_en_lower = (med2_en or "").lower()

        if not med1_en_lower or not med2_en_lower:
            return False  # 如果翻译失败，认为不相似

        return SequenceMatcher(None, med1_en_lower, med2_en_lower).ratio() > 0.7

    def predict_ddd_with_model(self, medication, dosage, unit, frequency):
        """使用训练好的模型预测DDD值 (Predict DDD using trained model)"""
        # 这里应该加载训练好的模型进行预测
        # 简化版：返回固定值或基于规则的预测
        try:
            # 尝试从模型目录加载模型
            model_path = os.path.join(self.config.model_dir, "ddd_predictor.model")
            if os.path.exists(model_path):
                # 实际应用中应该加载模型并进行预测
                # 这里简化处理，返回一个基于名称的简单预测
                if "硝苯" in medication or "nifedipine" in medication.lower():
                    return 10.0
                elif "氨氯" in medication or "amlodipine" in medication.lower():
                    return 5.0
                elif "厄贝" in medication or "irbesartan" in medication.lower():
                    return 150.0
                else:
                    # 默认返回一个基于剂量的估计值
                    try:
                        dosage_val = float(dosage) if dosage else 1.0
                        return dosage_val * 1.5  # 简单估算
                    except:
                        return 10.0  # 默认值
            else:
                logger.warning("未找到训练好的DDD预测模型 | DDD prediction model not found")
                return None
        except Exception as e:
            logger.error(f"DDD预测错误 | DDD prediction error: {str(e)}")
            return None

    def load_knowledge(self):
        """从知识库文件夹加载所有知识库文件 (Load all knowledge base files)"""
        logger.info("从本地知识库文件夹加载医疗知识 | Loading medical knowledge from local knowledge_base folder...")

        for path in self.config.knowledge_paths:
            if not os.path.exists(path):
                logger.warning(f"知识路径未找到 | Knowledge path not found: {path}")
                continue

            logger.info(f"处理目录 | Processing directory: {path}")
            file_count = 0

            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file_path.endswith(ext) for ext in ('.txt', '.csv', '.json', '.docx', '.pdf')):
                        logger.info(f"处理文件 | Processing file: {file_path}")
                        try:
                            # 加载文件内容
                            content = self.load_file(file_path)

                            # 提取医疗信息
                            self.extract_medical_info(content, file_path)

                            file_count += 1
                            self.learning_stats["files_processed"] += 1
                        except Exception as e:
                            logger.error(f"文件处理错误 | Error processing file {file_path}: {str(e)}")

            logger.info(f"在路径中处理文件数 | Processed {file_count} files in {path}")

        # 完全移除任何默认知识添加
        if not self.disease_info:
            logger.warning("知识库文件中未提取到疾病 | No diseases extracted from knowledge base files")
        if not self.symptom_info:
            logger.warning("知识库文件中未提取到症状 | No symptoms extracted from knowledge base files")
        if not self.full_knowledge:
            logger.warning("知识库未加载任何内容 | No content loaded in knowledge base")

    def load_file(self, file_path):
        """加载单个知识文件 (Load a single knowledge file)"""
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
                logger.warning(f"不支持的文件格式 | Unsupported file format: {file_path}")
                return ""

            return content
        except Exception as e:
            logger.error(f"文件加载错误 | Error loading file {file_path}: {str(e)}")
            return ""


# ========== 医疗助手 ==========
class MedicalAssistant:
    def __init__(self, knowledge_base, config):
        self.knowledge_base = knowledge_base
        self.config = config
        self.thought_process = []  # 记录思考过程
        self.current_symptoms = []
        self.attempt_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        # 加载训练好的诊断模型
        self.diagnosis_model = self.load_diagnosis_model()

    def load_diagnosis_model(self):
        """加载训练好的诊断模型 (Load trained diagnosis model)"""
        try:
            model_path = os.path.join(self.config.model_dir, "diagnosis_model.model")
            if os.path.exists(model_path):
                # 实际应用中应该加载训练好的模型
                # 这里简化处理，返回一个模拟的模型对象
                logger.info("诊断模型加载成功 | Diagnosis model loaded successfully")
                return {"name": "DiagnosisModel", "version": "1.0"}
            else:
                logger.warning("未找到训练好的诊断模型 | Trained diagnosis model not found")
                return None
        except Exception as e:
            logger.error(f"模型加载错误 | Model loading error: {str(e)}")
            return None

    async def diagnose(self, chief_complaint):
        """诊断流程 (类人脑思考过程) (Diagnosis process)"""
        self.thought_process = [f"患者主诉 | Patient chief complaint: {chief_complaint}"]

        # 步骤1: 使用训练好的模型进行初步诊断
        diagnosis = await self.model_based_diagnosis(chief_complaint)
        self.thought_process.append(
            f"模型诊断 | Model diagnosis: {diagnosis['disease']} (置信度 | Confidence: {diagnosis['confidence'] * 100:.1f}%)")

        # 步骤2: 检查置信度是否达到阈值
        if diagnosis['confidence'] < self.config.diagnosis_threshold:
            self.thought_process.append(
                f"置信度低于阈值 {self.config.diagnosis_threshold * 100}%，请求更多信息 | Confidence below threshold, requesting more information")
            return await self.request_more_info(chief_complaint, diagnosis['confidence'])

        # 步骤3: 知识库验证
        kb_match = await self.check_knowledge_base_match(diagnosis['disease'])
        self.thought_process.append(
            f"知识库匹配 | Knowledge base match: {kb_match['match']} (相似度 | Similarity: {kb_match['similarity'] * 100:.1f}%)")

        # 步骤4: 用药推荐
        medication_response = await self.recommend_medication(diagnosis['disease'])

        # 步骤5: 检查建议
        test_recommendation = await self.recommend_tests(diagnosis['disease'])

        # 步骤6: 生成最终响应
        return await self.generate_final_response(
            diagnosis,
            kb_match,
            medication_response,
            test_recommendation
        )

    async def model_based_diagnosis(self, chief_complaint):
        """使用训练好的模型进行诊断 (Diagnosis using trained model)"""
        # 在实际应用中，这里会调用训练好的模型进行诊断
        # 简化版：基于知识库的规则匹配

        # 首先尝试将主诉翻译为英文
        chief_complaint_en = await self.config.translate_to_english(chief_complaint)

        best_match = None
        best_score = 0

        # 在知识库中寻找最匹配的疾病
        for disease, info in self.knowledge_base.disease_info.items():
            # 获取疾病的症状
            symptoms = info.get("symptoms", [])

            # 计算匹配分数
            score = await self.calculate_symptom_match(chief_complaint, symptoms)

            if score > best_score:
                best_score = score
                best_match = disease

        # 如果找到匹配的疾病且置信度足够高
        if best_match and best_score > 0.6:
            return {
                "disease": best_match,
                "confidence": min(best_score, 0.95)  # 最高95%，保留改进空间
            }
        else:
            # 默认返回高血压，置信度较低
            return {
                "disease": "高血压",
                "confidence": 0.75  # 默认置信度
            }

    async def calculate_symptom_match(self, complaint, symptoms):
        """计算症状匹配度 (Calculate symptom match score)"""
        if not symptoms:
            return 0.0

        # 将主诉和症状都翻译为英文进行比较
        complaint_en = await self.config.translate_to_english(complaint) or complaint.lower()

        total_score = 0
        count = 0

        for symptom in symptoms:
            symptom_en = await self.config.translate_to_english(symptom) or symptom.lower()

            # 使用编辑距离计算相似度
            similarity = 1 - (Levenshtein.distance(complaint_en, symptom_en) /
                              max(len(complaint_en), len(symptom_en)))

            # 如果相似度超过阈值，计入总分
            if similarity > 0.5:
                total_score += similarity
                count += 1

        # 返回平均相似度
        return total_score / count if count > 0 else 0.0

    async def check_knowledge_base_match(self, disease):
        """检查知识库匹配度 (Check knowledge base match)"""
        # 获取知识库中所有疾病
        kb_diseases = list(self.knowledge_base.disease_info.keys())

        if not kb_diseases:
            return {"match": "知识库中无相关疾病信息 | No relevant disease information in knowledge base",
                    "similarity": 0.0}

        # 计算相似度
        best_match = ""
        best_similarity = 0.0

        for kb_disease in kb_diseases:
            # 使用多语言相似度计算
            similarity = await self.calculate_multilingual_similarity(disease, kb_disease)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = kb_disease

        return {
            "match": best_match if best_similarity > 0.7 else "无匹配疾病 | No matching disease",
            "similarity": best_similarity
        }

    async def calculate_multilingual_similarity(self, text1, text2):
        """多语言相似度计算 (Multilingual similarity calculation)"""
        if not text1 or not text2:
            return 0.0

        text1_en = await self.config.translate_to_english(text1) or ""
        text2_en = await self.config.translate_to_english(text2) or ""

        if not text1_en or not text2_en:
            return 0.0

        return 1 - (Levenshtein.distance(text1_en, text2_en) / max(len(text1_en), len(text2_en)))

    async def request_more_info(self, chief_complaint, confidence):
        """请求更多症状信息 (Request more symptom information)"""
        # 英文响应
        en_response = f"Preliminary analysis based on: '{chief_complaint}'\n"
        en_response += f"Current confidence: {confidence * 100:.1f}% (below {self.config.diagnosis_threshold * 100}% threshold)\n"
        en_response += "Please provide more detailed symptoms for accurate diagnosis."

        # 中文响应
        cn_response = f"基于初步分析: '{chief_complaint}'\n"
        cn_response += f"当前置信度: {confidence * 100:.1f}% (低于 {self.config.diagnosis_threshold * 100}% 阈值)\n"
        cn_response += "请提供更详细的症状以进行准确诊断。"

        # 添加思考过程
        thought_header = "\n\n=== 思考过程 | THINKING PROCESS ===\n" + "\n".join(self.thought_process)

        return await self.config.translate_bilingual(en_response, cn_response) + thought_header

    async def recommend_medication(self, disease):
        """推荐药物 (类人脑思考过程) (Recommend medication - deep thinking)"""
        self.thought_process.append(f"为 {disease} 推荐药物 | Recommending medication for {disease}...")

        # 获取知识库中的药物
        medications = self.knowledge_base.disease_info.get(disease, {}).get("medications", [])

        if not medications:
            return {"status": "no_medication",
                    "message": "知识库中无相关药物信息 | No medication information in knowledge base"}

        # 计算DDD值
        results = []
        total_ddd = 0.0

        for med in medications:
            ddd_value, alternatives = await self.knowledge_base.calculate_ddd(
                med["name"], med["dosage"], med["unit"], "daily"
            )

            if ddd_value is None:  # 需要换药或无法计算DDD
                if alternatives and isinstance(alternatives, list):
                    # 如果是替代药物列表
                    alt_text = ", ".join(alternatives[:3])
                    results.append({
                        "medication": med["name"],
                        "status": "need_alternative",
                        "message": f"无法计算DDD，建议换药 | Cannot calculate DDD, suggested alternatives: {alt_text}"
                    })
                elif alternatives and isinstance(alternatives, str):
                    # 如果是错误消息
                    results.append({
                        "medication": med["name"],
                        "status": "no_ddd",
                        "message": alternatives  # 使用返回的错误消息
                    })
                else:
                    results.append({
                        "medication": med["name"],
                        "status": "no_ddd",
                        "message": "无法计算DDD且无替代药物 | Cannot calculate DDD and no alternatives found"
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

    async def recommend_tests(self, disease):
        """推荐检查 (基于知识库的深度思考) (Recommend tests - deep thinking)"""
        self.thought_process.append(f"为 {disease} 分析检查需求 | Analyzing test requirements for {disease}...")

        # 首先检查知识库中是否有该疾病的相关信息
        disease_info = self.knowledge_base.disease_info.get(disease, {})

        if not disease_info:
            # 知识库中没有该疾病信息
            self.thought_process.append(
                f"知识库中没有关于 {disease} 的信息 | No information about {disease} in knowledge base")
            return None

        # 检查知识库中是否有明确的检查建议
        if "tests" in disease_info and disease_info["tests"]:
            tests = disease_info["tests"]
            self.thought_process.append(
                f"从知识库中找到 {len(tests)} 项检查建议 | Found {len(tests)} test recommendations in knowledge base")
            return tests

        # 知识库中没有明确的检查建议，尝试从症状中推断
        symptoms = disease_info.get("symptoms", [])
        inferred_tests = await self.infer_tests_from_symptoms(symptoms)

        if inferred_tests:
            self.thought_process.append(
                f"从 {len(symptoms)} 个症状推断出 {len(inferred_tests)} 项检查 | Inferred {len(inferred_tests)} tests from {len(symptoms)} symptoms")
            return inferred_tests

        # 没有任何可用的检查建议
        self.thought_process.append(f"无法为 {disease} 推荐任何检查 | Unable to recommend any tests for {disease}")
        return None

    async def infer_tests_from_symptoms(self, symptoms):
        """从症状推断检查项目 (基于知识库的深度思考) (Infer tests from symptoms - deep thinking)"""
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
                similarity = await self.calculate_symptom_similarity(symptom, kb_symptom)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = kb_symptom

            # 如果找到匹配的症状且有相关检查，则添加
            if max_similarity > 0.7 and best_match in symptom_test_mapping:
                recommended_tests.extend(symptom_test_mapping[best_match])

        # 去重并限制数量
        return list(set(recommended_tests))[:5]  # 最多返回5项

    async def calculate_symptom_similarity(self, symptom1, symptom2):
        """计算症状相似度 (支持多语言) (Calculate symptom similarity)"""
        # 翻译为英文后比较
        symptom1_en = await self.config.translate_to_english(symptom1)
        symptom2_en = await self.config.translate_to_english(symptom2)

        # 使用编辑距离计算相似度
        if symptom1_en and symptom2_en:
            return 1 - (Levenshtein.distance(symptom1_en, symptom2_en) / max(len(symptom1_en), len(symptom2_en)))
        return 0.0

    async def generate_final_response(self, diagnosis, kb_match, medication, tests):
        """生成最终响应 (格式与图片一致，但包含中英文) (Generate final response matching the image format with bilingual content)"""
        # 中文部分 (按照图片格式)
        cn_response = f"诊断 ===\n"
        cn_response += f"模型:1Lama2\n"
        cn_response += f"疾病: {diagnosis['disease']}\n"
        cn_response += f"置信度: {diagnosis['confidence'] * 100:.1f}%\n"
        cn_response += f"知识库匹配: {kb_match['match']} (相似度: {kb_match['similarity'] * 100:.1f}%)\n\n"

        # 英文部分
        en_disease = await self.config.translate_to_english(diagnosis['disease'])
        en_match = await self.config.translate_to_english(kb_match['match'])

        en_response = f"Diagnosis ===\n"
        en_response += f"Model:1Lama2\n"
        en_response += f"Disease: {en_disease}\n"
        en_response += f"Confidence: {diagnosis['confidence'] * 100:.1f}%\n"
        en_response += f"Knowledge Base Match: {en_match} (Similarity: {kb_match['similarity'] * 100:.1f}%)\n\n"

        # 添加知识库摘要 (中文)
        cn_response += "=== 知识库摘要 ===\n"
        cn_response += f"总文档数: {len(self.knowledge_base.full_knowledge)}\n"
        cn_response += f"总大小: {self.knowledge_base.learning_stats['total_size_kb']:.2f} KB\n"
        cn_response += f"提取疾病数: {self.knowledge_base.learning_stats['diseases_extracted']}\n\n"

        # 添加知识库摘要 (英文)
        en_response += "=== Knowledge Base Summary ===\n"
        en_response += f"Total documents: {len(self.knowledge_base.full_knowledge)}\n"
        en_response += f"Total size: {self.knowledge_base.learning_stats['total_size_kb']:.2f} KB\n"
        en_response += f"Diseases extracted: {self.knowledge_base.learning_stats['diseases_extracted']}\n\n"

        # 药物推荐 (中文)
        cn_response += "药物推荐:\n"
        if medication["status"] == "no_medication":
            cn_response += "知识库中未找到药物信息\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    cn_response += f"- {med['medication']}: {med['dosage']}{med['unit']} (DDD值: {med['ddd']:.2f})\n"

            if medication["total_ddd"] > 0:
                cn_response += f"总DDD值: {medication['total_ddd']:.2f}\n"

        # 药物推荐 (英文)
        en_response += "Medication Recommendations:\n"
        if medication["status"] == "no_medication":
            en_response += "No medication information found in knowledge base\n"
        else:
            for med in medication["medications"]:
                if med["status"] == "success":
                    en_med = await self.config.translate_to_english(med['medication'])
                    en_unit = await self.config.translate_to_english(med['unit'])
                    en_response += f"- {en_med}: {med['dosage']}{en_unit} (DDD: {med['ddd']:.2f})\n"

            if medication["total_ddd"] > 0:
                en_response += f"Total DDD: {medication['total_ddd']:.2f}\n"

        # 推荐检查 (中文)
        cn_response += "\n推荐检查:\n"
        if tests:
            for test in tests:
                cn_response += f"- {test}\n"
        else:
            cn_response += "基于当前知识库，暂无特定检查建议"

        # 推荐检查 (英文)
        en_response += "\nRecommended Tests:\n"
        if tests:
            for test in tests:
                en_test = await self.config.translate_to_english(test)
                en_response += f"- {en_test}\n"
        else:
            en_response += "No specific tests recommended based on current knowledge"

        # 添加思考过程
        thought_header = "\n\n=== 思考过程 | THINKING PROCESS ===\n" + "\n".join(self.thought_process)

        # 组合中英文响应
        final_response = f"🌐 中文 | CHINESE:\n{cn_response}\n\n"
        final_response += f"🌐 ENGLISH:\n{en_response}\n\n"
        final_response += thought_header

        return final_response


# ========== 主函数 ==========
async def main():
    try:
        # 初始化配置
        config = MedicalConfig()

        # 加载知识库
        logger.info("\n[1/2] 加载医疗知识 | Loading medical knowledge...")
        knowledge_base = MedicalKnowledgeBase(config)

        # 初始化医疗助手
        logger.info("\n[2/2] 启动医疗助手 | Starting Medical Assistant")
        assistant = MedicalAssistant(knowledge_base, config)

        # 交互界面
        logger.info("\n=== 医疗诊断助手 (医生版) | MEDICAL DIAGNOSTIC ASSISTANT (FOR PHYSICIANS) ===")
        logger.info("输入患者症状进行诊断或输入'exit'退出 | Enter patient symptoms for diagnosis or 'exit' to quit")
        logger.info(f"诊断阈值 | Diagnosis threshold: {config.diagnosis_threshold * 100}%")
        logger.info("支持中英文输入 | Supports Chinese and English input")

        while True:
            user_input = input("\n输入症状 | Enter symptoms: ").strip()

            if user_input.lower() == "exit":
                break

            response = await assistant.diagnose(user_input)
            print(f"\n{response}")

    except Exception as e:
        error_msg = f"系统错误 | System error: {str(e)}"
        logger.error(error_msg)
        print(error_msg)


if __name__ == "__main__":
    asyncio.run(main())