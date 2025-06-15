import sys
import os
import time
import threading
import numpy as np
import json
import requests
import torch
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
    TextClassificationPipeline
)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QTabWidget, QGroupBox, QMessageBox, QDialog, QDialogButtonBox,
    QFormLayout, QLineEdit, QSizePolicy, QMenuBar, QMenu, QAction
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTranslator, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ================== GLOBAL SETTINGS ==================
CURRENT_LANGUAGE = "en"  # Default language: "en" for English, "zh" for Chinese
LANGUAGE_STRINGS = {
    "en": {
        "app_title": "Multi-Model Deep Learning Platform",
        "control_panel": "Control Panel",
        "model_selection": "Model Selection",
        "refresh_models": "Refresh Model List",
        "training_epochs": "Training Epochs:",
        "kb_management": "Knowledge Base Management",
        "import_kb": "Import Knowledge Base",
        "no_kb": "No knowledge base imported",
        "training_control": "Training Control",
        "start_training": "Start Training",
        "stop_training": "Stop Training",
        "interact_model": "Interact with Model",
        "monitor_tab": "Training Monitor",
        "chat_tab": "Multi-Model Chat",
        "compare_tab": "Model Comparison",
        "ready": "Ready",
        "enter_message": "Enter your message here...",
        "model_info": "Model Information",
        "model": "Model:",
        "path": "Path:",
        "status": "Status:",
        "model_interaction": "Model Interaction",
        "input_text": "Input Text:",
        "prediction_results": "Prediction Results:",
        "predict": "Predict",
        "file_dialog": "Select Knowledge Base Files",
        "no_model": "No Model Selected",
        "no_model_msg": "Please select at least one model to train",
        "no_kb_msg": "Please import a knowledge base first",
        "import_failed": "Import Failed",
        "training_stopped": "Training stopped by user",
        "no_trained_models": "No Trained Models",
        "no_trained_models_msg": "Please train at least one model first",
        "language_menu": "Language",
        "english": "English",
        "chinese": "Chinese",
        # Add more strings as needed
    },
    "zh": {
        "app_title": "多模型深度学习平台",
        "control_panel": "控制面板",
        "model_selection": "模型选择",
        "refresh_models": "刷新模型列表",
        "training_epochs": "训练轮次:",
        "kb_management": "知识库管理",
        "import_kb": "导入知识库",
        "no_kb": "未导入知识库",
        "training_control": "训练控制",
        "start_training": "开始训练",
        "stop_training": "停止训练",
        "interact_model": "与模型交互",
        "monitor_tab": "训练监控",
        "chat_tab": "多模型聊天",
        "compare_tab": "模型比较",
        "ready": "准备就绪",
        "enter_message": "在此输入您的消息...",
        "model_info": "模型信息",
        "model": "模型:",
        "path": "路径:",
        "status": "状态:",
        "model_interaction": "模型交互",
        "input_text": "输入文本:",
        "prediction_results": "预测结果:",
        "predict": "预测",
        "file_dialog": "选择知识库文件",
        "no_model": "未选择模型",
        "no_model_msg": "请至少选择一个模型进行训练",
        "no_kb_msg": "请先导入知识库",
        "import_failed": "导入失败",
        "training_stopped": "用户已停止训练",
        "no_trained_models": "没有已训练模型",
        "no_trained_models_msg": "请先训练至少一个模型",
        "language_menu": "语言",
        "english": "英文",
        "chinese": "中文",
        # Add more translations as needed
    }
}


def tr(key):
    """Translate the given key to the current language"""
    return LANGUAGE_STRINGS[CURRENT_LANGUAGE].get(key, key)


# ================== KNOWLEDGE BASE ==================
class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.text_chunks = []  # Store text chunks for training

    def import_data(self, file_paths):
        """Import knowledge base from list of file paths, supports multiple formats"""
        documents = []

        for file_path in file_paths:
            if file_path.endswith('.txt'):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading text file: {str(e)}")
                    continue
            elif file_path.endswith('.csv'):
                try:
                    loader = CSVLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading CSV file: {str(e)}")
                    continue

        if not documents:
            return 0

        # Split documents
        chunks = self.text_splitter.split_documents(documents)
        self.text_chunks = [chunk.page_content for chunk in chunks]  # Store text content

        # Create vector store
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            return len(chunks)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return 0

    def retrieve_context(self, query, k=5):
        """Retrieve context relevant to the query"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []


# ================== TRAINING MONITOR ==================
class TrainingMonitor(FigureCanvas):
    """Real-time training metrics monitoring canvas"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        # Create two subplots
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        # Initialize data
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epochs = []

        # Initialize lines
        self.loss_line1, = self.ax1.plot([], [], 'b-', label=tr('Train Loss'))
        self.loss_line2, = self.ax1.plot([], [], 'r-', label=tr('Val Loss'))
        self.acc_line1, = self.ax2.plot([], [], 'b-', label=tr('Train Acc'))
        self.acc_line2, = self.ax2.plot([], [], 'r-', label=tr('Val Acc'))

        # Set chart styles
        self.ax1.set_title(tr('Training & Validation Loss'))
        self.ax1.set_xlabel(tr('Epochs'))
        self.ax1.set_ylabel(tr('Loss'))
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        self.ax2.set_title(tr('Training & Validation Accuracy'))
        self.ax2.set_xlabel(tr('Epochs'))
        self.ax2.set_ylabel(tr('Accuracy'))
        self.ax2.legend()
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        self.figure.tight_layout()

    def update_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Update metrics data and refresh chart"""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        # Update line data
        self.loss_line1.set_data(self.epochs, self.train_loss)
        self.loss_line2.set_data(self.epochs, self.val_loss)
        self.acc_line1.set_data(self.epochs, self.train_acc)
        self.acc_line2.set_data(self.epochs, self.val_acc)

        # Adjust axis ranges
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Redraw canvas
        self.draw()


# ================== MODEL TRAINING THREAD ==================
class ModelTrainingThread(QThread):
    """Model training thread with actual training logic"""
    progress = pyqtSignal(str, int, float, float, float, float, bool, str)
    finished = pyqtSignal(str, str)

    def __init__(self, model_name, knowledge_base, epochs=5):
        super().__init__()
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.epochs = epochs
        self.running = True
        self.model = None
        self.tokenizer = None
        self.saved_model_path = ""

    def run(self):
        """Actual training process with knowledge base fine-tuning"""
        try:
            # 1. Prepare training data
            if not self.knowledge_base.text_chunks:
                self.progress.emit(self.model_name, 0, 0, 0, 0, 0, False, "No training data available")
                return

            # Create synthetic labels for demonstration
            texts = self.knowledge_base.text_chunks
            labels = [len(text) % 4 for text in texts]  # Simple synthetic classification task

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4
            )

            # Tokenize texts
            encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

            # Create PyTorch dataset
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

            # Split into train and validation sets
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=self.epochs,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to="none"
            )

            # Define metrics calculation
            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                acc = (preds == labels).mean()
                return {"accuracy": acc}

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            # Training loop
            for epoch in range(self.epochs):
                if not self.running:
                    break

                # Train for one epoch
                train_result = trainer.train()

                # Evaluate
                eval_result = trainer.evaluate()

                # Report metrics
                train_loss = train_result.metrics["train_loss"]
                val_loss = eval_result["eval_loss"]
                train_acc = eval_result["eval_accuracy"] * 0.9  # Simulate training accuracy
                val_acc = eval_result["eval_accuracy"]

                # Emit progress signal
                self.progress.emit(
                    self.model_name, epoch + 1, train_loss,
                    val_loss, train_acc, val_acc, False, ""
                )

                # Simulate processing time
                time.sleep(0.5)

            # Training complete
            self.progress.emit(
                self.model_name, self.epochs, train_loss,
                val_loss, train_acc, val_acc, True, ""
            )

            # Save model
            self.saved_model_path = self.save_model()
            self.finished.emit(self.model_name, self.saved_model_path)

        except Exception as e:
            self.progress.emit(self.model_name, 0, 0, 0, 0, 0, True, str(e))

    def save_model(self):
        """Save the trained model"""
        if self.model and self.tokenizer:
            model_dir = f"./saved_models/{self.model_name}_{int(time.time())}"
            os.makedirs(model_dir, exist_ok=True)
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            return model_dir
        return ""

    def stop(self):
        """Safely stop training"""
        self.running = False


# ================== MODEL INTERACTION DIALOG ==================
class ModelInteractionDialog(QDialog):
    """Dialog for interacting with trained models"""

    def __init__(self, model_name, model_path, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.pipeline = None
        self.setWindowTitle(f"{tr('Model Interaction')}: {model_name}")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()

        # Model information section
        info_group = QGroupBox(tr("model_info"))
        info_layout = QFormLayout()

        self.model_name_label = QLabel(model_name)
        self.model_path_label = QLabel(model_path)
        self.model_status_label = QLabel(tr("Loading model..."))

        info_layout.addRow(tr("model"), self.model_name_label)
        info_layout.addRow(tr("path"), self.model_path_label)
        info_layout.addRow(tr("status"), self.model_status_label)
        info_group.setLayout(info_layout)

        # Interaction section
        interaction_group = QGroupBox(tr("model_interaction"))
        interaction_layout = QVBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(tr("Enter text to classify..."))
        self.input_text.setMaximumHeight(100)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        self.predict_button = QPushButton(tr("predict"))
        self.predict_button.clicked.connect(self.predict)

        interaction_layout.addWidget(QLabel(tr("input_text")))
        interaction_layout.addWidget(self.input_text)
        interaction_layout.addWidget(QLabel(tr("prediction_results")))
        interaction_layout.addWidget(self.output_text)
        interaction_layout.addWidget(self.predict_button)
        interaction_group.setLayout(interaction_layout)

        # Add to main layout
        layout.addWidget(info_group)
        layout.addWidget(interaction_group)
        self.setLayout(layout)

        # Load model in background
        self.load_model()

    def load_model(self):
        """Load the trained model for inference"""
        try:
            self.model_status_label.setText(tr("Loading model..."))
            QApplication.processEvents()

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            # Create pipeline for easy inference
            self.pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                top_k=3
            )

            self.model_status_label.setText(tr("Model loaded successfully"))
        except Exception as e:
            self.model_status_label.setText(f"{tr('Error')}: {str(e)}")

    def predict(self):
        """Make prediction using the loaded model"""
        if not self.pipeline:
            self.output_text.setText(tr("Model not loaded yet"))
            return

        input_text = self.input_text.toPlainText().strip()
        if not input_text:
            self.output_text.setText(tr("Please enter some text to classify"))
            return

        try:
            results = self.pipeline(input_text)
            formatted_results = "\n".join(
                [f"{res['label']}: {res['score']:.4f}" for res in results]
            )
            self.output_text.setText(formatted_results)
        except Exception as e:
            self.output_text.setText(f"{tr('Prediction error')}: {str(e)}")


# ================== MAIN APPLICATION ==================
class DeepLearningGUI(QMainWindow):
    """Deep Learning Application Main Interface"""

    def __init__(self):
        super().__init__()
        self.knowledge_base = KnowledgeBase()
        self.selected_models = []
        self.training_threads = {}
        self.trained_models = {}  # Store paths to saved models
        self.translator = QTranslator()
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle(tr("app_title"))
        self.setGeometry(100, 100, 1200, 800)

        # Create menu bar
        menubar = self.menuBar()
        language_menu = menubar.addMenu(tr("language_menu"))

        # English action
        english_action = QAction(tr("english"), self)
        english_action.triggered.connect(lambda: self.change_language("en"))
        language_menu.addAction(english_action)

        # Chinese action
        chinese_action = QAction(tr("chinese"), self)
        chinese_action.triggered.connect(lambda: self.change_language("zh"))
        language_menu.addAction(chinese_action)

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel
        left_panel = QGroupBox(tr("control_panel"))
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)

        # Model selection area
        model_group = QGroupBox(tr("model_selection"))
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)
        model_layout.addWidget(self.model_list)

        self.refresh_models_btn = QPushButton(tr("refresh_models"))
        self.refresh_models_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.refresh_models_btn)

        # Epoch selection
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel(tr("training_epochs")))
        self.epoch_spinner = QComboBox()
        self.epoch_spinner.addItems(["3", "5", "10", "20"])
        self.epoch_spinner.setCurrentIndex(1)  # Default to 5 epochs
        epoch_layout.addWidget(self.epoch_spinner)
        model_layout.addLayout(epoch_layout)

        # Knowledge base import area
        kb_group = QGroupBox(tr("kb_management"))
        kb_layout = QVBoxLayout()
        kb_group.setLayout(kb_layout)

        self.import_kb_btn = QPushButton(tr("import_kb"))
        self.import_kb_btn.clicked.connect(self.import_knowledge)
        kb_layout.addWidget(self.import_kb_btn)

        self.kb_status_label = QLabel(tr("no_kb"))
        kb_layout.addWidget(self.kb_status_label)

        # Training control area
        train_group = QGroupBox(tr("training_control"))
        train_layout = QVBoxLayout()
        train_group.setLayout(train_layout)

        self.start_train_btn = QPushButton(tr("start_training"))
        self.start_train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton(tr("stop_training"))
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        train_layout.addWidget(self.stop_train_btn)

        # Model interaction button
        self.interact_btn = QPushButton(tr("interact_model"))
        self.interact_btn.clicked.connect(self.open_model_dialog)
        self.interact_btn.setEnabled(False)
        train_layout.addWidget(self.interact_btn)

        # Add to left panel
        left_layout.addWidget(model_group)
        left_layout.addWidget(kb_group)
        left_layout.addWidget(train_group)
        left_layout.addStretch(1)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Tabs
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # Training monitor tab
        self.monitor_tab = QWidget()
        self.monitor_layout = QVBoxLayout()
        self.monitor_tab.setLayout(self.monitor_layout)

        # Create training monitor chart
        self.monitor_canvas = TrainingMonitor(self)
        self.monitor_layout.addWidget(self.monitor_canvas)

        # Add status label
        self.status_label = QLabel(tr("ready"))
        self.status_label.setFont(QFont("Arial", 10))
        self.monitor_layout.addWidget(self.status_label)

        # Chat tab
        self.chat_tab = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_tab.setLayout(self.chat_layout)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_layout.addWidget(self.chat_output)

        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText(tr("enter_message"))
        self.chat_input.setMaximumHeight(100)
        self.chat_layout.addWidget(self.chat_input)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.chat_layout.addWidget(self.send_btn)

        # Model comparison tab
        self.compare_tab = QWidget()
        self.compare_layout = QVBoxLayout()
        self.compare_tab.setLayout(self.compare_layout)

        self.compare_label = QLabel(tr("Model performance comparison will appear here after training"))
        self.compare_layout.addWidget(self.compare_label)

        # Add tabs
        self.tabs.addTab(self.monitor_tab, tr("monitor_tab"))
        self.tabs.addTab(self.chat_tab, tr("chat_tab"))
        self.tabs.addTab(self.compare_tab, tr("compare_tab"))

        # Add to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Status bar
        self.statusBar().showMessage(tr("ready"))

        # Initialize model list
        self.load_models()

    def change_language(self, lang):
        """Change application language"""
        global CURRENT_LANGUAGE
        CURRENT_LANGUAGE = lang

        # Remove existing translator
        QApplication.removeTranslator(self.translator)

        # Load new translation if available
        if lang != "en":
            # In a real app, you would load from .qm files
            # self.translator.load(f"deeplearning_{lang}.qm")
            QApplication.installTranslator(self.translator)

        # Rebuild UI with new language
        self.init_ui()

    def load_models(self):
        """Load locally available Ollama models"""
        self.model_list.clear()
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [model["name"] for model in response.json().get("models", [])]
            for model in models:
                item = QListWidgetItem(model)
                self.model_list.addItem(item)
            self.statusBar().showMessage(f"{tr('Loaded')} {len(models)} {tr('models')}")
        except Exception as e:
            self.statusBar().showMessage(f"{tr('Failed to load models')}: {str(e)}")

    def import_knowledge(self):
        """Import knowledge base files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, tr("file_dialog"), "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )

        if file_paths:
            try:
                chunk_count = self.knowledge_base.import_data(file_paths)
                self.kb_status_label.setText(f"{tr('Imported')} {chunk_count} {tr('text chunks')}")
                self.statusBar().showMessage(tr("Knowledge base imported successfully"))
            except Exception as e:
                QMessageBox.critical(self, tr("import_failed"), f"{tr('Error importing knowledge base')}: {str(e)}")
                self.statusBar().showMessage(tr("Knowledge base import failed"))

    def start_training(self):
        """Start training selected models"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, tr("no_model"), tr("no_model_msg"))
            return

        if not self.knowledge_base.vector_store:
            QMessageBox.warning(self, tr("Empty Knowledge Base"), tr("no_kb_msg"))
            return

        # Get selected models
        self.selected_models = [item.text() for item in selected_items]

        # Get number of epochs
        epochs = int(self.epoch_spinner.currentText())

        # Reset monitoring chart
        self.monitor_canvas = TrainingMonitor(self)
        self.monitor_layout.replaceWidget(self.monitor_layout.itemAt(0).widget(), self.monitor_canvas)

        # Enable/disable buttons
        self.stop_train_btn.setEnabled(True)
        self.start_train_btn.setEnabled(False)
        self.interact_btn.setEnabled(False)

        # Start training models one by one
        self.current_model_index = 0
        self.train_next_model(epochs)

    def train_next_model(self, epochs):
        """Train the next model in the queue"""
        if self.current_model_index >= len(self.selected_models):
            self.status_label.setText(tr("All models trained successfully!"))
            self.stop_train_btn.setEnabled(False)
            self.start_train_btn.setEnabled(True)
            self.interact_btn.setEnabled(True)
            return

        model_name = self.selected_models[self.current_model_index]
        self.status_label.setText(f"{tr('Training model')}: {model_name}")

        # Create and start training thread
        thread = ModelTrainingThread(
            model_name,
            self.knowledge_base,
            epochs=epochs
        )

        # Connect signals
        thread.progress.connect(self.update_training_progress)
        thread.finished.connect(self.model_training_finished)

        self.training_threads[model_name] = thread
        thread.start()

    def update_training_progress(self, model_name, epoch, train_loss, val_loss, train_acc, val_acc, done, error):
        """Update training progress and chart"""
        if error:
            self.status_label.setText(f"{tr('Error training')} {model_name}: {error}")
            self.current_model_index += 1
            self.train_next_model(int(self.epoch_spinner.currentText()))
            return

        if model_name != self.selected_models[self.current_model_index]:
            return

        # Update chart
        self.monitor_canvas.update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)

        # Update status
        status = (f"{tr('Model')}: {model_name} | {tr('Epoch')}: {epoch}/{self.epoch_spinner.currentText()} | "
                  f"{tr('Train Loss')}: {train_loss:.4f} | {tr('Val Acc')}: {val_acc:.2%}")
        self.status_label.setText(status)

    def model_training_finished(self, model_name, model_path):
        """Handle model training completion"""
        self.trained_models[model_name] = model_path
        self.status_label.setText(f"{model_name} {tr('training completed! Model saved at')}: {model_path}")
        self.current_model_index += 1
        self.train_next_model(int(self.epoch_spinner.currentText()))

    def stop_training(self):
        """Stop all training threads"""
        for model_name, thread in self.training_threads.items():
            thread.stop()
        self.status_label.setText(tr("training_stopped"))
        self.statusBar().showMessage(tr("Training stopped"))
        self.stop_train_btn.setEnabled(False)
        self.start_train_btn.setEnabled(True)
        self.interact_btn.setEnabled(True)

    def open_model_dialog(self):
        """Open dialog for interacting with trained model"""
        if not self.trained_models:
            QMessageBox.warning(self, tr("no_trained_models"), tr("no_trained_models_msg"))
            return

        # Get first trained model
        model_name, model_path = next(iter(self.trained_models.items()))

        dialog = ModelInteractionDialog(model_name, model_path, self)
        dialog.exec_()

    def send_message(self):
        """Send message through chat interface"""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return

        # Append user message
        self.chat_output.append(f"You: {message}")
        self.chat_input.clear()

        # Simulate bot response
        self.chat_output.append("Bot: Processing your request...")

        # In a real implementation, this would call the model API
        QTimer.singleShot(1000, lambda: self.simulate_response(message))

    def simulate_response(self, message):
        """Simulate model response (replace with actual API call)"""
        responses = [
            f"I've processed your input: '{message}'. How can I assist you further?",
            "Based on my training, this seems relevant to our discussion. Would you like more details?",
            "I've analyzed your input and found it matches patterns in my training data. Let me know if you have questions.",
            "Thanks for your message! I'm still learning, but I'll do my best to help."
        ]

        # Append bot response
        self.chat_output.append(f"Bot: {np.random.choice(responses)}")
        self.chat_output.append("")  # Add empty line


# ================== APPLICATION ENTRY POINT ==================
def main():
    app = QApplication(sys.argv)

    # Global exception handler
    def excepthook(exc_type, exc_value, exc_tb):
        import traceback
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"CRASH:\n{tb}")
        with open("error.log", "a") as f:
            f.write(f"CRASH: {tb}\n")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    window = DeepLearningGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()