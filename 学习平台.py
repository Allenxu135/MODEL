import sys
import os
import time
import numpy as np
import json
import requests
import torch
import multiprocessing as mp
import psutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
    QFormLayout, QSpinBox, QCheckBox, QDoubleSpinBox, QSlider, QSplitter
)
from PyQt5.QtGui import QFont, QColor, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ================== GLOBAL SETTINGS ==================
APP_STRINGS = {
    "app_title": "Deep Learning Studio",
    "control_panel": "Control Panel",
    "model_selection": "Model Selection",
    "refresh_models": "Refresh Model List",
    "training_epochs": "Training Epochs:",
    "kb_management": "Knowledge Base",
    "import_kb": "Import Data",
    "no_kb": "No data imported",
    "training_control": "Training Control",
    "start_training": "Start Training",
    "stop_training": "Stop Training",
    "interact_model": "Interact with Model",
    "monitor_tab": "Training Monitor",
    "chat_tab": "AI Chat",
    "compare_tab": "Model Comparison",
    "ready": "Ready",
    "enter_message": "Enter your message here...",
    "model_info": "Model Information",
    "model": "Model:",
    "path": "Path:",
    "status": "Status:",
    "model_interaction": "Model Interaction",
    "input_text": "Input Text:",
    "prediction_results": "Output:",
    "predict": "Generate",
    "file_dialog": "Select Data Files",
    "no_model": "No Model Selected",
    "no_model_msg": "Please select at least one model to train",
    "no_kb_msg": "Please import data first",
    "import_failed": "Import Failed",
    "training_stopped": "Training stopped",
    "no_trained_models": "No Trained Models",
    "no_trained_models_msg": "Please train at least one model first",
    "cpu_optimization": "Performance Settings",
    "cpu_cores": "CPU Cores:",
    "cpu_binding": "Bind CPU Cores",
    "use_half_precision": "Use Half Precision",
    "cpu_threads": "CPU Threads:",
    "enable_async": "Enable Async I/O",
    "configure_cpu": "Configure Settings",
    "precision_settings": "Precision Settings",
    "thread_settings": "Thread Settings",
    "current_settings": "Current Settings:",
    "bound_cores": "Bound CPU cores:",
    "precision": "Precision:",
    "thread_count": "Thread count:",
    "async_status": "Async I/O:",
    "model_trained": "Model trained:",
    "training_completed": "Training completed! Model saved at:",
    "failed_load": "Failed to load models:",
    "loaded_models": "Loaded {} models",
    "error_importing": "Error importing data:",
    "kb_imported": "Data imported successfully: {} text chunks",
    "training_model": "Training model: {}",
    "training_status": "Training status: Epoch {}/{} | Loss: {:.4f} | Acc: {:.2%}",
    "model_loaded": "Model loaded",
    "prediction_error": "Error: {}",
    "gen_settings": "Generation Settings",
    "max_tokens": "Max Tokens:",
    "temperature": "Temperature:",
    "use_context": "Use Knowledge Context",
    "batch_size": "Batch Size:",
    "learning_rate": "Learning Rate:",
    "model_status": "Model Status:",
    "no_models": "No models available",
    "install_models": "Please install models in Ollama",
    "active_model": "Active model: {}",
    "available_models": "Available: {} models",
    "model_not_loaded": "Model '{}' not loaded",
    "select_model": "Select a model",
    "ollama_unavailable": "Ollama service unavailable",
    "ollama_running": "Ollama: Running",
    "ollama_stopped": "Ollama: Not Running",
    "scanning_models": "Scanning local models...",
    "models_found": "Found {} models",
    "model_scan_failed": "Failed to scan models",
    "select_multiple": "Hold Ctrl/Cmd to select multiple models"
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
            # Use Ollama CLI to list models
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse output to get model names
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:  # Ensure there's at least one part
                            model_name = parts[0]
                            self.available_models.append(model_name)

                # Set default model if any found
                if self.available_models:
                    self.model_name = self.available_models[0]

            return True
        except Exception as e:
            print(f"Error scanning local models: {str(e)}")
            return False

    def set_model(self, model_name):
        """Set current model if it exists"""
        if model_name in self.available_models:
            self.model_name = model_name
            return True
        return False

    def load_model(self):
        """Verify if current model is available"""
        if not self.model_name:
            return False
        return True  # Since we scanned locally, we know it exists

    def generate_text(self, prompt, context=None):
        """Generate text with automatic fallback to other models"""
        if not self.model_name:
            return "No model available"

        full_prompt = context + "\n\n" + prompt if context else prompt

        # First try with selected model
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=full_prompt,
                options=self.generation_config
            )
            return response['response']
        except Exception as e:
            error_message = str(e)

        # If fails, try other available models
        for model in self.available_models:
            if model == self.model_name:
                continue
            try:
                response = self.client.generate(
                    model=model,
                    prompt=full_prompt,
                    options=self.generation_config
                )
                self.model_name = model  # Switch to successful model
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
        self.max_parallel_training = max(1, mp.cpu_count() // 2)  # Add parallel training control

    def apply_cpu_binding(self):
        """Bind process to specific CPU cores"""
        if self.enable_binding and self.cpu_cores:
            try:
                p = psutil.Process()
                p.cpu_affinity(self.cpu_cores)
            except Exception:
                pass


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
        self.executor = ThreadPoolExecutor(max_workers=cpu_settings.cpu_threads)

    def _load_file(self, file_path):
        """Load single file with error handling"""
        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                return loader.load()
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
                return loader.load()
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return []
        return []

    def import_data(self, file_paths):
        """Optimized import using parallel processing"""
        if not file_paths:
            return 0

        futures = [self.executor.submit(self._load_file, path) for path in file_paths]
        documents = []
        for future in futures:
            result = future.result()
            if result:
                documents.extend(result)

        if not documents:
            return 0

        split_futures = [self.executor.submit(self.text_splitter.split_documents, [doc])
                         for doc in documents]
        chunks = []
        for future in split_futures:
            result = future.result()
            if result:
                chunks.extend(result)

        self.text_chunks = [chunk.page_content for chunk in chunks]

        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            return len(chunks)
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return 0

    def retrieve_context(self, query, k=5):
        """Retrieve context with async option"""
        if not self.vector_store:
            return []

        if self.cpu_settings.enable_async:
            future = self.executor.submit(self.vector_store.similarity_search, query, k)
            return future.result()
        else:
            return self.vector_store.similarity_search(query, k=k)


# ================== TRAINING MONITOR ==================
class TrainingMonitor(FigureCanvas):
    """Real-time training metrics monitoring"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.loss_line, = self.ax.plot([], [], 'b-', label='Train Loss')
        self.ax.set_title('Training Metrics')
        self.ax.set_xlabel('Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.figure.tight_layout()
        self.epochs = []
        self.train_loss = []

    def update_metrics(self, epoch, train_loss):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.loss_line.set_data(self.epochs, self.train_loss)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()


# ================== MODEL TRAINING THREAD ==================
class ModelTrainingThread(QThread):
    progress = pyqtSignal(str, int, float, bool, str)
    finished = pyqtSignal(str, str)

    def __init__(self, model_name, knowledge_base, cpu_settings, epochs=5):
        super().__init__()
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.epochs = epochs
        self.running = True
        self.cpu_settings = cpu_settings
        self.cpu_settings.apply_cpu_binding()

    def run(self):
        try:
            if not self.knowledge_base.text_chunks:
                self.progress.emit(self.model_name, 0, 0, False, "No training data")
                return

            texts = self.knowledge_base.text_chunks
            labels = np.array([len(text) % 4 for text in texts], dtype=np.int64)

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            torch_dtype = torch.float16 if self.cpu_settings.use_half_precision else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4,
                torch_dtype=torch_dtype
            )

            # Tokenization
            encodings = self.tokenizer(texts, truncation=True, padding=True,
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
                no_cuda=True
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            for epoch in range(self.epochs):
                if not self.running:
                    break

                train_result = trainer.train()
                train_loss = train_result.metrics.get("train_loss", 0)
                self.progress.emit(self.model_name, epoch + 1, train_loss, False, "")

            self.progress.emit(self.model_name, self.epochs, train_loss, True, "")
            self.saved_model_path = self.save_model()
            self.finished.emit(self.model_name, self.saved_model_path)

        except Exception as e:
            self.progress.emit(self.model_name, 0, 0, True, str(e))

    def save_model(self):
        if self.model and self.tokenizer:
            model_dir = f"./saved_models/{self.model_name}_{int(time.time())}"
            os.makedirs(model_dir, exist_ok=True)
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            return model_dir
        return ""

    def stop(self):
        self.running = False


# ================== MODEL COMPARISON DIALOG ==================
class ModelComparisonDialog(QDialog):
    def __init__(self, model_names, knowledge_base, parent=None):
        super().__init__(parent)
        self.model_names = model_names
        self.knowledge_base = knowledge_base
        self.generator = parent.text_generator
        self.setWindowTitle("Model Comparison")
        self.setGeometry(200, 200, 1200, 800)

        main_layout = QVBoxLayout()

        # Input area
        input_layout = QHBoxLayout()
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(APP_STRINGS["enter_message"])
        self.input_text.setMinimumHeight(100)
        input_layout.addWidget(self.input_text, 5)

        self.generate_button = QPushButton(APP_STRINGS["predict"])
        self.generate_button.clicked.connect(self.generate)
        input_layout.addWidget(self.generate_button, 1)

        main_layout.addLayout(input_layout)

        # Tab widget for model outputs
        self.tab_widget = QTabWidget()
        self.output_texts = {}  # model_name -> QTextEdit
        self.status_labels = {}  # model_name -> QLabel

        for model_name in model_names:
            tab = QWidget()
            layout = QVBoxLayout()

            # Model info
            info_group = QGroupBox(f"Model: {model_name}")
            info_layout = QFormLayout()
            status_label = QLabel("Ready")
            info_layout.addRow(APP_STRINGS["status"], status_label)
            info_group.setLayout(info_layout)
            layout.addWidget(info_group)

            # Output area
            output_text = QTextEdit()
            output_text.setReadOnly(True)
            layout.addWidget(output_text)

            tab.setLayout(layout)
            self.tab_widget.addTab(tab, model_name)

            # Store references
            self.output_texts[model_name] = output_text
            self.status_labels[model_name] = status_label

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def generate(self):
        prompt = self.input_text.toPlainText().strip()
        if not prompt:
            return

        # Retrieve context once for all models
        context = None
        if self.knowledge_base.vector_store:
            context_chunks = self.knowledge_base.retrieve_context(prompt, k=3)
            context = "\n".join([c.page_content for c in context_chunks])

        for model_name in self.model_names:
            # Update status
            self.status_labels[model_name].setText("Generating...")
            self.output_texts[model_name].setText("Generating...")
            QApplication.processEvents()

            try:
                # Switch model
                self.generator.set_model(model_name)

                # Generate response
                response = self.generator.generate_text(prompt, context)
                self.output_texts[model_name].setText(response)
                self.status_labels[model_name].setText("Completed")
            except Exception as e:
                self.output_texts[model_name].setText(f"Error: {str(e)}")
                self.status_labels[model_name].setText("Error")


# ================== MODEL INTERACTION DIALOG ==================
class ModelInteractionDialog(QDialog):
    def __init__(self, model_name, model_path, knowledge_base, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.knowledge_base = knowledge_base
        self.generator = parent.text_generator
        self.generator.set_model(model_name)
        self.setWindowTitle(f"Model: {model_name}")
        self.setGeometry(200, 200, 900, 700)

        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Model info and settings
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        info_group = QGroupBox(APP_STRINGS["model_info"])
        info_layout = QFormLayout()
        self.model_name_label = QLabel(model_name)
        self.model_path_label = QLabel(model_path)
        self.model_status_label = QLabel("Loading model...")
        info_layout.addRow(APP_STRINGS["model"], self.model_name_label)
        info_layout.addRow(APP_STRINGS["path"], self.model_path_label)
        info_layout.addRow(APP_STRINGS["status"], self.model_status_label)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)

        # Generation settings
        settings_group = QGroupBox(APP_STRINGS["gen_settings"])
        settings_layout = QFormLayout()

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(10, 4096)
        self.max_tokens_spin.setValue(512)
        settings_layout.addRow(APP_STRINGS["max_tokens"], self.max_tokens_spin)

        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(0, 200)
        self.temp_slider.setValue(70)
        self.temp_value_label = QLabel("0.7")
        settings_layout.addRow(APP_STRINGS["temperature"], self.temp_slider)
        settings_layout.addRow(QLabel("Value:"), self.temp_value_label)

        self.context_check = QCheckBox(APP_STRINGS["use_context"])
        self.context_check.setChecked(True)
        settings_layout.addRow(self.context_check)

        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        left_layout.addStretch(1)
        left_panel.setLayout(left_layout)

        # Right panel: Interaction
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText(APP_STRINGS["enter_message"])
        self.input_text.setMinimumHeight(100)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        self.generate_button = QPushButton(APP_STRINGS["predict"])
        self.generate_button.clicked.connect(self.generate)

        right_layout.addWidget(QLabel(APP_STRINGS["input_text"]))
        right_layout.addWidget(self.input_text)
        right_layout.addWidget(QLabel(APP_STRINGS["prediction_results"]))
        right_layout.addWidget(self.output_text)
        right_layout.addWidget(self.generate_button)
        right_panel.setLayout(right_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 600])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        self.temp_slider.valueChanged.connect(self.update_temp_label)
        self.load_model()

    def update_temp_label(self, value):
        self.temp_value_label.setText(f"{value / 100:.2f}")

    def load_model(self):
        try:
            if self.generator.load_model():
                self.model_status_label.setText(APP_STRINGS["model_loaded"])
            else:
                self.model_status_label.setText("Failed to load model")
        except Exception as e:
            self.model_status_label.setText(f"Error: {str(e)}")

    def generate(self):
        if not self.generator.client:
            self.output_text.setText("Model not loaded")
            return

        prompt = self.input_text.toPlainText().strip()
        if not prompt:
            self.output_text.setText("Please enter a prompt")
            return

        try:
            # Update generation parameters
            self.generator.generation_config = {
                "max_new_tokens": self.max_tokens_spin.value(),
                "temperature": self.temp_slider.value() / 100,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }

            # Retrieve context if needed
            context = None
            if self.context_check.isChecked() and self.knowledge_base.vector_store:
                context_chunks = self.knowledge_base.retrieve_context(prompt, k=3)
                context = "\n".join([c.page_content for c in context_chunks])

            self.output_text.setText("Generating...")
            QApplication.processEvents()

            response = self.generator.generate_text(prompt, context)
            self.output_text.setText(response)

        except Exception as e:
            self.output_text.setText(f"Error: {str(e)}")


# ================== TRAINING SETTINGS DIALOG ==================
class TrainingSettingsDialog(QDialog):
    def __init__(self, cpu_settings, parent=None):
        super().__init__(parent)
        self.cpu_settings = cpu_settings
        self.setWindowTitle(APP_STRINGS["cpu_optimization"])
        self.setGeometry(300, 300, 500, 400)

        layout = QVBoxLayout()
        tab_widget = QTabWidget()

        # Performance tab
        perf_tab = QWidget()
        perf_layout = QFormLayout()

        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, mp.cpu_count())
        self.thread_spin.setValue(cpu_settings.cpu_threads)
        perf_layout.addRow(APP_STRINGS["cpu_threads"], self.thread_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(cpu_settings.batch_size)
        perf_layout.addRow(APP_STRINGS["batch_size"], self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setValue(cpu_settings.learning_rate)
        self.lr_spin.setDecimals(6)
        perf_layout.addRow(APP_STRINGS["learning_rate"], self.lr_spin)

        self.half_precision_checkbox = QCheckBox(APP_STRINGS["use_half_precision"])
        self.half_precision_checkbox.setChecked(cpu_settings.use_half_precision)
        perf_layout.addRow(self.half_precision_checkbox)

        self.async_checkbox = QCheckBox(APP_STRINGS["enable_async"])
        self.async_checkbox.setChecked(cpu_settings.enable_async)
        perf_layout.addRow(self.async_checkbox)

        perf_tab.setLayout(perf_layout)
        tab_widget.addTab(perf_tab, "Performance")

        # CPU binding tab
        core_tab = QWidget()
        core_layout = QVBoxLayout()

        self.cpu_list = QListWidget()
        self.cpu_list.setSelectionMode(QListWidget.MultiSelection)
        for i in range(psutil.cpu_count(logical=False)):
            item = QListWidgetItem(f"Core {i}")
            item.setData(Qt.UserRole, i)
            self.cpu_list.addItem(item)
            if i in cpu_settings.cpu_cores:
                item.setSelected(True)
        core_layout.addWidget(QLabel("Select CPU Cores:"))
        core_layout.addWidget(self.cpu_list)

        self.bind_checkbox = QCheckBox(APP_STRINGS["cpu_binding"])
        self.bind_checkbox.setChecked(cpu_settings.enable_binding)
        core_layout.addWidget(self.bind_checkbox)

        core_tab.setLayout(core_layout)
        tab_widget.addTab(core_tab, "CPU Binding")

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(tab_widget)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def accept(self):
        self.cpu_settings.cpu_threads = self.thread_spin.value()
        self.cpu_settings.batch_size = self.batch_spin.value()
        self.cpu_settings.learning_rate = self.lr_spin.value()
        self.cpu_settings.use_half_precision = self.half_precision_checkbox.isChecked()
        self.cpu_settings.enable_async = self.async_checkbox.isChecked()
        self.cpu_settings.enable_binding = self.bind_checkbox.isChecked()
        self.cpu_settings.cpu_cores = [item.data(Qt.UserRole)
                                       for item in self.cpu_list.selectedItems()]
        super().accept()


# ================== MAIN APPLICATION ==================
class DeepLearningGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cpu_settings = CPUSettings()
        self.knowledge_base = KnowledgeBase(self.cpu_settings)
        self.selected_models = []
        self.training_threads = {}
        self.trained_models = {}
        self.text_generator = TextGenerationModel()
        self.init_ui()
        self.scan_local_models()  # Fixed: Removed call to non-existent method

    def init_ui(self):
        self.setWindowTitle(APP_STRINGS["app_title"])
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left panel (30% width)
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout()

        # Ollama service status
        service_group = QGroupBox("Ollama Service")
        service_layout = QVBoxLayout()
        self.service_status = QLabel(APP_STRINGS["ollama_unavailable"])
        self.service_status.setStyleSheet("font-weight: bold;")
        service_layout.addWidget(self.service_status)
        service_group.setLayout(service_layout)
        left_layout.addWidget(service_group)

        # Model selection
        model_group = QGroupBox(APP_STRINGS["model_selection"])
        model_layout = QVBoxLayout()

        # Add multi-selection hint
        self.select_hint = QLabel(APP_STRINGS["select_multiple"])
        self.select_hint.setStyleSheet("font-style: italic; color: gray;")
        model_layout.addWidget(self.select_hint)

        self.model_list = QListWidget()
        # Enable multi-selection
        self.model_list.setSelectionMode(QListWidget.ExtendedSelection)
        model_layout.addWidget(self.model_list)

        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton(APP_STRINGS["refresh_models"])
        self.refresh_btn.clicked.connect(self.scan_local_models)
        refresh_layout.addWidget(self.refresh_btn)

        model_layout.addLayout(refresh_layout)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Model status
        self.model_status_label = QLabel(APP_STRINGS["scanning_models"])
        self.model_status_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.model_status_label)

        # Training settings
        train_settings_group = QGroupBox(APP_STRINGS["training_control"])
        train_settings_layout = QFormLayout()

        self.epoch_combo = QComboBox()
        self.epoch_combo.addItems(["3", "5", "10", "20", "50"])
        self.epoch_combo.setCurrentIndex(1)
        train_settings_layout.addRow(APP_STRINGS["training_epochs"], self.epoch_combo)

        settings_btn_layout = QHBoxLayout()
        self.settings_btn = QPushButton(APP_STRINGS["configure_cpu"])
        self.settings_btn.clicked.connect(self.open_training_settings)
        settings_btn_layout.addWidget(self.settings_btn)

        self.settings_status = QLabel(self.get_settings_summary())
        settings_btn_layout.addWidget(self.settings_status)

        train_settings_layout.addRow(settings_btn_layout)
        train_settings_group.setLayout(train_settings_layout)
        left_layout.addWidget(train_settings_group)

        # Training control
        train_control_group = QGroupBox("Training Actions")
        train_control_layout = QHBoxLayout()

        self.start_btn = QPushButton(APP_STRINGS["start_training"])
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton(APP_STRINGS["stop_training"])
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.interact_btn = QPushButton(APP_STRINGS["interact_model"])
        self.interact_btn.clicked.connect(self.open_model_dialog)
        self.interact_btn.setEnabled(False)
        self.compare_btn = QPushButton("Compare Models")
        self.compare_btn.clicked.connect(self.open_model_comparison)

        train_control_layout.addWidget(self.start_btn)
        train_control_layout.addWidget(self.stop_btn)
        train_control_layout.addWidget(self.interact_btn)
        train_control_layout.addWidget(self.compare_btn)
        train_control_group.setLayout(train_control_layout)
        left_layout.addWidget(train_control_group)

        # Data management
        data_group = QGroupBox(APP_STRINGS["kb_management"])
        data_layout = QVBoxLayout()

        self.import_btn = QPushButton(APP_STRINGS["import_kb"])
        self.import_btn.clicked.connect(self.import_knowledge)
        data_layout.addWidget(self.import_btn)

        self.kb_status = QLabel(APP_STRINGS["no_kb"])
        data_layout.addWidget(self.kb_status)

        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        left_layout.addStretch(1)
        left_panel.setLayout(left_layout)

        # Right panel (70% width)
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.tabs = QTabWidget()

        # Training monitor tab
        self.monitor_tab = QWidget()
        monitor_layout = QVBoxLayout()
        self.monitor_canvas = TrainingMonitor()
        monitor_layout.addWidget(self.monitor_canvas)
        self.training_status = QLabel(APP_STRINGS["ready"])
        monitor_layout.addWidget(self.training_status)
        self.monitor_tab.setLayout(monitor_layout)

        # Chat tab
        self.chat_tab = QWidget()
        chat_layout = QVBoxLayout()

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)

        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(80)
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.chat_input, 5)
        input_layout.addWidget(self.send_btn, 1)

        chat_layout.addWidget(self.chat_output, 5)
        chat_layout.addLayout(input_layout, 1)
        self.chat_tab.setLayout(chat_layout)

        self.tabs.addTab(self.monitor_tab, APP_STRINGS["monitor_tab"])
        self.tabs.addTab(self.chat_tab, APP_STRINGS["chat_tab"])

        right_layout.addWidget(self.tabs)
        right_panel.setLayout(right_layout)

        # Add to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        # Set up selection handler
        self.model_list.itemSelectionChanged.connect(self.on_model_selected)

        # Set up status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_ollama_service)
        self.status_timer.start(5000)  # Check every 5 seconds

    def get_settings_summary(self):
        return (f"Threads: {self.cpu_settings.cpu_threads} | "
                f"Batch: {self.cpu_settings.batch_size} | "
                f"LR: {self.cpu_settings.learning_rate:.0e}")

    def open_training_settings(self):
        dialog = TrainingSettingsDialog(self.cpu_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.settings_status.setText(self.get_settings_summary())
            self.knowledge_base = KnowledgeBase(self.cpu_settings)

    def check_ollama_service(self):
        """Check if Ollama service is running"""
        try:
            # Try to get model list to verify service
            self.text_generator.client.list()
            self.service_status.setText(APP_STRINGS["ollama_running"])
            self.service_status.setStyleSheet("color: green;")
            return True
        except Exception:
            self.service_status.setText(APP_STRINGS["ollama_stopped"])
            self.service_status.setStyleSheet("color: red;")
            return False

    def scan_local_models(self):
        """Scan for locally installed Ollama models"""
        self.model_list.clear()
        self.model_status_label.setText(APP_STRINGS["scanning_models"])
        self.model_status_label.setStyleSheet("color: blue;")
        QApplication.processEvents()  # Update UI immediately

        # Scan models in a separate thread to prevent UI freeze
        self.scan_thread = ModelScanThread()
        self.scan_thread.finished.connect(self.handle_model_scan_result)
        self.scan_thread.start()

    def handle_model_scan_result(self, models, success):
        """Handle the result of model scanning"""
        if success and models:
            self.text_generator.available_models = models
            self.text_generator.model_name = models[0] if models else None

            # Populate model list
            for model in models:
                item = QListWidgetItem(model)
                self.model_list.addItem(item)

            # Select the first model by default
            if self.model_list.count() > 0:
                self.model_list.item(0).setSelected(True)
                self.on_model_selected()

            self.model_status_label.setText(
                APP_STRINGS["models_found"].format(len(models))
            )
            self.model_status_label.setStyleSheet("color: green;")
        else:
            self.model_list.addItem(APP_STRINGS["no_models"])
            self.model_list.addItem(APP_STRINGS["install_models"])
            self.model_status_label.setText(APP_STRINGS["model_scan_failed"])
            self.model_status_label.setStyleSheet("color: red;")

    def on_model_selected(self):
        """Update active model when user selects from list"""
        self.selected_models = [item.text() for item in self.model_list.selectedItems()]
        if not self.selected_models:
            return

        # Update status display
        status_text = f"Selected models: {', '.join(self.selected_models)}"
        self.model_status_label.setText(status_text)
        self.model_status_label.setStyleSheet("color: green;")

    def import_knowledge(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, APP_STRINGS["file_dialog"], "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )

        if file_paths:
            try:
                chunk_count = self.knowledge_base.import_data(file_paths)
                self.kb_status.setText(f"Data imported: {chunk_count} chunks")
            except Exception as e:
                self.kb_status.setText(f"Error: {str(e)}")

    def start_training(self):
        self.selected_models = [item.text() for item in self.model_list.selectedItems()]

        if not self.selected_models:
            QMessageBox.warning(self, APP_STRINGS["no_model"], APP_STRINGS["no_model_msg"])
            return

        if not self.knowledge_base.vector_store:
            QMessageBox.warning(self, "No Data", APP_STRINGS["no_kb_msg"])
            return

        epochs = int(self.epoch_combo.currentText())

        # Create training monitor
        self.monitor_canvas = TrainingMonitor()
        if self.monitor_tab.layout():
            old_canvas = self.monitor_tab.layout().itemAt(0).widget()
            if old_canvas:
                old_canvas.deleteLater()
            self.monitor_tab.layout().insertWidget(0, self.monitor_canvas)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.interact_btn.setEnabled(False)

        # Start training for all selected models
        for model_name in self.selected_models:
            thread = ModelTrainingThread(
                model_name,
                self.knowledge_base,
                self.cpu_settings,
                epochs=epochs
            )
            thread.progress.connect(self.update_training_progress)
            thread.finished.connect(self.model_training_finished)
            self.training_threads[model_name] = thread
            thread.start()

    def update_training_progress(self, model_name, epoch, train_loss, done, error):
        if error:
            self.training_status.setText(f"Error: {model_name} - {error}")
            return

        self.monitor_canvas.update_metrics(epoch, train_loss)
        self.training_status.setText(
            APP_STRINGS["training_status"].format(
                model_name, epoch, self.epoch_combo.currentText(),
                train_loss, 0.0  # Accuracy placeholder
            )
        )

    def model_training_finished(self, model_name, model_path):
        self.trained_models[model_name] = model_path
        self.training_status.setText(
            f"Completed training for {model_name}!\nSaved at: {model_path}"
        )

        # Enable buttons when all training completes
        if all(not thread.isRunning() for thread in self.training_threads.values()):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.interact_btn.setEnabled(True)

    def stop_training(self):
        for model_name, thread in self.training_threads.items():
            if thread.isRunning():
                thread.stop()
                thread.wait()  # Wait for thread to safely exit

        self.training_status.setText(APP_STRINGS["training_stopped"])
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.interact_btn.setEnabled(True)

    def open_model_dialog(self):
        if not self.trained_models:
            QMessageBox.warning(self, APP_STRINGS["no_trained_models"],
                                APP_STRINGS["no_trained_models_msg"])
            return

        # Create a dialog to select which trained model to interact with
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Model for Interaction")
        layout = QVBoxLayout()

        label = QLabel("Select a trained model to interact with:")
        layout.addWidget(label)

        model_list = QListWidget()
        for model_name in self.trained_models:
            model_list.addItem(model_name)
        layout.addWidget(model_list)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted and model_list.currentItem():
            model_name = model_list.currentItem().text()
            model_path = self.trained_models[model_name]
            interaction_dialog = ModelInteractionDialog(model_name, model_path, self.knowledge_base, self)
            interaction_dialog.exec_()

    def open_model_comparison(self):
        """Open a dialog to compare multiple models"""
        selected_models = [item.text() for item in self.model_list.selectedItems()]

        if len(selected_models) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least two models to compare")
            return

        comparison_dialog = ModelComparisonDialog(selected_models, self.knowledge_base, self)
        comparison_dialog.exec_()

    def send_message(self):
        message = self.chat_input.toPlainText().strip()
        if not message:
            return

        self.chat_output.append(f"You: {message}")
        self.chat_input.clear()
        self.chat_output.append("AI: Thinking...")
        QApplication.processEvents()

        try:
            context_chunks = self.knowledge_base.retrieve_context(message, k=3)
            context = "\n".join([c.page_content for c in context_chunks])
            response = self.text_generator.generate_text(message, context)
            self.chat_output.append(f"AI: {response}")
        except Exception as e:
            self.chat_output.append(f"AI: Error - {str(e)}")


# ================== MODEL SCAN THREAD ==================
class ModelScanThread(QThread):
    finished = pyqtSignal(list, bool)  # Models list, success status

    def run(self):
        models = []
        success = False
        try:
            # Use Ollama CLI to list models
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse output to get model names
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:  # Ensure there's at least one part
                            model_name = parts[0]
                            models.append(model_name)
                success = True
        except Exception as e:
            print(f"Error scanning local models: {str(e)}")

        self.finished.emit(models, success)


# ================== APPLICATION ENTRY POINT ==================
def main():
    app = QApplication(sys.argv)

    # Global exception handler
    def excepthook(exc_type, exc_value, exc_tb):
        import traceback
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(f"Error:\n{tb}")
        with open("error.log", "a") as f:
            f.write(f"Error: {tb}\n")

    sys.excepthook = excepthook

    # Configure PyTorch
    torch.set_num_threads(mp.cpu_count() - 1)
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() - 1)
    os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count() - 1)

    window = DeepLearningGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()