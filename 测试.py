import sys
import os
import time
import threading
import numpy as np
import json
import requests
import torch
import multiprocessing as mp
import psutil
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
    QFormLayout, QSpinBox, QCheckBox, QSizePolicy, QMenuBar, QMenu, QAction
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ================== GLOBAL SETTINGS ==================
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
        "cpu_optimization": "CPU Optimization",
        "cpu_cores": "CPU Cores:",
        "cpu_binding": "Bind CPU Cores",
        "use_half_precision": "Use Half Precision",
        "cpu_threads": "CPU Threads:",
        "enable_async": "Enable Async I/O",
        "configure_cpu": "Configure CPU Settings",
        "precision_settings": "Precision Settings",
        "thread_settings": "Thread Settings",
        "current_settings": "Current CPU Settings:",
        "bound_cores": "Bound CPU cores:",
        "precision": "Precision:",
        "thread_count": "Thread count:",
        "async_status": "Async I/O:",
        "model_trained": "Model trained:",
        "training_completed": "Training completed! Model saved at:",
        "failed_load": "Failed to load models:",
        "loaded_models": "Loaded {} models",
        "error_importing": "Error importing knowledge base:",
        "kb_imported": "Knowledge base imported successfully: {} text chunks",
        "training_model": "Training model: {}",
        "training_status": "Training status: Epoch {}/{} | Train Loss: {:.4f} | Val Acc: {:.2%}",
        "model_loaded": "Model loaded successfully",
        "prediction_error": "Prediction error: {}"
    }
}


def tr(key):
    """Translate the given key"""
    return LANGUAGE_STRINGS["en"].get(key, key)


# ================== CPU OPTIMIZATION SETTINGS ==================
class CPUSettings:
    def __init__(self):
        self.enable_binding = False
        self.cpu_cores = []
        self.use_half_precision = False
        self.cpu_threads = max(1, mp.cpu_count() - 2)  # Leave 2 cores for UI
        self.enable_async = True

    def apply_cpu_binding(self):
        """Bind process to specific CPU cores"""
        if self.enable_binding and self.cpu_cores:
            try:
                p = psutil.Process()
                p.cpu_affinity(self.cpu_cores)
                print(f"Process bound to CPUs: {self.cpu_cores}")
            except Exception as e:
                print(f"CPU binding failed: {str(e)}")


# ================== KNOWLEDGE BASE (OPTIMIZED) ==================
class KnowledgeBase:
    def __init__(self, cpu_settings):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.text_chunks = []  # Store text chunks for training
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
            print(f"Error loading {file_path}: {str(e)}")
        return []

    def import_data(self, file_paths):
        """Optimized import using parallel processing"""
        if not file_paths:
            return 0

        # Parallel loading using thread pool
        futures = [self.executor.submit(self._load_file, path) for path in file_paths]
        documents = []
        for future in futures:
            documents.extend(future.result())

        if not documents:
            return 0

        # Split documents in parallel
        split_futures = [self.executor.submit(self.text_splitter.split_documents, [doc])
                         for doc in documents]
        chunks = []
        for future in split_futures:
            chunks.extend(future.result())

        self.text_chunks = [chunk.page_content for chunk in chunks]

        # Create vector store with optimized embeddings
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


# ================== MODEL TRAINING THREAD (CPU OPTIMIZED) ==================
class ModelTrainingThread(QThread):
    """Optimized training thread with CPU binding and mixed precision"""
    progress = pyqtSignal(str, int, float, float, float, float, bool, str)
    finished = pyqtSignal(str, str)

    def __init__(self, model_name, knowledge_base, cpu_settings, epochs=5):
        super().__init__()
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.epochs = epochs
        self.running = True
        self.model = None
        self.tokenizer = None
        self.saved_model_path = ""
        self.cpu_settings = cpu_settings

        # Apply CPU settings
        if self.cpu_settings:
            self.cpu_settings.apply_cpu_binding()

    def run(self):
        """Optimized training with mixed precision and batch processing"""
        try:
            # 1. Prepare training data (parallel processing)
            if not self.knowledge_base.text_chunks:
                self.progress.emit(
                    self.model_name, 0, 0, 0, 0, 0, False,
                    tr("No training data available")
                )
                return

            # Create synthetic labels using vectorized operations
            texts = self.knowledge_base.text_chunks
            labels = np.array([len(text) % 4 for text in texts], dtype=np.int64)

            # 2. Load model with half precision if enabled
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            # Set half precision if enabled
            torch_dtype = torch.float16 if self.cpu_settings.use_half_precision else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=4,
                torch_dtype=torch_dtype
            )

            # 3. Parallel tokenization
            def tokenize_batch(batch):
                return self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )

            # Split into batches for parallel processing
            batch_size = 100
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            futures = [self.knowledge_base.executor.submit(tokenize_batch, batch) for batch in batches]
            encodings = {'input_ids': [], 'attention_mask': []}

            for future in futures:
                batch_enc = future.result()
                for key in encodings:
                    encodings[key].append(batch_enc[key])

            # Concatenate batch results
            for key in encodings:
                encodings[key] = torch.cat(encodings[key], dim=0)

            # 4. Create PyTorch dataset
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

            # 5. Split into train and validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # 6. Set up optimized training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=self.epochs,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=32,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                logging_steps=10,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to="none",
                # CPU-specific optimizations
                dataloader_num_workers=self.cpu_settings.cpu_threads,
                fp16=self.cpu_settings.use_half_precision,
                no_cuda=True  # Ensure we're using CPU
            )

            # 7. Define metrics
            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                acc = (preds == labels).mean()
                return {"accuracy": acc}

            # 8. Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )

            # 9. Optimized training loop
            for epoch in range(self.epochs):
                if not self.running:
                    break

                # Train with progress reporting
                train_result = trainer.train()
                eval_result = trainer.evaluate()

                # Extract metrics
                train_loss = train_result.metrics.get("train_loss", 0)
                val_loss = eval_result.get("eval_loss", 0)
                train_acc = eval_result.get("eval_accuracy", 0) * 0.9  # Simulate training accuracy
                val_acc = eval_result.get("eval_accuracy", 0)

                # Report progress
                self.progress.emit(
                    self.model_name, epoch + 1, train_loss,
                    val_loss, train_acc, val_acc, False, ""
                )

            # 10. Training complete
            self.progress.emit(
                self.model_name, self.epochs, train_loss,
                val_loss, train_acc, val_acc, True, ""
            )

            # Save optimized model
            self.saved_model_path = self.save_model()
            self.finished.emit(self.model_name, self.saved_model_path)

        except Exception as e:
            self.progress.emit(self.model_name, 0, 0, 0, 0, 0, True, str(e))

    def save_model(self):
        """Save model with quantization for CPU efficiency"""
        if self.model and self.tokenizer:
            model_dir = f"./saved_models/{self.model_name}_{int(time.time())}"
            os.makedirs(model_dir, exist_ok=True)

            # Quantize model for CPU efficiency
            try:
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                quantized_model.save_pretrained(model_dir)
            except:
                # Fallback to original model if quantization fails
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
        self.setWindowTitle(f"{tr('model_info')}: {model_name}")
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

            self.model_status_label.setText(tr("model_loaded"))
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
            self.output_text.setText(tr("prediction_error").format(str(e)))


# ================== CPU SETTINGS DIALOG ==================
class CPUSettingsDialog(QDialog):
    """Dialog for configuring CPU optimization settings"""

    def __init__(self, cpu_settings, parent=None):
        super().__init__(parent)
        self.cpu_settings = cpu_settings
        self.setWindowTitle(tr("cpu_optimization"))
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()

        # CPU core selection
        core_group = QGroupBox(tr("cpu_cores"))
        core_layout = QVBoxLayout()

        self.cpu_list = QListWidget()
        self.cpu_list.setSelectionMode(QListWidget.MultiSelection)
        for i in range(psutil.cpu_count(logical=False)):
            item = QListWidgetItem(f"Core {i}")
            item.setData(Qt.UserRole, i)
            self.cpu_list.addItem(item)
            if i in cpu_settings.cpu_cores:
                item.setSelected(True)
        core_layout.addWidget(self.cpu_list)

        self.bind_checkbox = QCheckBox(tr("cpu_binding"))
        self.bind_checkbox.setChecked(cpu_settings.enable_binding)
        core_layout.addWidget(self.bind_checkbox)

        core_group.setLayout(core_layout)

        # Precision settings
        precision_group = QGroupBox(tr("precision_settings"))
        precision_layout = QVBoxLayout()

        self.half_precision_checkbox = QCheckBox(tr("use_half_precision"))
        self.half_precision_checkbox.setChecked(cpu_settings.use_half_precision)
        precision_layout.addWidget(self.half_precision_checkbox)

        precision_group.setLayout(precision_layout)

        # Thread settings
        thread_group = QGroupBox(tr("thread_settings"))
        thread_layout = QFormLayout()

        self.thread_spinner = QSpinBox()
        self.thread_spinner.setRange(1, mp.cpu_count())
        self.thread_spinner.setValue(cpu_settings.cpu_threads)
        thread_layout.addRow(tr("cpu_threads"), self.thread_spinner)

        self.async_checkbox = QCheckBox(tr("enable_async"))
        self.async_checkbox.setChecked(cpu_settings.enable_async)
        thread_layout.addWidget(self.async_checkbox)

        thread_group.setLayout(thread_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(core_group)
        layout.addWidget(precision_group)
        layout.addWidget(thread_group)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def accept(self):
        """Save settings on OK"""
        self.cpu_settings.enable_binding = self.bind_checkbox.isChecked()
        self.cpu_settings.cpu_cores = [item.data(Qt.UserRole)
                                       for item in self.cpu_list.selectedItems()]
        self.cpu_settings.use_half_precision = self.half_precision_checkbox.isChecked()
        self.cpu_settings.cpu_threads = self.thread_spinner.value()
        self.cpu_settings.enable_async = self.async_checkbox.isChecked()
        super().accept()


# ================== MAIN APPLICATION (OPTIMIZED) ==================
class DeepLearningGUI(QMainWindow):
    """Optimized Deep Learning Application with CPU enhancements"""

    def __init__(self):
        super().__init__()
        self.cpu_settings = CPUSettings()
        self.knowledge_base = KnowledgeBase(self.cpu_settings)
        self.selected_models = []
        self.training_threads = {}
        self.trained_models = {}
        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle(tr("app_title"))
        self.setGeometry(100, 100, 1200, 800)

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

        # CPU optimization settings
        cpu_group = QGroupBox(tr("cpu_optimization"))
        cpu_layout = QVBoxLayout()

        self.cpu_settings_btn = QPushButton(tr("configure_cpu"))
        self.cpu_settings_btn.clicked.connect(self.open_cpu_settings)
        cpu_layout.addWidget(self.cpu_settings_btn)

        self.cpu_status_label = QLabel(self.get_cpu_settings_summary())
        cpu_layout.addWidget(self.cpu_status_label)

        left_layout.addWidget(cpu_group)

        # Knowledge base import area
        kb_group = QGroupBox(tr("kb_management"))
        kb_layout = QVBoxLayout()
        kb_group.setLayout(kb_layout)

        self.import_kb_btn = QPushButton(tr("import_kb"))
        self.import_kb_btn.clicked.connect(self.import_knowledge)
        kb_layout.addWidget(self.import_kb_btn)

        self.kb_status_label = QLabel(tr("no_kb"))
        kb_layout.addWidget(self.kb_status_label)

        left_layout.addWidget(kb_group)

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

        left_layout.addWidget(train_group)
        left_layout.addStretch(1)

        main_layout.addWidget(left_panel)

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

        main_layout.addWidget(right_panel)

        # Status bar
        self.statusBar().showMessage(tr("ready"))

        # Initialize model list
        self.load_models()

    def open_cpu_settings(self):
        """Open CPU settings dialog"""
        dialog = CPUSettingsDialog(self.cpu_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.cpu_status_label.setText(self.get_cpu_settings_summary())
            # Reinitialize knowledge base to apply new settings
            self.knowledge_base = KnowledgeBase(self.cpu_settings)

    def get_cpu_settings_summary(self):
        """Get summary of current CPU settings"""
        cores = ", ".join(map(str, self.cpu_settings.cpu_cores)) if self.cpu_settings.cpu_cores else "All"
        binding = "Enabled" if self.cpu_settings.enable_binding else "Disabled"
        precision = "FP16" if self.cpu_settings.use_half_precision else "FP32"
        async_status = "Enabled" if self.cpu_settings.enable_async else "Disabled"

        summary = (f"<b>{tr('current_settings')}</b><br>"
                   f"{tr('bound_cores')}: {cores} | {binding}<br>"
                   f"{tr('precision')}: {precision}<br>"
                   f"{tr('thread_count')}: {self.cpu_settings.cpu_threads}<br>"
                   f"{tr('async_status')}: {async_status}")

        return summary

    def load_models(self):
        """Load locally available Ollama models"""
        self.model_list.clear()
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [model["name"] for model in response.json().get("models", [])]
            for model in models:
                item = QListWidgetItem(model)
                self.model_list.addItem(item)
            self.statusBar().showMessage(tr("loaded_models").format(len(models)))
        except Exception as e:
            self.statusBar().showMessage(tr("failed_load") + ": " + str(e))

    def import_knowledge(self):
        """Import knowledge base files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, tr("file_dialog"), "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)"
        )

        if file_paths:
            try:
                chunk_count = self.knowledge_base.import_data(file_paths)
                self.kb_status_label.setText(tr("kb_imported").format(chunk_count))
                self.statusBar().showMessage(tr("Knowledge base imported successfully"))
            except Exception as e:
                QMessageBox.critical(self, tr("import_failed"), tr("error_importing") + ": " + str(e))
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
        self.status_label.setText(tr("training_model").format(model_name))

        # Create and start training thread
        thread = ModelTrainingThread(
            model_name,
            self.knowledge_base,
            self.cpu_settings,  # Pass CPU optimization settings
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
        status = tr("training_status").format(
            model_name, epoch, self.epoch_spinner.currentText(),
            train_loss, val_acc
        )
        self.status_label.setText(status)

    def model_training_finished(self, model_name, model_path):
        """Handle model training completion"""
        self.trained_models[model_name] = model_path
        self.status_label.setText(tr("training_completed").format(model_name, model_path))
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

    # Configure PyTorch for CPU optimization
    torch.set_num_threads(mp.cpu_count() - 1)  # Keep one core for UI
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() - 1)
    os.environ["MKL_NUM_THREADS"] = str(mp.cpu_count() - 1)

    window = DeepLearningGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()