import sys
import os
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate

import ollama
import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QLabel, QComboBox, QListWidget, QListWidgetItem,
    QTabWidget, QProgressBar, QGroupBox, QSplitter, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt, QSize, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class KnowledgeBase:
    def __init__(self):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def import_data(self, file_paths):
        """从文件路径列表导入知识库，支持多种格式"""
        documents = []

        for file_path in file_paths:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file_path.endswith('.csv'):
                loader = CSVLoader(file_path)
                documents.extend(loader.load())
            # 可扩展其他格式

        if not documents:
            return 0

        # 分割文档
        chunks = self.text_splitter.split_documents(documents)

        # 创建向量存储
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        return len(chunks)

    def retrieve_context(self, query, k=5):
        """检索与查询相关的上下文"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []


class TrainingMonitor(FigureCanvas):
    """实时训练指标监控画布"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        # 创建两个子图
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)

        # 初始化数据
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epochs = []

        # 初始化曲线
        self.loss_line1, = self.ax1.plot([], [], 'b-', label='Train Loss')
        self.loss_line2, = self.ax1.plot([], [], 'r-', label='Val Loss')
        self.acc_line1, = self.ax2.plot([], [], 'b-', label='Train Acc')
        self.acc_line2, = self.ax2.plot([], [], 'r-', label='Val Acc')

        # 设置图表样式
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epochs')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epochs')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True, linestyle='--', alpha=0.7)

        self.figure.tight_layout()

    def update_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """更新指标数据并刷新图表"""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        # 更新曲线数据
        self.loss_line1.set_data(self.epochs, self.train_loss)
        self.loss_line2.set_data(self.epochs, self.val_loss)
        self.acc_line1.set_data(self.epochs, self.train_acc)
        self.acc_line2.set_data(self.epochs, self.val_acc)

        # 调整坐标轴范围
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

        # 重绘画布
        self.draw()


class ModelTrainingThread(threading.Thread):
    """模型训练线程"""

    def __init__(self, model_name, knowledge_base, callback):
        super().__init__()
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.callback = callback
        self.running = True

    def run(self):
        """模拟训练过程，实际应用中需替换为真实训练逻辑"""
        epochs = 10
        train_loss = [0.8 - 0.07 * i for i in range(epochs)]
        val_loss = [0.75 - 0.06 * i for i in range(epochs)]
        train_acc = [0.2 + 0.08 * i for i in range(epochs)]
        val_acc = [0.15 + 0.085 * i for i in range(epochs)]

        for epoch in range(epochs):
            if not self.running:
                break

            # 模拟训练一个epoch
            time.sleep(1)

            # 回调更新UI
            self.callback(self.model_name, epoch + 1, train_loss[epoch],
                          val_loss[epoch], train_acc[epoch], val_acc[epoch])

        # 训练完成
        self.callback(self.model_name, epochs, train_loss[-1],
                      val_loss[-1], train_acc[-1], val_acc[-1], done=True)

    def stop(self):
        """停止训练"""
        self.running = False


class DeepLearningGUI(QMainWindow):
    """深度学习应用主界面"""

    def __init__(self):
        super().__init__()
        self.knowledge_base = KnowledgeBase()
        self.selected_models = []
        self.training_threads = {}
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("多模型深度学习平台")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 左侧面板
        left_panel = QGroupBox("控制面板")
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)

        # 模型选择区域
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.MultiSelection)
        model_layout.addWidget(self.model_list)

        self.refresh_models_btn = QPushButton("刷新模型列表")
        self.refresh_models_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.refresh_models_btn)

        # 知识库导入区域
        kb_group = QGroupBox("知识库管理")
        kb_layout = QVBoxLayout()
        kb_group.setLayout(kb_layout)

        self.import_kb_btn = QPushButton("导入知识库")
        self.import_kb_btn.clicked.connect(self.import_knowledge)
        kb_layout.addWidget(self.import_kb_btn)

        self.kb_status_label = QLabel("未导入知识库")
        kb_layout.addWidget(self.kb_status_label)

        # 训练控制区域
        train_group = QGroupBox("训练控制")
        train_layout = QVBoxLayout()
        train_group.setLayout(train_layout)

        self.start_train_btn = QPushButton("开始训练")
        self.start_train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("停止训练")
        self.stop_train_btn.clicked.connect(self.stop_training)
        train_layout.addWidget(self.stop_train_btn)

        # 添加到左侧面板
        left_layout.addWidget(model_group)
        left_layout.addWidget(kb_group)
        left_layout.addWidget(train_group)
        left_layout.addStretch(1)

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # 选项卡
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # 训练监控标签页
        self.monitor_tab = QWidget()
        self.monitor_layout = QVBoxLayout()
        self.monitor_tab.setLayout(self.monitor_layout)

        # 创建训练监控图表
        self.monitor_canvas = TrainingMonitor(self)
        self.monitor_layout.addWidget(self.monitor_canvas)

        # 添加状态标签
        self.status_label = QLabel("准备就绪")
        self.monitor_layout.addWidget(self.status_label)

        # 聊天标签页（保留原功能）
        self.chat_tab = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_tab.setLayout(self.chat_layout)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.chat_layout.addWidget(self.chat_output)

        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(100)
        self.chat_layout.addWidget(self.chat_input)

        self.send_btn = QPushButton("发送")
        self.chat_layout.addWidget(self.send_btn)

        # 添加标签页
        self.tabs.addTab(self.monitor_tab, "训练监控")
        self.tabs.addTab(self.chat_tab, "多模型聊天")

        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 初始化模型列表
        self.load_models()

    def load_models(self):
        """加载本地可用的Ollama模型"""
        self.model_list.clear()
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = [model["name"] for model in response.json().get("models", [])]
            for model in models:
                item = QListWidgetItem(model)
                self.model_list.addItem(item)
            self.statusBar().showMessage(f"已加载 {len(models)} 个模型")
        except Exception as e:
            self.statusBar().showMessage(f"加载模型失败: {str(e)}")

    def import_knowledge(self):
        """导入知识库文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择知识库文件", "",
            "文本文件 (*.txt);;CSV文件 (*.csv);;所有文件 (*.*)"
        )

        if file_paths:
            try:
                chunk_count = self.knowledge_base.import_data(file_paths)
                self.kb_status_label.setText(f"已导入 {chunk_count} 个文本片段")
                self.statusBar().showMessage("知识库导入成功")
            except Exception as e:
                QMessageBox.critical(self, "导入失败", f"导入知识库时出错: {str(e)}")
                self.statusBar().showMessage("知识库导入失败")

    def start_training(self):
        """开始训练选中的模型"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "未选择模型", "请先选择要训练的模型")
            return

        if not self.knowledge_base.vector_store:
            QMessageBox.warning(self, "知识库为空", "请先导入知识库")
            return

        # 获取选中的模型
        self.selected_models = [item.text() for item in selected_items]

        # 重置监控图表
        self.monitor_canvas = TrainingMonitor(self)
        self.monitor_layout.replaceWidget(self.monitor_layout.itemAt(0).widget(), self.monitor_canvas)

        # 开始逐个训练模型
        self.current_model_index = 0
        self.train_next_model()

    def train_next_model(self):
        """训练下一个模型"""
        if self.current_model_index >= len(self.selected_models):
            self.status_label.setText("所有模型训练完成！")
            return

        model_name = self.selected_models[self.current_model_index]
        self.status_label.setText(f"正在训练模型: {model_name}")

        # 创建并启动训练线程
        thread = ModelTrainingThread(
            model_name,
            self.knowledge_base,
            self.update_training_progress
        )
        self.training_threads[model_name] = thread
        thread.start()

    def update_training_progress(self, model_name, epoch, train_loss, val_loss, train_acc, val_acc, done=False):
        """更新训练进度和图表"""
        if model_name != self.selected_models[self.current_model_index]:
            return

        # 更新图表
        self.monitor_canvas.update_metrics(epoch, train_loss, val_loss, train_acc, val_acc)

        # 更新状态
        status = f"模型: {model_name} | Epoch: {epoch}/10 | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.4f}"
        self.status_label.setText(status)

        # 如果当前模型训练完成，开始下一个
        if done:
            self.current_model_index += 1
            self.train_next_model()

    def stop_training(self):
        """停止所有训练线程"""
        for model_name, thread in self.training_threads.items():
            thread.stop()
        self.status_label.setText("训练已停止")
        self.statusBar().showMessage("训练已停止")


def main():
    app = QApplication(sys.argv)
    window = DeepLearningGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()