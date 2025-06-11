import ollama
import time
from pathlib import Path
import json
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, base_model: str, data_path: str, output_name: str):
        self.base_model = base_model
        self.data_path = data_path
        self.output_name = output_name
        self.validation_split = 0.15  # 15%数据用于验证

    def _prepare_dataset(self) -> str:
        """将数据集转换为Ollama要求的聊天格式"""
        # 读取并预处理完整数据集
        with open(self.data_path, 'r', encoding='utf-8') as f:
            samples = [line.strip() for line in f if line.strip()]

        # 划分训练/验证集
        split_idx = int(len(samples) * (1 - self.validation_split))
        train_data = samples[:split_idx]
        val_data = samples[split_idx:]

        # 转换为Ollama格式[4](@ref)
        def format_chat(sample):
            return {
                "messages": [
                    {"role": "user", "content": sample.split('|')[0]},
                    {"role": "assistant", "content": sample.split('|')[1]}
                ]
            }

        # 保存训练集
        train_path = f"{self.output_name}_train.jsonl"
        with open(train_path, 'w') as f:
            for sample in train_data:
                f.write(json.dumps(format_chat(sample)) + '\n')

        # 保存验证集
        val_path = f"{self.output_name}_val.jsonl"
        with open(val_path, 'w') as f:
            for sample in val_data:
                f.write(json.dumps(format_chat(sample)) + '\n')

        return train_path, val_path

    def _create_modelfile(self, train_path: str) -> str:
        """生成包含完整训练参数的Modelfile"""
        modelfile_content = f"""
FROM {self.base_model}
SYSTEM "您是一个领域专家助手"
PARAMETER num_epochs 5
PARAMETER learning_rate 0.0002
PARAMETER num_ctx 4096
TEMPLATE """
        # 添加完整训练数据引用
        modelfile_content += f"\"\"\"{Path(train_path).read_text(encoding='utf-8')}\"\"\"\n"

        modelfile_path = f"{self.output_name}.Modelfile"
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        return modelfile_path

    def _monitor_training(self, job_id: str):
        """实时监控训练进度"""
        print(f"🚀 训练已启动 | 任务ID: {job_id}")
        progress_bar = tqdm(total=100, desc="训练进度")
        prev_progress = 0

        while True:
            status = ollama.show(job_id)
            current = status.get('progress', 0)

            # 更新进度条
            if current > prev_progress:
                progress_bar.update(current - prev_progress)
                prev_progress = current

            # 完成检测
            if status.get('status') == 'completed':
                progress_bar.close()
                print(f"✅ 训练完成! 峰值准确率: {status['metrics']['accuracy']:.2%}")
                return

            # 错误处理
            if status.get('status') == 'failed':
                progress_bar.close()
                print(f"❌ 训练失败: {status['error']}")
                exit(1)

            time.sleep(5)

    def fine_tune(self):
        """执行端到端微调流程"""
        # 1. 数据预处理
        train_path, val_path = self._prepare_dataset()
        print(
            f"📊 数据集加载完成 | 训练样本: {Path(train_path).read_text().count('')} | 验证样本: {Path(val_path).read_text().count('')}")

        # 2. 创建训练配置
        modelfile_path = self._create_modelfile(train_path)
        print(f"⚙️ 生成Modelfile: {modelfile_path}")

        # 3. 启动训练任务
        job = ollama.create(
            name=self.output_name,
            modelfile=modelfile_path,
            stream=False
        )

        # 4. 监控训练过程
        self._monitor_training(job['id'])

        # 5. 模型验证
        val_results = ollama.evaluate(
            model=self.output_name,
            dataset=val_path,
            metrics=['perplexity', 'accuracy']
        )
        print(f"🧪 验证结果: 困惑度={val_results['perplexity']:.2f} | 准确率={val_results['accuracy']:.2%}")

        return {
            "model": self.output_name,
            "val_metrics": val_results
        }


if __name__ == "__main__":
    trainer = ModelTrainer(
        base_model="llama3",
        data_path="finance_qa.txt",  # 格式: 问题|答案
        output_name="finance-expert-v1"
    )

    result = trainer.fine_tune()
    print(f"\n🦙 新模型 '{result['model']}' 已部署!")
    print("💡 在model.py中选择此模型进行对话测试")