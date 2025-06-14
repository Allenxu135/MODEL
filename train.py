import os
import json
import csv
import argparse
from datetime import datetime
from model import OllamaModel


class KnowledgeTrainer:
    def __init__(self, base_model="mistral"):
        self.base_model = base_model
        self.client = OllamaModel(base_model)

    def _load_data(self, data_path):
        """Load and convert CSV/TXT/JSON to standardized format"""
        if data_path.endswith('.csv'):
            with open(data_path, 'r', encoding='utf-8') as f:
                return [{"instruction": row['question'], "output": row['answer']}
                        for row in csv.DictReader(f)]

        elif data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        else:  # TXT processing
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        parts = line.strip().split('|', 1)
                        data.append({"instruction": parts[0].strip(), "output": parts[1].strip()})
                    else:
                        data.append({"instruction": line.strip(), "output": ""})
            return data

    def _generate_prompts(self, data):
        """Convert knowledge to prompt-completion pairs"""
        return [
            f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            for item in data
        ]

    def train(self, data_path, output_model_name, epochs=3):
        """
        Core training workflow:
        :param data_path: Knowledge file path (CSV/TXT/JSON)
        :param output_model_name: New model identifier
        :param epochs: Training iterations
        """
        # 1. Load and standardize knowledge
        knowledge = self._load_data(data_path)
        print(f"✅ Loaded {len(knowledge)} knowledge entries")

        # 2. Prepare training data
        training_data = self._generate_prompts(knowledge)

        # 3. Configure adapter training (Parameter-Efficient Fine-Tuning)
        from peft import LoraConfig, get_peft_model
        from transformers import TrainingArguments, Trainer

        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        adapter_model = get_peft_model(self.client.model, peft_config)

        # 4. Set up training environment
        training_args = TrainingArguments(
            output_dir="./adapters",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            save_strategy="epoch",
            learning_rate=2e-4,
            report_to="none"
        )

        # 5. Execute training
        trainer = Trainer(
            model=adapter_model,
            args=training_args,
            train_dataset=training_data,
        )
        print("🚀 Starting knowledge injection...")
        trainer.train()

        # 6. Save deployable model
        self._export_model(adapter_model, output_model_name)
        print(f"🎉 Knowledge-optimized model saved as '{output_model_name}'")
        print(f"Use in model.py: model = OllamaModel('{output_model_name}')")

    def _export_model(self, model, name):
        """Package model for Ollama deployment"""
        from transformers import AutoTokenizer
        import torch

        # 1. Save adapter weights
        model.save_pretrained(f"./adapters/{name}")

        # 2. Generate Ollama Modelfile
        modelfile = f"""
FROM {self.base_model}
ADAPTER ./adapters/{name}
TEMPLATE \"\"\"[INST] {{ .System }} {{ .Prompt }} [/INST]
\"\"\"
SYSTEM \"\"\"You are a {name} assistant with specialized knowledge\"\"\"
        """
        with open(f"./models/Modelfile.{name}", "w") as f:
            f.write(modelfile.strip())

        # 3. Build new Ollama model
        os.system(f"ollama create {name} -f ./models/Modelfile.{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Injection Trainer")
    parser.add_argument("--data", required=True, help="Path to knowledge file (CSV/TXT/JSON)")
    parser.add_argument("--model", required=True, help="Output model name")
    parser.add_argument("--base", default="mistral", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")

    args = parser.parse_args()

    print(f"🧠 Starting knowledge training at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Base model: {args.base}")
    print(f"Knowledge source: {args.data}")

    trainer = KnowledgeTrainer(args.base)
    trainer.train(args.data, args.model, args.epochs)