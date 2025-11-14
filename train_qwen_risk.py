import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os

os.environ["HTTP_PROXY"] = "http://172.23.224.1:7890"
os.environ["HTTPS_PROXY"] = "http://172.23.224.1:7890"
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import warnings
warnings.filterwarnings("ignore")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理函数
def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)[:1000]  # 限制样本数量
    
    # 过滤有效数据
    df = df[df['Lsa_summary'].notna() & df['risk_deepseek'].notna()]
    df = df[df['risk_deepseek'] != 0]  # 移除无效的风险标签
    
    print(f"有效数据数量: {len(df)}")
    print(f"风险分布: {df['risk_deepseek'].value_counts().sort_index()}")
    
    return df

def create_prompt_template(text, risk_score, stock_symbol="STOCK"):
    """创建训练提示模板 - 使用风险评估格式"""
    # 使用与risk_deepseek_deepinfra.py相同的对话格式
    system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
    
    # 构建用户输入
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    # 构建完整的对话
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant: {risk_score}"""
    
    return conversation

def prepare_dataset(df, tokenizer, max_length=512):
    """准备训练数据集"""
    print("正在准备数据集...")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Lsa_summary']
        risk_score = int(row['risk_deepseek'])
        stock_symbol = row.get('Stock_symbol', 'STOCK')  # 获取股票符号，如果没有则使用默认值
        
        if pd.isna(text) or text == '':
            continue
            
        prompt = create_prompt_template(text, risk_score, stock_symbol)
        texts.append(prompt)
        labels.append(risk_score)
    
    # 分割训练集和验证集 (80% 训练, 20% 验证)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(eval_texts)}")
    
    # 创建训练数据集
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    # 创建验证数据集
    eval_dataset = Dataset.from_dict({
        'text': eval_texts,
        'label': eval_labels
    })
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        # 对于语言模型，labels就是input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # 对训练集和验证集进行tokenization
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_tokenized, eval_tokenized

def create_model_and_tokenizer():
    """创建模型和分词器"""
    print("正在加载模型和分词器...")
    
    model_name = "./Qwen3-1.7B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Quant type
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时的数据类型
        bnb_4bit_use_double_quant=True,  # 使用双量化
    )

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./qwen_risk_model"):
    """训练模型"""
    print("开始训练模型...")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # report_to=None,  # 禁用wandb等报告工具
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到: {output_dir}")

def main():
    """主函数"""
    # 数据路径
    csv_path = "risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv"
    
    # 加载和预处理数据
    df = load_and_preprocess_data(csv_path)
    
    # 创建模型和分词器
    model, tokenizer = create_model_and_tokenizer()
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    
    # 训练模型
    train_model(model, tokenizer, train_dataset, eval_dataset)
    
    print("训练完成！")

if __name__ == "__main__":
    main() 