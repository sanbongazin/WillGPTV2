#なぜかNoteBookなら動作するのでやってみて
# モデルの準備
model_name = "cyberagent/open-calm-1b"
peft_name = "lorappo-open-calm-1b"
output_dir = "lorappo-open-calm-1b-results"

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM
import torch,transformers

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name)

from datasets import load_dataset
dataset = load_dataset("sanbongazin/WilladgeArticle")

# dataset["train"] = dataset["train"].rename_column("instruction", "input_ids")
# dataset["train"] = dataset["train"].rename_columen("output", "completion")

# dataset["train"]["input_ids"] = dataset["train"]["input_ids"]
# dataset["train"]["completion"] = dataset["train"]["completion"]

data=dataset["train"]
formatted_data=[]

CUTOFF_LEN = 256  # コンテキスト長
# トークナイズ関数
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

# プロンプトテンプレートの準備
# def generate_prompt(data_point):
#     result = f"""### 指示:
#     {data_point["input"]}

# ### 回答:
#     {data_point["completion"]}
#     """
#     # 改行→<NL>
#     result = result.replace('\n', '<NL>')
#     return result

for i,j in zip(data["completion"],data["input"]):
    formatted_data.append(tokenize(i,tokenizer))


from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    torch_dtype=torch.float16,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# LoRAのパラメータ
lora_config = LoraConfig(
    inference_mode=False, 
    r=8, lora_alpha=32, 
    lora_dropout=0.1,
    target_modules="all-linear"
    )

# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)


# 学習可能パラメータの確認
model.print_trainable_parameters()



training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=6,
    num_train_epochs=3,
    logging_dir='./logs',
    remove_unused_columns=False
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=formatted_data,
    args = transformers.TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=900,
    #勾配と学習率を設定
    gradient_accumulation_steps = 2,
    learning_rate = 5e-5,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=5000,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    
),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
# LoRAモデルの保存
trainer.model.save_pretrained(peft_name,push_to_hub=True)
tokenizer.save_pretrained(peft_name,push_to_hub=True)
model.save_pretrained(peft_name,push_to_hub=True)

from huggingface_hub import notebook_login

notebook_login()
model.push_to_hub("sanbongazin/willgpt-neox-small_v2")