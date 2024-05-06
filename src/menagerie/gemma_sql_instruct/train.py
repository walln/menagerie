"""Finetune gemma-2b."""

import torch
from peft.tuners.lora import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, setup_chat_format

import wandb
from menagerie.gemma_sql_instruct.common import create_dataset

train_dataset, _ = create_dataset(12500, 2500)

wandb.init(project="gemma-2b-finetune")

model_id = "google/gemma-2b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # flash attn install broken with build isolation
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# Ensure the model can handle chatml format
model, tokenizer = setup_chat_format(model, tokenizer)  # type: ignore PreTrainedTokenizerFast also valid


peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # "Deepspeed needs False and it will be default soon anyways."
    optim="adamw_torch_fused",  # try "paged_adamw_8bit"?
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="wandb",
)

# max_seq_length = 2048
max_seq_length = 1024  # cant handle 2048 on local

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    packing=True,
    dataset_kwargs={"add_special_tokens": False, "append_concat_token": False},
)

trainer.train()  # type: ignore Broken types in trl

trainer.save_model()

wandb.finish()
