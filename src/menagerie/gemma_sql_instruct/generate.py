"""Evaluate the model."""

from random import randint

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from menagerie.gemma_sql_instruct.common import create_dataset

peft_model_id = "./merged"
model = AutoModelForCausalLM.from_pretrained(
    peft_model_id, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("./output")
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
assert pipe.tokenizer is not None

_, eval_dataset = create_dataset()

SAMPLES = 10

for sample_num in range(SAMPLES):
    rand_idx = randint(0, len(eval_dataset))

    prompt = pipe.tokenizer.apply_chat_template(
        eval_dataset[rand_idx]["messages"][:2],
        tokenize=False,
        add_generation_prompt=True,
    )
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.1,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
    )

    assert isinstance(outputs, list)

    print("=" * 80)
    print(f"Sample {sample_num + 1}/{SAMPLES}")
    print(f"Query:\n{eval_dataset[rand_idx]['messages'][0]['content']}")
    print(f"Query:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
    print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][2]['content']}")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
