
import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
from modeling_qwen_transformers import Qwen2MoeForCausalLM
import peft
from peft import LoraConfig, get_peft_model
from peft import TaskType
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import Dataset
from transformers import default_data_collator
import bitsandbytes as bnb
class PasskeyRetrievalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt, answer = self.data[index]
        return prompt, answer
    

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument('--num_tokens', type=int, default=1000000, help='number of tokens for the test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')

    args = parser.parse_args()
    return args



def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an execute line at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, accelerator, use_cache=False, n_garbage=60000, seed=666, segment_length=2048, num_train_epochs=3, train_batch_size=1, learning_rate=3e-4):
    # Generate training data
    train_data = []
    for _ in range(1000):
        prompt, answer = generate_prompt_landmark(n_garbage, seed)
        train_data.append({'text': prompt, 'labels': answer})

    train_dataset = Dataset.from_list(train_data)

    def tokenize_function(examples):
        # Tokenize the text and labels
        inputs = tokenizer(examples['text'], padding="max_length", truncation=True)
        inputs['labels'] = tokenizer(examples['labels'], padding="max_length", truncation=True)['input_ids']
        return inputs


    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    
    train_dataloader = DataLoader(tokenized_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=default_data_collator)

    # Prepare the model for LoRa training
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['beta',"q_proj", "v_proj", "k_proj", "o_proj"], # Include 'beta' in the target modules
    )
    model = get_peft_model(model, peft_config)

    # Prepare the optimizer and scheduler
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_train_epochs,)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    for epoch in range(num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):

            # Segment the input_ids into smaller chunks
            input_segments = torch.tensor_split(batch['input_ids'], list(range(segment_length, batch['input_ids'].shape[1], segment_length)))
            label_segments = torch.tensor_split(batch['labels'], list(range(segment_length, batch['labels'].shape[1], segment_length)))

            M_Z = None
            for i in range(len(input_segments)):
                outputs = model(input_ids=input_segments[i], labels=label_segments[i], M_Z=M_Z)
                M_Z = outputs.M_Z
                loss = outputs.loss
                accelerator.backward(loss)
                total_loss += loss.detach().float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss.item()}")

    # Evaluation
    model.eval()
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:].to(accelerator.device) # drop BOS

    # Segment the input_ids into smaller chunks
    input_segments = torch.tensor_split(input_ids[0], list(range(segment_length, input_ids.shape[1], segment_length)))

    M_Z = None
    for i in range(len(input_segments)-1):
        outputs = model(input_ids=input_segments[i].unsqueeze(0), M_Z=M_Z)
        M_Z = outputs.M_Z

    generation_output = model.generate(
        input_ids=input_segments[-1].unsqueeze(0), max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache, M_Z=M_Z
    )

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()
    # All tensors are on the CPU, so we can compare them directly
    answer_ids = answer_ids.cpu()
    is_correct = (model_answer == answer_ids[0]).all().item()
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct


def main(args):
    print("base model", args.base_model)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
    )
    config.use_cache=False
    config.num_experts_per_tok = 1
    config.max_position_embeddings = 2048

    # Load model and tokenizer
    model = Qwen2MoeForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        use_fast=False,
    )

    accelerator = Accelerator(mixed_precision='bf16')

    n_garbage = args.num_tokens
    is_correct = passkey_retrieval_test(model, tokenizer, accelerator, use_cache=False, n_garbage=n_garbage, seed=420, train_batch_size=args.batch_size)
    print(f"Accuracy: {'Passed' if is_correct else 'Failed'}")


if __name__ == "__main__":
    args = parse_config()
    main(args)