import argparse
import os
import time

import torch

from tqdm import tqdm, trange
import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftConfig, prepare_model_for_kbit_training, get_peft_model, PeftModelForCausalLM


def load_dpo_dataset(data_dir: str, max_sample: int = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    with open(os.path.join(data_dir, f"test.json"), 'r+') as f:
        data = json.load(f)
    list_data = [{
        "prompt": item["prompt"],
        "chosen": item["chosen"],
        "rejected": item["rejected"],
    } for item in data]
    dataset = Dataset.from_list(list_data)
    '''
    if os.path.exists('data_cache') is False:
        os.makedirs('data_cache')
    dataset = dataset.map(
        load_from_cache_file=True,
        cache_file_name='data_cache/grouped_train_data.arrow',
    )
    '''
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-path', type=str, required=True)
    parser.add_argument('--generation-len', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeat-n', type=int, default=1)
    parser.add_argument('--log-dir', type=str, default='../log')
    args = parser.parse_args()

    peft_config = PeftConfig.from_pretrained(args.adapter_path)

    """Configure bitsandbytes for 4-bit quantization"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    """Model instantiation"""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        device_map='auto'
    )

    """Model instantiation"""
    model = prepare_model_for_kbit_training(model)
    model = PeftModelForCausalLM.from_pretrained(model, args.adapter_path)
    model.eval()
    # model = AutoPeftModelForCausalLM.from_pretrained(args.adapter_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f'[!] init model over ...')

    """Prepare dataset"""
    dataset = load_dpo_dataset('../data')
    print(f'[!] load dpo dataset over ...')

    """
    Inference Loop
    """
    log_file = open(
        file=os.path.join(args.log_dir, f"{os.path.basename(peft_config.base_model_name_or_path)}-{str(time.struct_time)}.jsonl"),
        mode='w+',
        encoding='utf-8'
    )
    loss_list = []
    token_list = []
    with torch.no_grad():
        for i in trange(0, len(dataset), args.batch_size):
            input_ids = tokenizer(
                [sample for sample in dataset[i: i + args.batch_size]['prompt']],
                padding=True,
                truncation=True,
                max_length=1792,
                return_tensors='pt'
            )['input_ids'].cuda()
            batch_size = input_ids.size(0)
            len_prompt = input_ids.size(-1)

            """Text generation"""
            outputs = model.generate(
                input_ids=input_ids,
                max_length=min(int(len_prompt + args.generation_len * 1.2), 2048),
                min_length=min(int(len_prompt + args.generation_len * 1.1), 2048),
                early_stopping=False,
                num_return_sequences=args.repeat_n,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None,
                do_sample=True,
                top_p=0.92,
                top_k=100
            )[:, len_prompt: len_prompt + args.generation_len].tolist()

            generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j in range(batch_size):
                for k in range(args.repeat_n):
                    tqdm.write(f"Sample {i * args.batch_size + j + 1} / {len(dataset)} ({k + 1}): {generation[j * args.repeat_n + k]}")

            token_list.extend(outputs)

            """Calculate perplexity"""
            model_output = model(input_ids)
            logits = model_output.logits
            loss = torch.nn.functional.cross_entropy(
                input=logits[:, :-1].reshape(-1, logits.size()[-1]),
                target=input_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='none'
            ).cpu().tolist()
            loss_list.extend(loss)

            """Logging"""
            log_file.write(
                json.dumps(
                    [
                        {
                            "prompt": tokenizer.decode(input_ids[j].cpu(), skip_special_tokens=True),
                            "response": [
                                {
                                    "text": text,
                                    "token": token,
                                    "num_tokens": len(token)
                                } for token, text in zip(outputs[j * args.repeat_n: (j + 1) * args.repeat_n],
                                                         generation[j * args.repeat_n: (j + 1) * args.repeat_n])
                            ],
                            "loss": loss[j * args.repeat_n: (j + 1) * args.repeat_n]
                        } for j in range(batch_size)
                    ]
                )
            )

        log_file.close()
