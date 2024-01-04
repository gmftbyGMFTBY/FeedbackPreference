import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from trl import DPOTrainer
import yaml
import ipdb
import json
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="kaist-ai/prometheus-7b-v1.0",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_8bit", metadata={"help": "the optimizer type"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    max_prompt_length: Optional[int] = field(default=2048, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=3010, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=1500, metadata={"help": "the saving frequency"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. "
            )
        },
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
                    '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
                    'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
                    "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


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
    with open(os.path.join(data_dir, f"train.json"), 'r+') as f:
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
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    output_dir = os.path.join(script_args.output_dir, os.path.basename(script_args.model_name_or_path))
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        do_eval=False,
        output_dir=output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_feedback"
    )
    print(f'[!] training args:', training_args)

    """Configure bitsandbytes for 4-bit quantization"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    """Model instantiation"""
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name_or_path,
        quantization_config=bnb_config,
        # device_map='auto'
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name_or_path,
        quantization_config=bnb_config,
        # device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f'[!] init model over ...')

    """Prepare for k-bit training"""
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    """Prepare dataset"""
    dataset = load_dpo_dataset('../data')
    print(f'[!] load dpo dataset over ...')
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        beta=0.1,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length
    )
    model.config.use_cache = False
    model_ref.config.use_cache = False
    dpo_trainer.train()
    # dpo_trainer.save_model(script_args.output_dir)
    # save
    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
