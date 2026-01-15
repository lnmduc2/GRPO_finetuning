from ChatBotSynthetic.training.sft.dataset_loader import DatasetLoader
from ChatBotSynthetic.training.sft.config_loader import ConfigLoader
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
import copy
from typing import List, Dict
from datetime import datetime, date
import json

def apply_chat_template(self, convo: List[Dict]):
    """
    Apply VLM ChatML template to a conversation (text-only, bypass image processor).

    Args:
        convo: List of message dictionaries with 'role' and 'content' keys.
                Supported roles: system, user, assistant, tool

    Returns:
        Formatted conversation string in ChatML format with <think> tags
    """
    # Convert any datetime objects to strings before applying template
    def convert_datetimes(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_datetimes(item) for item in obj]
        else:
            return obj

    # Deep copy and convert convo to avoid mutating original
    convo_clean: List[Dict] = convert_datetimes(convo)  # type: ignore

    # Strip <think> tags from non-last assistant messages
    # The template will add <think> only for the last assistant turn
    for i, msg in enumerate(convo_clean):
        if msg["role"] == "assistant" and i < len(convo_clean) - 1:
            # Strip think tags from content
            if "content" in msg and msg["content"]:
                msg["content"] = self._strip_think_tags(msg["content"])
            # Remove reasoning_content field for non-last turns
            if "reasoning_content" in msg:
                del msg["reasoning_content"]

    system_prompt = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."
    convo_clean.insert(0, {"role": "system", "content": system_prompt})
    
    # === BUILD FORMATTED TEXT MANUALLY (bypass VLM image processor) ===
    # Format theo VLM template: vlm_chat_template.jinja
    formatted_parts = []
    
    # Track tool responses để group chúng lại
    i = 0
    while i < len(convo_clean):
        msg = convo_clean[i]
        role = msg["role"]
        content = self._render_content(msg.get("content", ""))
        
        if role == "system":
            formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        
        elif role == "user":
            formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        
        elif role == "assistant":
            is_last = (i == len(convo_clean) - 1)
            reasoning = msg.get("reasoning_content", "")
            
            if is_last and reasoning:
                # Last assistant with thinking
                assistant_text = f"<|im_start|>assistant\n<think>\n{reasoning.strip()}\n</think>\n\n{content}"
            else:
                assistant_text = f"<|im_start|>assistant\n{content}"
            
            # Add tool calls if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    if content or tool_calls.index(tc) > 0:
                        assistant_text += "\n"
                    assistant_text += self._format_tool_call(tc)
            
            assistant_text += "<|im_end|>\n"
            formatted_parts.append(assistant_text)
        
        elif role == "tool":
            # Group consecutive tool responses
            tool_content = f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response>"
            # Check for more consecutive tool messages
            while i + 1 < len(convo_clean) and convo_clean[i + 1]["role"] == "tool":
                i += 1
                next_content = self._render_content(convo_clean[i].get("content", ""))
                tool_content += f"\n<tool_response>\n{next_content}\n</tool_response>"
            tool_content += "<|im_end|>\n"
            formatted_parts.append(tool_content)
        
        i += 1
    
    formatted_text = "".join(formatted_parts)
    
    # === Separate prompt and answer ===
    for msq in convo_clean:
        if msq["role"] == "tool":
            msq["content"] = json.dumps(msq["content"]) if not isinstance(msq["content"], str) else msq["content"]
        if "tool_calls" in msq:
            msq["tool_calls"] = [json.dumps(tool_call) if not isinstance(tool_call, str) else tool_call for tool_call in msq["tool_calls"]]
    
    prompt = convo_clean[:-1]
    answer = convo_clean[-1]
    formatted_anwser = f"<think>\n{answer.get('reasoning_content', '')}\n</think>\n\n"
    if answer.get('content'):
        formatted_anwser += answer['content']
    elif answer.get('tool_calls'):
        formatted_anwser += self._format_tool_call(answer['tool_calls'][0] if isinstance(answer['tool_calls'][0], dict) else json.loads(answer['tool_calls'][0]))

    if prompt[-1]["role"] == "tool":
        prompt[-1]["role"] = "user"
    return formatted_text, prompt, formatted_anwser

def _format_tool_call(self, tool_call: Dict) -> str:
    """Format a tool call to VLM template format."""
    if "function" in tool_call:
        tool_call = tool_call["function"]
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    if isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args, ensure_ascii=False)
    return f'<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>'

def _render_content(self, content) -> str:
    """Render content to string (VLM format - text only, no images)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # VLM multimodal format: list of {type: text/image, text: ...}
        # We only handle text items since we're doing text-only training
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(item["text"])
                elif item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    elif isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content) if content else ""

# Monkey patch dataset loader to apply VLM chat template instead of using the tokenizer in the config
DatasetLoader.apply_chat_template = apply_chat_template
DatasetLoader._render_content = _render_content
DatasetLoader._format_tool_call = _format_tool_call

# configs for training
max_seq_length = 4096 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
config = ConfigLoader.load_config("ChatBotSynthetic/configs/sft.yaml")
config = {
    **config,
    "dataset": {
        "train_path": "ChatBotSynthetic/data/train.jsonl",
        "validation_path": "ChatBotSynthetic/data/validation.jsonl",
        "format": "json",  # Note: use "json" not "jsonl" for HF datasets
        "message_field": "messages",
        "text_field": "text"
    }
}
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-VL-4B-Thinking",
    max_seq_length = max_seq_length,
)

# ConfigLoader tự động đọc file template và lưu vào key "chat_template"
tokenizer.chat_template = config["chat_template"]

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
    # Chỉ áp dụng LoRA vào N layer cuối để model ko bị forget
    layers_to_transform = list(range(24, 32)),
)

loader = DatasetLoader(config, tokenizer)
train_dataset = loader.prepare_dataset(split="train")
dev_dataset = loader.prepare_dataset(split="validation")



def preprocess_labels(example):
    # 1. Render text đầy đủ
    text = example['text']
    
    separator = "<|im_start|>assistant\n"
    start_idx = text.rfind(separator)
    
    if start_idx == -1:
        # Không tìm thấy -> Mask toàn bộ (ignore)
        tokenized = tokenizer(images=None, text=text, videos=None, truncation=True, max_length=max_seq_length, padding=False, add_special_tokens=False)
        # VLM tokenizer trả về batched format [[...]], cần lấy [0]
        input_ids = tokenized["input_ids"][0] if isinstance(tokenized["input_ids"][0], list) else tokenized["input_ids"]
        labels = [-100] * len(input_ids)
    else:
        # Tách text thành 2 phần: [Prompt] và [Response]
        prompt_text = text[:start_idx + len(separator)]
        response_text = text[start_idx + len(separator):]
        
        # VLM tokenizer trả về batched format [[...]], cần lấy [0]
        prompt_tokenized = tokenizer(images=None, text=prompt_text, videos=None, add_special_tokens=False)["input_ids"]
        response_tokenized = tokenizer(images=None, text=response_text, videos=None, add_special_tokens=False)["input_ids"]
        
        prompt_ids = prompt_tokenized[0] if isinstance(prompt_tokenized[0], list) else prompt_tokenized
        response_ids = response_tokenized[0] if isinstance(response_tokenized[0], list) else response_tokenized
        
        # Gộp lại: labels của phần prompt là -100
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        
        # Cắt nếu quá dài
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1]*len(input_ids)}

# Áp dụng
train_dataset = train_dataset.map(preprocess_labels) 
dev_dataset = dev_dataset.map(preprocess_labels)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = dev_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 4,   # 3090 24GB + 4B VLM + seq_len 4096
        gradient_accumulation_steps = 2,  # effective batch = 32
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        # max_steps = 10,
        learning_rate = 2e-4,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use TrackIO/WandB etc
        eval_strategy="steps",       # Hoặc "epoch". Kích hoạt đánh giá trên dev_dataset
        eval_steps=50,                
        load_best_model_at_end=True,
    ),
)

import torch
torch.cuda.empty_cache()
import gc
gc.collect()

print("CUDA cache cleared. VRAM should now be refreshed.")
trainer_stats = trainer.train()