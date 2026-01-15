"""
Merge the pretrained with LORA weights for vllm inference
"""
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/GRPO_finetuning/outputs/checkpoint-416", # Change the checkpoint path here
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = False,
)

model.save_pretrained_merged("Qwen3_4B_VL_thinking_Heineken", tokenizer, save_method = "merged_16bit")