"""
Merge the pretrained with LORA weights for vllm inference
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/GRPO_finetuning/outputs/checkpoint-25", # Change the checkpoint path here
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

model.push_to_hub_merged("Heineken_qwen-3-8B_chatbot-v2", tokenizer, save_method = "merged_16bit", token="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")