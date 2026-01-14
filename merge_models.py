"""
Merge the pretrained with LORA weights for vllm inference
"""
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/GRPO_finetuning/PhuongNam_Heineken_qwen-3-8B_chatbot_grpo", # Change the checkpoint path here
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

model.save_pretrained_merged("NamModel", tokenizer, save_method = "merged_16bit")