SYSTEM_PROMPT = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."

from unsloth import FastLanguageModel
from transformers import TextStreamer


# 1. Cấu hình
adapter_path = "/root/GRPO_finetuning/outputs_new/checkpoint-25" # Change the checkpoint path here
max_seq_length = 4096
dtype = None # Để None cho tự động (T4 sẽ là float16)
load_in_4bit = True

# 2. Load Model 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Bật chế độ Inference (Rất quan trọng để fix lỗi Attribute và tăng tốc)
FastLanguageModel.for_inference(model)

# Hàm xử lý clean <think> và tokenize
def process_and_tokenize(messages):
    def format_multi_turn_to_tool_calling(messages):
        full_string = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == "system":
                full_string += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                full_string += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ["tool", "observation"]:
                full_string += f"<|im_start|>tool\n{content}<|im_end|>\n"
            elif role == "assistant":
                full_string += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                
        # Kết thúc bằng việc "mồi" cho model chuẩn bị suy nghĩ và trả lời
        full_string += "<|im_start|>assistant\n" 
        
        return full_string

    formatted_input_str = format_multi_turn_to_tool_calling(messages)
    return tokenizer([formatted_input_str], return_tensors = "pt").to(model.device)["input_ids"]


messages = [
{"role": "system", "content": f"""
{SYSTEM_PROMPT}

Bạn được quyền access vào tool find_info để tra cứu thông tin khách hàng.
Tool này có params phone_number (str), outlet_id (str), outlet_name (str).
Tùy vào dữ liệu mà khách hàng cung cấp, hãy gọi tool với param phù hợp.
"""}, # Phải có system prompt ở đầu theo như training format
]

print("--- Hệ thống CSKH Heineken đã sẵn sàng (Gõ 'exit' hoặc 'quit' để dừng) ---")

while True:
    # 3. Nhận input từ người dùng
    user_input = input("\n[User]: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Cảm ơn anh đã sử dụng hệ thống!")
        break
    
    # Thêm tin nhắn của user vào lịch sử
    messages.append({"role": "user", "content": user_input})

    # 4. Chuẩn bị đầu vào cho model
    inputs = process_and_tokenize(messages)

    # 5. Cấu hình Streamer để hiển thị text ngay khi đang generate
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)

    print("[Assistant]: ", end="")
    
    # 6. Generate phản hồi
    outputs = model.generate(
        input_ids = inputs,
        streamer = text_streamer,
        max_new_tokens = 4096,
        use_cache = True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    # 7. Lưu phản hồi của AI vào lịch sử để context được tiếp nối
    # Ta tách phần token mới generate được từ outputs
    new_tokens = outputs[0][inputs.shape[-1]:]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    messages.append({"role": "assistant", "content": assistant_response})

    
        