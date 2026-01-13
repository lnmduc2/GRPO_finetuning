SYSTEM_PROMPT = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."

from unsloth import FastLanguageModel
from transformers import TextStreamer
import re
import json
from utils import process_and_tokenize, parse_tool_calls, execute_tool_call

# ========== MOCK TOOL ==========
def find_info(phone_number: str = None, outlet_id: str = None, outlet_name: str = None) -> dict:
    """
    Mock tool để tra cứu thông tin khách hàng.
    Params:
        phone_number (str): Số điện thoại khách hàng
        outlet_id (str): Mã cửa hàng (Outlet ID)
        outlet_name (str): Tên cửa hàng
    Returns:
        dict: Thông tin khách hàng (mock data)
    """
    # Mock data - thay thế bằng logic thực tế khi cần
    mock_db = {
        "61505571": {
            "outlet_id": "61505571",
            "outlet_name": "Tạp hóa Minh Tâm",
            "phone_number": "0936145585",
            "address": "123 Nguyễn Văn Cừ, P.4, Q.5, TP.HCM",
            "status": "active",
            "last_login": "2026-01-10",
        },
        "0936145585": {
            "outlet_id": "61505571",
            "outlet_name": "Tạp hóa Minh Tâm", 
            "phone_number": "0936145585",
            "address": "123 Nguyễn Văn Cừ, P.4, Q.5, TP.HCM",
            "status": "active",
            "last_login": "2026-01-10",
        }
    }
    
    # Tìm theo outlet_id
    if outlet_id and outlet_id in mock_db:
        return {"success": True, "data": mock_db[outlet_id]}
    
    # Tìm theo phone_number
    if phone_number and phone_number in mock_db:
        return {"success": True, "data": mock_db[phone_number]}
    
    # Tìm theo outlet_name (mock - không tìm thấy)
    if outlet_name:
        return {"success": False, "message": f"Không tìm thấy cửa hàng với tên '{outlet_name}'"}
    
    return {"success": False, "message": "Không tìm thấy thông tin. Vui lòng kiểm tra lại dữ liệu."}


# Registry các tools có sẵn
AVAILABLE_TOOLS = {
    "find_info": find_info,
}




# ========== CONFIG ==========
# 1. Cấu hình
adapter_path = "/root/GRPO_finetuning/outputs/checkpoint-25" # Change the checkpoint path here
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
    inputs = process_and_tokenize(tokenizer, messages)

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

    # 8. Parse và thực thi tool calls nếu có
    tool_calls = parse_tool_calls(assistant_response)
    
    while tool_calls:
        print(f"\n[System] Phát hiện {len(tool_calls)} tool call(s), đang thực thi...")
        
        # Thực thi từng tool call và thu thập kết quả
        all_results = []
        for i, tc in enumerate(tool_calls):
            tool_name = tc.get("name", "unknown")
            print(f"  -> Gọi tool '{tool_name}' với params: {tc.get('arguments', {})}")
            result = execute_tool_call(tc, AVAILABLE_TOOLS)
            all_results.append(result)
            print(f"  <- Kết quả: {result[:200]}{'...' if len(result) > 200 else ''}")
        
        # Thêm kết quả tool vào messages (gộp tất cả kết quả nếu có nhiều tool calls)
        combined_result = "\n---\n".join(all_results)
        messages.append({"role": "tool", "content": combined_result})
        
        # Generate tiếp để model xử lý kết quả tool
        inputs = process_and_tokenize(tokenizer, messages)
        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        print("\n[Assistant]: ", end="")
        outputs = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=4096,
            use_cache=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )
        
        # Lưu response mới
        new_tokens = outputs[0][inputs.shape[-1]:]
        assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": assistant_response})
        
        # Kiểm tra xem response mới có tool calls không (để tiếp tục loop nếu cần)
        tool_calls = parse_tool_calls(assistant_response)
