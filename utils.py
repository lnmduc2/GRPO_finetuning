import re
import json

# Hàm xử lý clean <think> và tokenize
def process_and_tokenize(tokenizer, messages):
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
    return tokenizer([formatted_input_str], return_tensors = "pt").to("cuda")["input_ids"]

def parse_tool_calls(response: str) -> list:
    """
    Parse tất cả tool calls từ response của assistant.
    Mỗi tool call nằm trong cặp <tool_call> và </tool_call>
    Returns:
        list: Danh sách các tool call đã parse được
    """
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, response, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            tool_calls.append(tool_call)
        except json.JSONDecodeError as e:
            print(f"[Warning] Không parse được tool call: {match[:50]}... Error: {e}")
    
    return tool_calls

def execute_tool_call(tool_call: dict, AVAILABLE_TOOLS: dict) -> str:
    """
    Thực thi một tool call và trả về kết quả dạng string.
    """
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    if tool_name not in AVAILABLE_TOOLS:
        return json.dumps({"error": f"Tool '{tool_name}' không tồn tại"}, ensure_ascii=False)
    
    try:
        result = AVAILABLE_TOOLS[tool_name](**arguments)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)