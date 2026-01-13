import re
import json
import inspect
from typing import get_type_hints, Callable

def get_all_tools_info(available_tools: dict) -> str:
    """
    Lấy thông tin của tất cả tools để làm context cho LLM.
    
    Args:
        available_tools: Dict mapping tool name -> function
        
    Returns:
        str: String chứa thông tin của tất cả tools
    """
    def get_tool_info(tool_func: Callable) -> str:
        # Lấy tên function
        tool_name = tool_func.__name__
        
        # Lấy docstring
        docstring = inspect.getdoc(tool_func) or "Không có mô tả."
        
        # Lấy signature và type hints
        sig = inspect.signature(tool_func)
        try:
            type_hints = get_type_hints(tool_func)
        except Exception:
            type_hints = {}
        
        # Xây dựng input schema
        input_params = []
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, "Any")
            if hasattr(param_type, "__name__"):
                param_type = param_type.__name__
            elif hasattr(param_type, "_name"):  # For typing generics like Optional
                param_type = str(param_type)
            
            # Kiểm tra default value
            if param.default is inspect.Parameter.empty:
                default_str = "(required)"
            elif param.default is None:
                default_str = "(optional, default=None)"
            else:
                default_str = f"(optional, default={repr(param.default)})"
            
            input_params.append(f"    - {param_name}: {param_type} {default_str}")
        
        input_schema = "\n".join(input_params) if input_params else "    Không có tham số"
        
        # Lấy output schema (return type)
        return_type = type_hints.get("return", "Any")
        if hasattr(return_type, "__name__"):
            output_schema = return_type.__name__
        elif hasattr(return_type, "_name"):
            output_schema = str(return_type)
        else:
            output_schema = str(return_type)
        
        # Format output string
        tool_info = f"""Tool: {tool_name}
    Mô tả: {docstring}
    Input Schema:
    {input_schema}
    Output Schema: {output_schema}"""
        
        return tool_info


    tools_info = []
    for _, tool_func in available_tools.items():
        tools_info.append(get_tool_info(tool_func))
    
    return "\n\n---\n\n".join(tools_info)


# Hàm xử lý clean <think> và tokenize
def format_conversation_template(messages):
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

def process_and_tokenize(tokenizer, messages):
    formatted_input_str = format_conversation_template(messages)
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

def execute_tool_call(tool_call: dict, available_tools: dict) -> str:
    """
    Thực thi một tool call và trả về kết quả dạng string.
    """
    tool_name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    
    if tool_name not in available_tools:
        return json.dumps({"error": f"Tool '{tool_name}' không tồn tại"}, ensure_ascii=False)
    
    try:
        result = available_tools[tool_name](**arguments)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)