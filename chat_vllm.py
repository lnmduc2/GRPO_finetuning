from vllm import LLM, SamplingParams
from ChatBotSynthetic.synthetic_pipeline.mock_tools import TOOL_REGISTRY, tra_cuu_thong_tin
from utils import get_all_tools_info, format_conversation_template, parse_tool_calls, execute_tool_call

if __name__ == '__main__':
    llm = LLM(
        model = "Heineken_qwen-3-8B_chatbot-v2",
        tokenizer = "Heineken_qwen-3-8B_chatbot-v2",
        max_model_len = 4096,
        dtype = "float16",
        cpu_offload_gb = 8 # Offload bớt ra CPU vì 5060 Ti chỉ có 16GiB
    )
    sampling_params = SamplingParams(
        temperature=0.6, 
        top_p=0.95, 
        top_k=20,
        max_tokens=1024  # Tăng max_tokens để model có thể sinh ra response dài hơn
    )

    # Các tools có sẵn
    AVAILABLE_TOOLS = {
        "tra_cuu_thong_tin": tra_cuu_thong_tin,
    }

    messages = [
    {"role": "system", "content": f"""
Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn.

Bạn được quyền access vào các tool có sẵn sau để tra cứu thông tin khách hàng:
{get_all_tools_info(AVAILABLE_TOOLS)}
"""}, # Phải có system prompt ở đầu theo như training format
    ]

    while True:
        # 3. Nhận input từ người dùng
        user_input = input("\n[User]: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Cảm ơn anh đã sử dụng hệ thống!")
            break
        
        # Thêm tin nhắn của user vào lịch sử
        messages.append({"role": "user", "content": user_input})
        current_conversation = format_conversation_template(messages)
        
        # Chạy LLM
        outputs = llm.generate(current_conversation, sampling_params)

        response = outputs[0].outputs[0].text
        print(f"[Assistant]: {response}")

        # Parse tool calls
        tool_calls = parse_tool_calls(response)
        while tool_calls:
            print(f"\n[System] Phát hiện {len(tool_calls)} tool call(s), đang thực thi...")
            # Thực thi từng tool call và thu thập kết quả
            all_results = []
            for i, tc in enumerate(tool_calls):
                tool_name = tc.get("name", "unknown")
                print(f"  -> Gọi tool '{tool_name}' với params: {tc.get('arguments', {})}")
                result = execute_tool_call(tc, AVAILABLE_TOOLS)
                print(f"  <- Kết quả: {result[:200]}{'...' if len(result) > 200 else ''}")

            # Thêm kết quả tool vào messages (gộp tất cả kết quả nếu có nhiều tool calls)
            combined_result = "\n---\n".join(all_results)
            messages.append({"role": "tool", "content": combined_result})

            # Generate tiếp để model xử lý kết quả tool
            current_conversation = format_conversation_template(messages)
            outputs = llm.generate(current_conversation, sampling_params)
            response = outputs[0].outputs[0].text
            print(f"[Assistant]: {response}")

            # Lưu response mới
            messages.append({"role": "assistant", "content": response})
            
            # Kiểm tra xem response mới có tool calls không (để tiếp tục loop nếu cần)
            tool_calls = parse_tool_calls(response)
