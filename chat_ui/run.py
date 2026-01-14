import sys
import os
import multiprocessing
from pathlib import Path
import asyncio

# QUAN TRỌNG: Set spawn method TRƯỚC khi import bất cứ thứ gì liên quan đến CUDA
# Phải nằm ngoài mọi if/else để chạy ngay khi import module
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

from nicegui import ui, app
from vllm import LLM, SamplingParams
from ChatBotSynthetic.synthetic_pipeline.mock_tools import TOOL_REGISTRY
from utils import get_all_tools_info, format_conversation_template, parse_tool_calls, execute_tool_call


# =============================================================================
# LƯU MODEL VÀO app OBJECT - persist qua re-execution vì app được cache trong sys.modules
# =============================================================================
def get_storage():
    """Lấy storage từ app object (persist qua runpy re-execution)."""
    if not hasattr(app, '_model_storage'):
        app._model_storage = {
            'llm': None,
            'sampling_params': None,
            'messages': [],
            'initialized': False
        }
    return app._model_storage


def is_initialized():
    """Check xem model đã được init chưa."""
    storage = get_storage()
    return storage['initialized'] and storage['llm'] is not None


def init():
    """Khởi tạo model và các tham số."""
    storage = get_storage()
    
    # Check nếu đã init rồi thì skip
    if is_initialized():
        print("⏭️ Model đã được khởi tạo, sử dụng instance có sẵn...")
        return
    
    print("🚀 Đang khởi tạo model...")
    
    storage['llm'] = LLM(
        model=str(Path(__file__).parent.parent / "NamModel"),
        tokenizer=str(Path(__file__).parent.parent / "NamModel"),
        max_model_len=4096,
        dtype="float16",
        quantization="bitsandbytes", 
    )
    
    storage['sampling_params'] = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=1024
    )
    
    # System prompt
    storage['messages'] = [
        {"role": "system", "content": f"""
Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn.

Bạn được quyền access vào các tool có sẵn sau để tra cứu thông tin khách hàng:
{get_all_tools_info(TOOL_REGISTRY)}
"""}
    ]
    
    storage['initialized'] = True
    print("✅ Model đã sẵn sàng!")


def generate_response(user_message: str) -> list:
    """Generate response từ LLM và xử lý tool calls."""
    storage = get_storage()
    
    if not is_initialized():
        raise RuntimeError("Model chưa được khởi tạo!")
    
    responses = []
    
    # Thêm tin nhắn user
    storage['messages'].append({"role": "user", "content": user_message})
    current_conversation = format_conversation_template(storage['messages'])
    
    # Generate response
    outputs = storage['llm'].generate(current_conversation, storage['sampling_params'])
    response = outputs[0].outputs[0].text
    responses.append(("assistant", response))
    
    # Parse và xử lý tool calls
    tool_calls = parse_tool_calls(response)
    
    while tool_calls:
        all_results = []
        tool_info = []
        
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            params = tc.get("arguments", {})
            result = execute_tool_call(tc, TOOL_REGISTRY)
            all_results.append(result)
            tool_info.append(f"🔧 Tool: {tool_name}\n📥 Params: {params}\n📤 Result: {result[:300]}{'...' if len(result) > 300 else ''}")
        
        responses.append(("tool", "\n---\n".join(tool_info)))
        
        combined_result = "\n---\n".join(all_results)
        storage['messages'].append({"role": "tool", "content": combined_result})
        
        current_conversation = format_conversation_template(storage['messages'])
        outputs = storage['llm'].generate(current_conversation, storage['sampling_params'])
        response = outputs[0].outputs[0].text
        responses.append(("assistant", response))
        
        tool_calls = parse_tool_calls(response)
    
    storage['messages'].append({"role": "assistant", "content": response})
    return responses


# =============================================================================
# UI PAGE - Định nghĩa bằng decorator để NiceGUI quản lý đúng cách
# =============================================================================
@ui.page('/')
def main_page():
    """Trang chính - Chat UI."""
    
    # Custom CSS - Heineken Style (Green + White)
    ui.add_head_html('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;500;600;700&display=swap');
        
        :root {
            --heineken-green: #00843D;
            --heineken-light-green: #4CAF50;
            --heineken-dark-green: #006830;
            --heineken-accent: #8BC34A;
        }
        
        body { 
            font-family: 'Quicksand', sans-serif !important; 
            background: #f8faf8;
        }
        
        .chat-container {
            background: linear-gradient(180deg, #ffffff 0%, #e8f5e9 50%, #c8e6c9 100%);
            min-height: 100vh;
        }
        
        .message-user {
            background: linear-gradient(135deg, var(--heineken-green) 0%, var(--heineken-light-green) 100%);
            color: white;
            border-radius: 24px 24px 6px 24px;
            padding: 14px 20px;
            max-width: 70%;
            margin-left: auto;
            box-shadow: 0 4px 20px rgba(0, 132, 61, 0.25);
            font-weight: 500;
        }
        
        .message-assistant {
            background: #ffffff;
            color: #2d3436;
            border-radius: 24px 24px 24px 6px;
            padding: 14px 20px;
            max-width: 70%;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid #e0e0e0;
            font-weight: 500;
        }
        
        .message-tool {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            color: var(--heineken-dark-green);
            border-radius: 16px;
            padding: 14px 20px;
            max-width: 85%;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85em;
            box-shadow: 0 4px 15px rgba(0, 132, 61, 0.15);
            border: 1px solid var(--heineken-accent);
        }
        
        .chat-header {
            background: linear-gradient(135deg, var(--heineken-green) 0%, var(--heineken-dark-green) 100%);
            padding: 24px;
            border-radius: 0 0 40px 40px;
            box-shadow: 0 8px 32px rgba(0, 132, 61, 0.3);
        }
        
        .input-area {
            background: #ffffff;
            border-radius: 30px;
            padding: 10px 16px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1);
            border: 2px solid #e8f5e9;
        }
        
        .input-area:focus-within {
            border-color: var(--heineken-green);
            box-shadow: 0 4px 24px rgba(0, 132, 61, 0.2);
        }
        
        .send-btn {
            background: linear-gradient(135deg, var(--heineken-green) 0%, var(--heineken-light-green) 100%) !important;
            border-radius: 50% !important;
            width: 52px !important;
            height: 52px !important;
            transition: all 0.3s ease !important;
        }
        
        .send-btn:hover {
            transform: scale(1.1) rotate(15deg);
            box-shadow: 0 6px 24px rgba(0, 132, 61, 0.4);
        }
        
        /* Star decoration */
        .chat-header::before {
            content: "★";
            position: absolute;
            font-size: 120px;
            color: rgba(255, 255, 255, 0.08);
            right: 20px;
            top: -20px;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #e8f5e9;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--heineken-green);
            border-radius: 4px;
        }
    </style>
    ''')
    
    with ui.column().classes('w-full min-h-screen chat-container'):
        # Header
        with ui.row().classes('w-full chat-header items-center justify-center relative overflow-hidden'):
            ui.label('★').classes('absolute text-9xl text-white/10 -right-4 -top-8')
            ui.image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Heineken_logo.svg/1200px-Heineken_logo.svg.png').classes('w-36 drop-shadow-lg')
            with ui.column().classes('ml-6'):
                ui.label('Heineken Vietnam').classes('text-3xl font-bold text-white tracking-wide')
                ui.label('Hệ thống Chăm Sóc Khách Hàng AI').classes('text-sm text-white/90 font-medium')
        
        # Chat area
        chat_container = ui.scroll_area().classes('flex-grow w-full max-w-4xl mx-auto p-4')
        
        with chat_container:
            messages_column = ui.column().classes('w-full gap-3')
        
        # Input area
        with ui.row().classes('w-full max-w-4xl mx-auto p-4 input-area mb-6 items-center'):
            user_input = ui.input(placeholder='Nhập tin nhắn...').classes(
                'flex-grow bg-transparent border-none text-gray-800'
            ).props('borderless dense')
            
            spinner = ui.spinner('dots', size='lg', color='green').classes('hidden')
            send_button = ui.button(icon='send').classes('send-btn text-white')
        
        async def send_message():
            message = user_input.value.strip()
            if not message:
                return
            
            user_input.value = ''
            
            with messages_column:
                with ui.row().classes('w-full justify-end'):
                    ui.label(message).classes('message-user')
            
            chat_container.scroll_to(percent=1.0)
            spinner.classes(remove='hidden')
            send_button.disable()
            
            try:
                responses = await asyncio.to_thread(generate_response, message)
                
                with messages_column:
                    for role, content in responses:
                        if role == "assistant":
                            with ui.row().classes('w-full justify-start'):
                                ui.label(content).classes('message-assistant')
                        elif role == "tool":
                            with ui.row().classes('w-full justify-center'):
                                ui.label(content).classes('message-tool whitespace-pre-wrap')
                
                chat_container.scroll_to(percent=1.0)
                
            except Exception as e:
                with messages_column:
                    with ui.row().classes('w-full justify-center'):
                        ui.label(f"❌ Lỗi: {str(e)}").classes('text-red-400')
            
            finally:
                spinner.classes(add='hidden')
                send_button.enable()
        
        send_button.on('click', send_message)
        user_input.on('keydown.enter', send_message)
        
        # Welcome message
        with messages_column:
            with ui.row().classes('w-full justify-start'):
                ui.label('Xin chào! Tôi là trợ lý CSKH của Heineken Vietnam. Tôi có thể giúp gì cho anh/chị?').classes('message-assistant')


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Init model TRƯỚC khi start UI server
    init()
    
    # Chạy NiceGUI server
    ui.run(
        title='Heineken CSKH Chatbot',
        host='0.0.0.0',
        port=8080,
        reload=False,  # QUAN TRỌNG: Tắt reload để không re-execute script
        favicon='🍺'
    )
