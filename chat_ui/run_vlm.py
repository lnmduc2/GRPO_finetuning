import sys
import os
import multiprocessing
from pathlib import Path
import asyncio
import base64
import io

import traceback # Thêm dòng này vào đầu file cùng các import khác

# Hàm mới: thêm vào bên cạnh các hàm helper khác
def make_thumbnail(data_url: str, size=(200, 200)) -> str:
    """Tạo thumbnail nhỏ từ base64 gốc để hiển thị UI không bị lag/crash."""
    try:
        if not data_url: return ""
        img = data_url_to_image(data_url)
        img.thumbnail(size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
    except Exception:
        # Nếu lỗi resize thì trả về luôn ảnh gốc (fallback)
        return data_url

# QUAN TRỌNG: Set spawn method TRƯỚC khi import bất cứ thứ gì liên quan đến CUDA
# Phải nằm ngoài mọi if/else để chạy ngay khi import module
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

import re
from nicegui import ui, app
from vllm import LLM, SamplingParams
from ChatBotSynthetic.synthetic_pipeline.mock_tools import MockTools
from utils import get_all_tools_info, parse_tool_calls, execute_tool_call
from PIL import Image

MAX_PASTED_IMAGES = 4
IMAGE_PLACEHOLDER = "<image>"

TOOL_REGISTRY = {
    "tra_cuu_thong_tin": MockTools.tra_cuu_thong_tin,
    "kiem_tra_mqh": MockTools.kiem_tra_mqh,
    "kiem_tra_don_hang": MockTools.kiem_tra_don_hang, 
    "tao_ticket": MockTools.tao_ticket,
    "gui_huong_dan": MockTools.gui_huong_dan,
}

def parse_think_tags(text: str) -> tuple[str | None, str]:
    """Parse <think>...</think> tags from response.
    Returns: (thinking_content, main_response)
    """
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        thinking = match.group(1).strip()
        # Remove the think tags and get the main response
        main_response = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        return thinking, main_response
    
    return None, text.strip()


# =============================================================================
# LƯU MODEL VÀO app OBJECT - persist qua re-execution vì app được cache trong sys.modules
# Messages sẽ được tạo mới mỗi lần load page (F5 = reset conversation)
# =============================================================================
def get_model_storage():
    """Lấy model storage từ app object (persist qua runpy re-execution)."""
    if not hasattr(app, '_model_storage'):
        app._model_storage = {
            'llm': None,
            'sampling_params': None,
            'initialized': False
        }
    return app._model_storage


def is_initialized():
    """Check xem model đã được init chưa."""
    storage = get_model_storage()
    return storage['initialized'] and storage['llm'] is not None


def get_system_prompt():
    """Tạo system prompt với tool info."""
    return f"""Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn.

Bạn được quyền access vào các tool có sẵn sau để tra cứu thông tin khách hàng:
{get_all_tools_info(TOOL_REGISTRY)}"""


def create_new_conversation():
    """Tạo conversation mới với system prompt."""
    return [{"role": "system", "content": get_system_prompt()}]


def build_user_content_items(user_text: str, image_count: int) -> list:
    """Build user content list for Qwen3-VL template."""
    items = []
    if image_count > 0:
        items.extend([{"type": "image"} for _ in range(image_count)])
    if user_text and user_text.strip():
        items.append({"type": "text", "text": user_text.strip()})
    return items


def data_url_to_image(data_url: str) -> Image.Image:
    """Convert data URL to PIL image."""
    header, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def render_content_qwen3_vl(content) -> str:
    """Render content with Qwen3-VL vision tokens."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "image" or "image" in item or "image_url" in item:
                parts.append("<|vision_start|><|image_pad|><|vision_end|>")
            elif item_type == "video" or "video" in item:
                parts.append("<|vision_start|><|video_pad|><|vision_end|>")
            elif "text" in item:
                parts.append(item["text"])
    return "".join(parts)


def format_conversation_template_qwen3_vl(messages: list) -> str:
    """Format messages to Qwen3-VL template (vision aware)."""
    full_string = ""
    if messages and messages[0]["role"] == "system":
        full_string += (
            "<|im_start|>system\n"
            + render_content_qwen3_vl(messages[0]["content"])
            + "<|im_end|>\n"
        )
        start_index = 1
    else:
        start_index = 0

    for msg in messages[start_index:]:
        role = msg["role"]
        content = render_content_qwen3_vl(msg["content"])
        if role == "user":
            full_string += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            full_string += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        elif role in ["tool", "observation"]:
            full_string += (
                "<|im_start|>user\n<tool_response>\n"
                + content
                + "\n</tool_response><|im_end|>\n"
            )

    full_string += "<|im_start|>assistant\n"
    return full_string


def init():
    """Khởi tạo model và các tham số."""
    storage = get_model_storage()
    
    # Check nếu đã init rồi thì skip
    if is_initialized():
        print("⏭️ Model đã được khởi tạo, sử dụng instance có sẵn...")
        return
    
    print("🚀 Đang khởi tạo model...")
    
    storage['llm'] = LLM(
        model="unsloth/Qwen3-VL-4B-Thinking",
        tokenizer="unsloth/Qwen3-VL-4B-Thinking",
        max_model_len=12000,
        dtype="float16",
        quantization="bitsandbytes", 
    )
    
    storage['sampling_params'] = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=1024
    )
    
    storage['initialized'] = True
    print("✅ Model đã sẵn sàng!")


def generate_response(user_message: str, messages: list, image_data_urls: list | None = None) -> list:
    """Generate response từ LLM và xử lý tool calls."""
    storage = get_model_storage()
    
    if not is_initialized():
        return [("assistant", "❌ Lỗi: Model chưa được khởi tạo.")]
    
    responses = []
    
    try:
        # Thêm tin nhắn user với image placeholders
        image_data_urls = image_data_urls or []
        user_content = build_user_content_items(user_message, len(image_data_urls))
        messages.append({"role": "user", "content": user_content})
        
        current_conversation = format_conversation_template_qwen3_vl(messages)
        
        # --- LOGIC GENERATE LẦN 1 ---
        if image_data_urls:
            print(f"🖼️ Đang xử lý {len(image_data_urls)} ảnh...")
            images = [data_url_to_image(url) for url in image_data_urls]
            # Input format chuẩn cho vLLM multimodal: List of Dicts
            generate_input = [{
                "prompt": current_conversation,
                "multi_modal_data": {"image": images},
            }]
            outputs = storage['llm'].generate(generate_input, storage['sampling_params'])
        else:
            # Input format chuẩn cho text: List of Strings (QUAN TRỌNG: Phải bọc trong [])
            outputs = storage['llm'].generate([current_conversation], storage['sampling_params'])
            
        response = outputs[0].outputs[0].text
        responses.append(("assistant", response))
        
        # Parse và xử lý tool calls
        tool_calls = parse_tool_calls(response)
        
        # --- VÒNG LẶP XỬ LÝ TOOL ---
        while tool_calls:
            all_results = []
            tool_info = []
            
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                params = tc.get("arguments", {})
                
                # Execute tool
                result = execute_tool_call(tc, TOOL_REGISTRY)
                all_results.append(result)
                
                # Lưu info để hiển thị UI
                tool_info.append({
                    "name": tool_name,
                    "params": params,
                    "result": result[:500] + ('...' if len(result) > 500 else '')
                })
            
            # Đẩy tool info ra UI
            responses.append(("tool", tool_info))
            
            # Cập nhật conversation history với kết quả tool
            combined_result = "\n---\n".join(all_results)
            messages.append({"role": "tool", "content": combined_result})
            
            # Generate tiếp sau khi có kết quả tool
            current_conversation = format_conversation_template_qwen3_vl(messages)
            
            # Khi generate tiếp (thường là text only), vẫn phải bọc list []
            outputs = storage['llm'].generate([current_conversation], storage['sampling_params'])
            
            response = outputs[0].outputs[0].text
            responses.append(("assistant", response))
            
            # Check xem model có gọi tool tiếp không
            tool_calls = parse_tool_calls(response)
        
        # Lưu response cuối cùng vào history
        messages.append({"role": "assistant", "content": response})
        return responses

    except Exception as e:
        traceback.print_exc()
        return [("assistant", f"❌ Lỗi hệ thống Backend: {str(e)}")]

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
        
        /* Thinking dropdown styles - ChatGPT style */
        .thinking-dropdown {
            max-width: 85%;
            margin: 8px 0;
        }
        
        .thinking-dropdown .q-expansion-item {
            background: #fafafa !important;
            border-radius: 16px !important;
            border: 1px solid #e0e0e0 !important;
            overflow: hidden;
        }
        
        .thinking-dropdown .q-expansion-item__container {
            background: transparent !important;
        }
        
        .thinking-dropdown .q-item {
            padding: 12px 16px !important;
            min-height: 48px !important;
        }
        
        .thinking-dropdown .q-item__section--avatar {
            min-width: 32px !important;
            padding-right: 12px !important;
        }
        
        .thinking-dropdown .q-expansion-item__content {
            background: #f5f5f5 !important;
            border-top: 1px solid #e8e8e8 !important;
        }
        
        .thinking-content {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.8em;
            color: #555;
            padding: 12px 16px;
            white-space: pre-wrap;
            word-break: break-word;
            line-height: 1.6;
        }
        
        .thinking-header-text {
            color: #666;
            font-weight: 600;
            font-size: 0.9em;
        }
        
        .thinking-icon {
            color: #f59e0b;
        }
        
        .tool-icon {
            color: var(--heineken-green);
        }
        
        /* Thinking specific styles */
        .thinking-dropdown .q-expansion-item {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
            border: 1px solid #fcd34d !important;
        }
        
        .thinking-dropdown .q-expansion-item__content {
            background: #fffef5 !important;
            border-top: 1px solid #fcd34d !important;
        }

        /* Pasted image preview (Gemini-like) */
        .image-preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 0 16px 8px 16px;
        }

        .image-preview {
            position: relative;
            width: 84px;
            height: 84px;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid #e2e8f0;
            background: #fff;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
        }

        .image-preview-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .image-preview-remove {
            position: absolute;
            top: 4px;
            right: 4px;
            border: none;
            background: rgba(0, 0, 0, 0.6);
            color: #fff;
            border-radius: 999px;
            width: 20px;
            height: 20px;
            cursor: pointer;
            font-size: 12px;
            line-height: 20px;
            text-align: center;
        }

        .message-user-images {
            display: flex;
            gap: 8px;
            max-width: 70%;
            margin-left: auto;
            flex-wrap: wrap;
        }

        .user-image-thumb {
            width: 96px;
            height: 96px;
            border-radius: 12px;
            object-fit: cover;
            border: 2px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }
        
        /* Tool dropdown styles */
        .tool-dropdown .q-expansion-item {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%) !important;
            border: 1px solid var(--heineken-accent) !important;
        }
        
        .tool-dropdown .q-expansion-item__content {
            background: #f8fff8 !important;
            border-top: 1px solid var(--heineken-accent) !important;
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
    ui.add_head_html(f'''
    <script>
        window.__pendingImages = [];
        window.__maxImages = {MAX_PASTED_IMAGES};
        window.__previewContainerId = 'paste-preview-container';
        
        // Config cho resize ảnh - giảm kích thước để tránh lỗi WebSocket
        window.__imageMaxWidth = 800;
        window.__imageMaxHeight = 800;
        window.__imageQuality = 0.7;

        // Hàm resize ảnh trước khi lưu vào pendingImages
        async function resizeImageForTransmission(dataUrl) {{
            return new Promise((resolve) => {{
                const img = new Image();
                img.onload = () => {{
                    let width = img.width;
                    let height = img.height;
                    
                    // Tính toán kích thước mới giữ tỷ lệ
                    if (width > window.__imageMaxWidth || height > window.__imageMaxHeight) {{
                        const ratio = Math.min(
                            window.__imageMaxWidth / width,
                            window.__imageMaxHeight / height
                        );
                        width = Math.round(width * ratio);
                        height = Math.round(height * ratio);
                    }}
                    
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    // Convert sang JPEG với quality thấp hơn để giảm size
                    resolve(canvas.toDataURL('image/jpeg', window.__imageQuality));
                }};
                img.onerror = () => resolve(dataUrl); // Fallback nếu lỗi
                img.src = dataUrl;
            }});
        }}

        function renderPreviews() {{
            const container = document.getElementById(window.__previewContainerId);
            if (!container) return;
            container.innerHTML = '';
            window.__pendingImages.forEach((dataUrl, index) => {{
                const wrapper = document.createElement('div');
                wrapper.className = 'image-preview';

                const img = document.createElement('img');
                img.src = dataUrl;
                img.className = 'image-preview-img';

                const remove = document.createElement('button');
                remove.type = 'button';
                remove.className = 'image-preview-remove';
                remove.textContent = '✕';
                remove.onclick = () => {{
                    window.__pendingImages.splice(index, 1);
                    renderPreviews();
                }};

                wrapper.appendChild(img);
                wrapper.appendChild(remove);
                container.appendChild(wrapper);
            }});
        }}

        async function handlePaste(event) {{
            // Check if we're in the chat input area (Quasar q-input creates nested input)
            const activeEl = document.activeElement;
            const isInputFocused = activeEl && (
                activeEl.tagName === 'INPUT' ||
                activeEl.tagName === 'TEXTAREA' ||
                activeEl.closest('.input-area')
            );
            if (!isInputFocused) return;

            const items = (event.clipboardData || window.clipboardData)?.items;
            if (!items) return;
            const imageItems = [];
            for (const item of items) {{
                if (item.type && item.type.startsWith('image/')) {{
                    imageItems.push(item);
                }}
            }}
            if (!imageItems.length) return;
            event.preventDefault();
            for (const item of imageItems) {{
                if (window.__pendingImages.length >= window.__maxImages) break;
                const file = item.getAsFile();
                if (!file) continue;
                const dataUrl = await new Promise((resolve) => {{
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.readAsDataURL(file);
                }});
                // QUAN TRỌNG: Resize ảnh TRƯỚC khi lưu để tránh lỗi WebSocket transmission
                const resizedDataUrl = await resizeImageForTransmission(dataUrl);
                window.__pendingImages.push(resizedDataUrl);
            }}
            renderPreviews();
        }}

        // Attach to document level to catch all paste events
        document.addEventListener('paste', handlePaste);
    </script>
    ''')
    
    # Tạo conversation mới cho mỗi page load (F5 = reset)
    conversation_messages = create_new_conversation()
    
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
        
        # Input area with image preview
        with ui.column().classes('w-full max-w-4xl mx-auto mb-6 input-area'):
            # Image preview area (shows pasted images before sending) - above input
            ui.html(
                '<div id="paste-preview-container" class="image-preview-container"></div>',
                sanitize=False,
            )
            with ui.row().classes('w-full items-center p-2'):
                user_input = ui.input(placeholder='Nhập tin nhắn hoặc Ctrl+V để dán ảnh...').classes(
                    'flex-grow bg-transparent border-none text-gray-800'
                ).props('borderless dense')
                
                spinner = ui.spinner('dots', size='lg', color='green').classes('hidden')
                send_button = ui.button(icon='send').classes('send-btn text-white')
        
        async def send_message():
            # 1. Lấy dữ liệu
            message = (user_input.value or '').strip()
            
            # --- FIX: Tăng timeout lên 30s để kịp nhận ảnh lớn ---
            image_data_urls = await ui.run_javascript('window.__pendingImages || []', timeout=30.0)
            
            if not message and not image_data_urls:
                return
            
            # 2. Clear Input & Preview ngay lập tức
            user_input.value = ''
            # Tăng timeout cho lệnh clear đề phòng UI lag
            await ui.run_javascript('window.__pendingImages = []; renderPreviews();', timeout=5.0)
            
            # 3. Render tin nhắn User lên UI (Dùng Thumbnail cho ảnh)
            with messages_column:
                if message:
                    with ui.row().classes('w-full justify-end'):
                        ui.label(message).classes('message-user')
                
                if image_data_urls:
                    with ui.row().classes('w-full justify-end'):
                        with ui.row().classes('message-user-images'):
                            for url in image_data_urls:
                                # Tạo thumbnail hiển thị cho nhẹ UI
                                thumb_url = make_thumbnail(url)
                                ui.image(thumb_url).classes('user-image-thumb')
            
            # Scroll xuống dưới & Bật spinner
            chat_container.scroll_to(percent=1.0)
            spinner.classes(remove='hidden')
            send_button.disable()
            
            try:
                # 4. Gọi Backend (Chạy thread riêng để ko block UI)
                responses = await asyncio.to_thread(
                    generate_response,
                    message,
                    conversation_messages,
                    image_data_urls, 
                )
                
                # 5. Render tin nhắn Assistant & Tool lên UI
                with messages_column:
                    for role, content in responses:
                        if role == "assistant":
                            # Parse thinking logic
                            thinking, main_response = parse_think_tags(content)
                            
                            # Render Thinking (nếu có)
                            if thinking:
                                with ui.row().classes('w-full justify-start thinking-dropdown'):
                                    with ui.expansion(
                                        text='', icon='psychology', value=False
                                    ).classes('w-full') as expansion:
                                        expansion._props['header-class'] = 'thinking-header'
                                        expansion._props['expand-icon-class'] = 'text-gray-500'
                                        
                                        with expansion.add_slot('header'):
                                            with ui.row().classes('items-center gap-2'):
                                                ui.icon('lightbulb', size='sm').classes('thinking-icon')
                                                ui.label('Suy nghĩ...').classes('thinking-header-text')
                                        
                                        with ui.column().classes('thinking-content w-full'):
                                            ui.label(thinking).classes('text-sm text-gray-600 whitespace-pre-wrap')
                            
                            # Render Main Response
                            if main_response:
                                with ui.row().classes('w-full justify-start'):
                                    ui.label(main_response).classes('message-assistant')
                        
                        elif role == "tool":
                            # Render Tool Dropdown
                            tool_list = content
                            tool_count = len(tool_list)
                            tool_names = ', '.join([t['name'] for t in tool_list])
                            
                            with ui.row().classes('w-full justify-start thinking-dropdown tool-dropdown'):
                                with ui.expansion(
                                    text='', icon='build', value=False
                                ).classes('w-full') as expansion:
                                    expansion._props['header-class'] = 'tool-header'
                                    expansion._props['expand-icon-class'] = 'text-gray-500'
                                    
                                    with expansion.add_slot('header'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('construction', size='sm').classes('tool-icon')
                                            ui.label(f'Đã dùng {tool_count} tool: {tool_names}').classes('thinking-header-text')
                                    
                                    with ui.column().classes('thinking-content w-full gap-3'):
                                        for tool in tool_list:
                                            with ui.card().classes('w-full bg-white/50'):
                                                ui.label(f"🔧 {tool['name']}").classes('font-bold text-green-800')
                                                ui.label(f"📥 Params: {tool['params']}").classes('text-xs text-gray-600')
                                                with ui.scroll_area().classes('max-h-32'):
                                                    ui.label(f"📤 Result: {tool['result']}").classes('text-xs text-gray-700 whitespace-pre-wrap')
                
                chat_container.scroll_to(percent=1.0)
                
            except Exception as e:
                with messages_column:
                    with ui.row().classes('w-full justify-center'):
                        ui.label(f"❌ Lỗi UI: {str(e)}").classes('text-red-400')
                traceback.print_exc()
            
            finally:
                spinner.classes(add='hidden')
                send_button.enable()
                # Dọn dẹp pending images lần nữa cho chắc
                await ui.run_javascript('window.__pendingImages = []; renderPreviews();', timeout=5.0)
            # 1. Lấy dữ liệu
            message = (user_input.value or '').strip()
            image_data_urls = await ui.run_javascript('window.__pendingImages || []')
            
            if not message and not image_data_urls:
                return
            
            # 2. Clear Input & Preview ngay lập tức
            user_input.value = ''
            await ui.run_javascript('window.__pendingImages = []; renderPreviews();')
            
            # 3. Render tin nhắn User lên UI (Dùng Thumbnail cho ảnh)
            with messages_column:
                if message:
                    with ui.row().classes('w-full justify-end'):
                        ui.label(message).classes('message-user')
                
                if image_data_urls:
                    with ui.row().classes('w-full justify-end'):
                        with ui.row().classes('message-user-images'):
                            for url in image_data_urls:
                                # FIX: Tạo thumbnail nhỏ để hiển thị UI mượt hơn
                                thumb_url = make_thumbnail(url)
                                ui.image(thumb_url).classes('user-image-thumb')
            
            # Scroll xuống dưới & Bật spinner
            chat_container.scroll_to(percent=1.0)
            spinner.classes(remove='hidden')
            send_button.disable()
            
            try:
                # 4. Gọi Backend (Chạy thread riêng để ko block UI)
                # Truyền image_data_urls (ảnh gốc full HD) cho LLM
                responses = await asyncio.to_thread(
                    generate_response,
                    message,
                    conversation_messages,
                    image_data_urls, 
                )
                
                # 5. Render tin nhắn Assistant & Tool lên UI
                with messages_column:
                    for role, content in responses:
                        if role == "assistant":
                            # Parse thinking logic
                            thinking, main_response = parse_think_tags(content)
                            
                            # Render Thinking (nếu có)
                            if thinking:
                                with ui.row().classes('w-full justify-start thinking-dropdown'):
                                    with ui.expansion(
                                        text='', icon='psychology', value=False
                                    ).classes('w-full') as expansion:
                                        expansion._props['header-class'] = 'thinking-header'
                                        expansion._props['expand-icon-class'] = 'text-gray-500'
                                        
                                        with expansion.add_slot('header'):
                                            with ui.row().classes('items-center gap-2'):
                                                ui.icon('lightbulb', size='sm').classes('thinking-icon')
                                                ui.label('Suy nghĩ...').classes('thinking-header-text')
                                        
                                        with ui.column().classes('thinking-content w-full'):
                                            ui.label(thinking).classes('text-sm text-gray-600 whitespace-pre-wrap')
                            
                            # Render Main Response
                            if main_response:
                                with ui.row().classes('w-full justify-start'):
                                    ui.label(main_response).classes('message-assistant')
                        
                        elif role == "tool":
                            # Render Tool Dropdown
                            tool_list = content
                            tool_count = len(tool_list)
                            tool_names = ', '.join([t['name'] for t in tool_list])
                            
                            with ui.row().classes('w-full justify-start thinking-dropdown tool-dropdown'):
                                with ui.expansion(
                                    text='', icon='build', value=False
                                ).classes('w-full') as expansion:
                                    expansion._props['header-class'] = 'tool-header'
                                    expansion._props['expand-icon-class'] = 'text-gray-500'
                                    
                                    with expansion.add_slot('header'):
                                        with ui.row().classes('items-center gap-2'):
                                            ui.icon('construction', size='sm').classes('tool-icon')
                                            ui.label(f'Đã dùng {tool_count} tool: {tool_names}').classes('thinking-header-text')
                                    
                                    with ui.column().classes('thinking-content w-full gap-3'):
                                        for tool in tool_list:
                                            with ui.card().classes('w-full bg-white/50'):
                                                ui.label(f"🔧 {tool['name']}").classes('font-bold text-green-800')
                                                ui.label(f"📥 Params: {tool['params']}").classes('text-xs text-gray-600')
                                                with ui.scroll_area().classes('max-h-32'):
                                                    ui.label(f"📤 Result: {tool['result']}").classes('text-xs text-gray-700 whitespace-pre-wrap')
                
                chat_container.scroll_to(percent=1.0)
                
            except Exception as e:
                with messages_column:
                    with ui.row().classes('w-full justify-center'):
                        ui.label(f"❌ Lỗi UI: {str(e)}").classes('text-red-400')
                traceback.print_exc()
            
            finally:
                spinner.classes(add='hidden')
                send_button.enable()
                # Dọn dẹp pending images lần nữa cho chắc
                await ui.run_javascript('window.__pendingImages = []; renderPreviews();')
        
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
