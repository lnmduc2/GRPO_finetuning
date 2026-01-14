# === MEMORY OPTIMIZATION ENV VARS ===
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Giảm memory fragmentation

# Load model và tokenizer
import torch
from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length = 512,

    full_finetuning = False,
    load_in_4bit = False,  # MUST be False for QAT - load full precision weights
    dtype = torch.bfloat16,  # Use bf16 for memory efficiency
    
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Giảm từ 16 xuống 8 để tiết kiệm VRAM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,  # Giảm theo tỷ lệ với r (alpha/r = 2)
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    qat_scheme = "int4",
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

        
import re
from unsloth.chat_templates import get_chat_template
from ChatBotSynthetic.training.sft import DatasetLoader
dataset_config = {
        "dataset": {
            "train_path": "ChatBotSynthetic/data/train.jsonl",
            "validation_path": "ChatBotSynthetic/data/validation.jsonl",
            "format": "json",  # Note: use "json" not "jsonl" for HF datasets
            "message_field": "messages",
            "text_field": "text"
        }
    }
loader = DatasetLoader(dataset_config, tokenizer)
train_dataset = loader.prepare_dataset(split="train")
val_dataset = loader.prepare_dataset(split="validation")

SYSTEM_PROMPT = "Bạn là nhân viên CSKH Heineken Vietnam đang hỗ trợ trợ khách hàng theo những quy trình có sẵn."

def format_multi_turn_grpo(example):
    """
    Format dataset theo đúng định dạng của GRPO training với reasoning_content và tool_calls
    """
    import json
    from datetime import datetime, date
    
    def serialize_tool_args(obj):
        """Helper function để serialize các object không phải JSON standard"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: serialize_tool_args(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_tool_args(item) for item in obj]
        elif obj is None:
            return None
        else:
            return obj
    
    messages = example["messages"]
    
    # Xử lý từng message để format lại content
    formatted_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        tool_calls = msg.get("tool_calls")
        
        # Xây dựng nội dung hoàn chỉnh cho assistant
        if role == "assistant":
            full_content = ""
            
            # Thêm phần reasoning vào thẻ <think> nếu có
            if reasoning and reasoning.strip():
                full_content += f"<think>{reasoning}</think>"
            
            # Thêm phần content chính (response text)
            if content and content.strip():
                if full_content:  # Nếu đã có reasoning, thêm xuống dòng
                    full_content += "\n"
                full_content += content
            
            # Xử lý tool_calls nếu có
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    
                    # Serialize tool_args để xử lý datetime và các object khác
                    serialized_args = serialize_tool_args(tool_args)
                    
                    # Tạo chuỗi tool call
                    tool_call_str = f"\n<tool_call>\n{json.dumps({'name': tool_name, 'arguments': serialized_args}, ensure_ascii=False, indent=2)}\n</tool_call>"
                    full_content += tool_call_str
            
            formatted_messages.append({
                "role": role,
                "content": full_content.strip() if full_content.strip() else ""
            })
        else:
            # Giữ nguyên các message không phải assistant
            formatted_messages.append({
                "role": role,
                "content": content
            })
    
    # Kiểm tra message cuối cùng phải là assistant
    if not formatted_messages or formatted_messages[-1]["role"] != "assistant":
        return {"prompt": None, "answer": None}
    
    # Lấy nội dung assistant cuối cùng làm answer
    last_assistant_content = formatted_messages[-1]["content"]
    
    # --- Xây dựng Prompt (Lịch sử hội thoại) ---
    prompt_history = []
    
    # Thêm system prompt nếu chưa có
    if formatted_messages[0]["role"] != "system":
        prompt_history.append({"role": "system", "content": SYSTEM_PROMPT})
    
    # Thêm tất cả các tin nhắn TRỪ tin nhắn assistant cuối cùng vào prompt
    prompt_history.extend(formatted_messages[:-1])
    
    # Tạo dict mới với các field cần thiết
    result = {
        "prompt": prompt_history,
        "answer": last_assistant_content
    }

    # Xóa các key procedure_id, procedure_name, procedure_purpose, procedure_main_flow, procedure_edge_cases, messages đi
    example.pop("procedure_id", None)
    example.pop("procedure_name", None)
    example.pop("procedure_purpose", None)
    example.pop("procedure_main_flow", None)
    example.pop("procedure_edge_cases", None)
    example.pop("messages", None)
    example.pop("text", None)
    
    return result

from datasets import load_dataset
train_dataset = train_dataset.map(format_multi_turn_grpo, batched=False)
val_dataset = val_dataset.map(format_multi_turn_grpo, batched=False)

import re

RESPONSE_PATTERN = (
        r"^<think>"  # Mở thẻ think
        r"\s*Tình huống:\s*.+?"  # Tình huống (bắt buộc có nội dung)
        r"\s*\n\s*Quy trình:\s*(?:không xác định|không liên quan|.+?)"  # Quy trình
        r"\s*\n\s*Bước:\s*(?:(?:\d+\s*-\s*[^\n]+)|(?:ngoại lệ\s*-\s*[^\n]+)|)"  # Bước
        r"\s*\n\s*Thông tin có:\s*.+?"  # Thông tin có
        r"\s*\n\s*Thông tin cần thêm:\s*.+?"  # Thông tin cần thêm
        r"\s*\n\s*Hành động:\s*.+?"  # Hành động
        r"\s*</think>"  # Đóng thẻ think
        r"\s*(?:"  # Sau think có thể là:
            r"<tool_call>.*?</tool_call>"  # Tool call
            r"|"  # HOẶC
            r".+"  # Response text thông thường
        r")"
    )

def match_response_pattern(res):
    return re.match(RESPONSE_PATTERN, res, re.DOTALL | re.IGNORECASE)

def match_format_exactly(prompts, completions, *args, **kwargs):
    """
    Pattern cho format:
    <think>
    Tình huống: ...
    Quy trình: ...
    Bước: ...
    Thông tin có: ...
    Thông tin cần thêm: ...
    Hành động: ...
    </think>
    [Nội dung trả lời HOẶC <tool_call>...</tool_call>]
    """
    
    responses = [c[0] for c in completions]
    scores = []
    
    for res in responses:
        if match_response_pattern(res):
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores

def filter_valid_format(example):
    # LƯU Ý: Dựa trên code test của bạn (original["answer"]), 
    # tôi giả định nội dung cần kiểm tra nằm ở cột 'answer'.
    # Nếu dataset sau khi format_multi_turn_grpo lưu ở cột khác (vd: 'completion'),
    # hãy đổi 'answer' thành tên cột đó.
    content = example.get("answer", "") 
    
    return match_response_pattern(content) is not None
    
# Kiểm tra số lượng ban đầu
print(f"Original Train Size: {len(train_dataset)}")
print(f"Original Val Size: {len(val_dataset)}")

# Filter train dataset
train_dataset_filtered = train_dataset.filter(filter_valid_format)

# Filter val dataset
val_dataset_filtered = val_dataset.filter(filter_valid_format)

# 4. Kết quả
print("-" * 30)
print(f"Filtered Train Size: {len(train_dataset_filtered)} (Removed: {len(train_dataset) - len(train_dataset_filtered)})")
print(f"Filtered Val Size:   {len(val_dataset_filtered)} (Removed: {len(val_dataset) - len(val_dataset_filtered)})")

# Cập nhật lại biến chính nếu cần
train_dataset = train_dataset_filtered
val_dataset = val_dataset_filtered

import os
from dotenv import load_dotenv
load_dotenv()
import re
import logging
import os
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# --- Cấu hình Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Client (Đảm bảo đã set biến môi trường OPENAI_API_KEY)
# Trong môi trường training, nên khởi tạo client bên ngoài function nếu có thể để tránh init lại nhiều lần
client = OpenAI()

# --- 1. Định nghĩa Schema Output (Giữ nguyên) ---
class QualityScore(BaseModel):
    """Cấu trúc điểm số trả về từ LLM Judge."""
    reasoning: str = Field(
        description="Giải thích ngắn gọn (dưới 50 từ) tại sao lại cho điểm này. "
                    "Tập trung vào sự sai lệch giữa suy nghĩ và câu trả lời."
    )
    score: float = Field(
        description="Điểm chất lượng từ 1.0 đến 5.0. 1.0 là tệ nhất, 5.0 là hoàn hảo."
    )

    @field_validator('score')
    def check_score_range(cls, v):
        if v < 1.0: return 1.0
        if v > 5.0: return 5.0
        return v

# --- 2. Helper gọi OpenAI cho từng item ---
def _call_judge_single(item: dict) -> float:
    """Hàm worker để xử lý từng request (để chạy song song)."""
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-mini", # Model rẻ và nhanh cho reward function
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": f"""
Bạn là Judge chuyên nghiệp đánh giá chất lượng AI CSKH Heineken Việt Nam.

### NHIỆM VỤ
Chấm điểm từ 0.0 đến 5.0 dựa trên 3 tiêu chí:
1. **Format**: Có tuân thủ đúng cấu trúc 6 trường không?
2. **Logic**: Suy luận có hợp lý và nhất quán không?
3. **Accuracy**: Câu trả lời có khớp Ground Truth không?

### CẤU TRÚC FORMAT BẮT BUỘC
```
<think>
Tình huống: [mô tả]
Quy trình: [tên quy trình hoặc "không xác định"]
Bước: [số - mô tả hoặc "ngoại lệ - ..."]
Thông tin có: [liệt kê]
Thông tin cần thêm: [liệt kê hoặc "Không"]
Hành động: [hành động cụ thể]
</think>
[Câu trả lời hoặc <tool_call>]
```

### THANG ĐIỂM CHI TIẾT

**5.0 điểm (Xuất sắc)**
- ✅ Format hoàn hảo (đã được pre-check)
- ✅ Logic rõ ràng, chính xác:
  * Tình huống được mô tả đúng với câu hỏi
  * Quy trình và Bước xác định chính xác
  * Thông tin có/cần liệt kê đầy đủ và logic
  * Hành động hợp lý (gọi tool đúng tham số HOẶC trả lời trực tiếp)
- ✅ Câu trả lời khớp 100% với Ground Truth (về ý nghĩa, cho phép diễn đạt khác)

**4.0 - 4.5 điểm (Tốt)**
- ✅ Format đúng
- ✅ Logic tốt nhưng có sai sót NHỎ:
  * Tình huống đúng nhưng mô tả chưa đầy đủ
  * Bước xác định đúng hướng nhưng chưa chính xác 100%
  * Thông tin có/cần thiếu 1-2 chi tiết không quan trọng
- ✅ Câu trả lời đúng hướng, sai lệch nhỏ (<10% nội dung)

**3.0 - 3.5 điểm (Trung bình)**
- ✅ Format đúng
- ⚠️ Logic có vấn đề TRUNG BÌNH:
  * Xác định Quy trình/Bước SAI nhưng vẫn trong phạm vi liên quan
  * Thông tin có/cần liệt kê THIẾU hoặc DƯ thừa đáng kể
  * Hành động đúng nhưng lý do không rõ ràng
- ⚠️ Câu trả lời đúng 60-80% nội dung

**2.0 - 2.5 điểm (Yếu)**
- ✅ Format đúng
- ❌ Logic SAI NGHIÊM TRỌNG:
  * Tình huống xác định SAI HOÀN TOÀN
  * Quy trình không liên quan đến câu hỏi
  * Suy luận một đằng, hành động một nẻo (mâu thuẫn)
  * Copy-paste hoặc điền bừa vào các trường
- ❌ Câu trả lời sai 40-60%

**1.0 - 1.5 điểm (Rất yếu)**
- ✅ Format đúng
- ❌ Logic hoàn toàn vô nghĩa hoặc không liên quan
- ❌ Câu trả lời SAI HOÀN TOÀN hoặc HALLUCINATION (bịa đặt thông tin)

**0.5 điểm (Lỗi Format)**
- ❌ Thiếu trường bắt buộc
- ❌ Sai thứ tự các trường
- ❌ Không khớp Regex pattern
(Điểm này đã được xử lý bằng pre-check, LLM không cần chấm)

### QUY TẮC CHẤM ĐIỂM
1. **Ưu tiên Accuracy**: Nếu câu trả lời SAI so với Ground Truth → tối đa 2.0 điểm
2. **Logic quan trọng hơn chi tiết**: Sai Quy trình/Bước nghiêm trọng hơn sai chi tiết nhỏ
3. **Không khoan nhượng Hallucination**: Bịa đặt thông tin → tối đa 1.5 điểm
4. **Tool call**: Nếu gọi tool, kiểm tra tham số có đúng với thông tin đã có không

### VÍ DỤ CHẤM ĐIỂM

**Ví dụ 5.0 điểm:**
- Câu hỏi: "Tôi là chủ điểm bán, mã 305210, SĐT 0909123456"
- Ground Truth: Xác thực thông tin và hỗ trợ
- AI Thinking: "Tình huống: KH là chủ điểm bán cung cấp mã và SĐT\nQuy trình: Quên/Đổi mật khẩu\nBước: 1 - Xác thực thông tin\n..."
- AI Response: Gọi tool tra_cuu_thong_tin với mã 305210
→ Logic chính xác, hành động đúng

**Ví dụ 1.0 điểm:**
- Câu hỏi: "Heineken Ken có vị gì?"
- Ground Truth: "Heineken có vị đắng nhẹ, hương trái cây"
- AI Thinking: "Tình huống: KH phàn nàn giá cả\nQuy trình: Xử lý khiếu nại\n..."
- AI Response: "Heineken có vị ngọt và màu vàng óng"
→ Xác định tình huống SAI, câu trả lời HALLUCINATION

### LƯU Ý
- Chấm điểm NGHIÊM KHẮC nhưng công bằng
- Reasoning phải NÊU RÕ lý do chấm điểm theo 3 tiêu chí: Format, Logic, Accuracy
- Không chấm điểm trung lập (3.0) khi không chắc chắn → Phải có căn cứ rõ ràng
                    """
                },
                {
                    "role": "user",
                    "content": (
                        f"# Dữ liệu đánh giá:\n"
                        f"- Câu hỏi khách hàng: {item['question']}\n"
                        f"- Đáp án chuẩn (Ground Truth): {item['ground_truth']}\n"
                        f"- AI Suy luận (Thinking Part): {item['ai_thinking']}\n"
                        f"- AI Trả lời (Response Part): {item['ai_response']}\n\n"
                        f"Hãy phân tích ngắn gọn và đưa ra điểm số chính xác."
                    )
                }
            ],
            response_format=QualityScore, # Tính năng Structured Outputs native
        )
        
        # Lấy kết quả đã được parse thành Pydantic object
        result: QualityScore = completion.choices[0].message.parsed
        
        # Debug nhẹ để xem model lý luận thế nào (có thể comment lại khi train thật)
        # logger.info(f"Reasoning: {result.reasoning} | Score: {result.score}")
        
        return result.score

    except Exception as e:
        print(f"\n>>> ❌ LỖI GỌI API: {type(e).__name__}")
        print(f"Chi tiết: {e}")
        # Nếu lỗi là AuthenticationError -> Sai key
        # Nếu lỗi là AttributeError: '...' has no attribute 'parse' -> Cần update openai
        return 0.0

# --- 3. Hàm Reward Function Chính ---
def judge_thinking_and_answer_alignment(prompts: List[Any], completions: List[Any], answer: List[Any], **kwargs) -> List[float]:
    """
    Reward function thay thế LangChain bằng OpenAI SDK + ThreadPoolExecutor.
    """
    
    # Regex tách thinking
    split_pattern = r"(?:<think>|<thinking>)(.*?)(?:</think>|</thinking>)(.*)"
    
    batch_inputs = []

    # --- Pre-processing Data (Normalize inputs) ---
    normalized_completions = []
    for c in completions:
        if isinstance(c, list) and len(c) > 0:
            content = c[0].get('content', '') if isinstance(c[0], dict) else str(c[0])
        elif isinstance(c, dict):
            content = c.get('content') or c.get('generated_text', '')
        else:
            content = str(c)
        normalized_completions.append(content)

    for prompt, text, gt in zip(prompts, normalized_completions, answer):
        # Lấy câu hỏi user
        user_q = prompt[-1]['content'] if isinstance(prompt, list) else str(prompt)

        # Tách thinking & response
        match = re.search(split_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            thinking_part = match.group(1).strip()
            response_part = match.group(2).strip()
        else:
            thinking_part = "N/A (Missing <think> tag)"
            response_part = text
        
        batch_inputs.append({
            "question": user_q,
            "ground_truth": gt,
            "ai_thinking": thinking_part,
            "ai_response": response_part
        })

    # --- Execution: Chạy song song để tối ưu tốc độ ---
    # LangChain dùng async/batch ngầm, với SDK thuần ta dùng ThreadPoolExecutor
    scores = [0.0] * len(batch_inputs)
    
    with ThreadPoolExecutor(max_workers=5) as executor: # Điều chỉnh max_workers tùy rate limit
        # Submit tất cả task và giữ map index để trả về đúng thứ tự
        future_to_index = {executor.submit(_call_judge_single, item): i for i, item in enumerate(batch_inputs)}
        
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                scores[idx] = future.result()
            except Exception as exc:
                logger.error(f"Item {idx} generated an exception: {exc}")
                scores[idx] = 0.0

    return scores

import re
import numpy as np

# Hàm xử lý clean <think> và tokenize
def process_and_tokenize(examples):
    # 1. Lấy Raw Text ra trước (chưa tokenize)
    texts = [tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False) for p in examples["prompt"]]
    
    # 2. Dùng Regex xóa bỏ <think> (và khoảng trắng thừa) NẾU nó nằm ở cuối cùng
    # r'<think>\s*$' : Tìm chữ <think> theo sau là khoảng trắng bất kỳ (\s*) ở cuối chuỗi ($)
    cleaned_texts = [re.sub(r'<think>\s*$', '', t).rstrip() for t in texts]
    
    # 3. Tokenize hàng loạt (nhanh hơn loop)
    # Lưu ý: add_special_tokens=False vì chat_template thường đã lo việc thêm bos/eos rồi
    tokens = tokenizer(cleaned_texts, add_special_tokens=False)["input_ids"]
    
    return {"tokens": tokens}

# Áp dụng vào dataset
tokenized = train_dataset.map(
    process_and_tokenize,
    batched=True,
)

# --- PHẦN KIỂM TRA LẠI (QUAN TRỌNG) ---
print("--- Sample decoded check ---")
decoded_sample = tokenizer.decode(tokenized[0]["tokens"])
print(decoded_sample) 
# Kiểm tra xem cuối chuỗi có bị dính <think> không.
# Nếu đúng, nó sẽ kết thúc bằng header của Assistant (vd: "<|im_start|>assistant")

# --- Phần tính toán Length giữ nguyên ---
tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
maximum_length = int(np.quantile(tokenized["L"], 0.9))
print("Max Length =", maximum_length)

# Filter
train_dataset = train_dataset.select(np.where(np.array(tokenized["L"]) <= maximum_length)[0])
del tokenized

def filter_valid_records(example):
    return example['prompt'][-1]['role'] != 'assistant'

train_dataset = train_dataset.filter(filter_valid_records)
val_dataset = val_dataset.filter(filter_valid_records)

def format_multi_turn_to_tool_calling(example):
    messages = example['prompt']
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
    
    return {"prompt": full_string}

train_dataset = train_dataset.map(format_multi_turn_to_tool_calling)
val_dataset = val_dataset.map(format_multi_turn_to_tool_calling)

max_seq_length = 512 # Giảm từ 4096 xuống 2048 để tiết kiệm VRAM
max_prompt_length = min(maximum_length + 1, 512)  # Cap prompt length
max_completion_length = max_seq_length - max_prompt_length

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    torch_compile=False,
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    
    # === MEMORY OPTIMIZATION ===
    per_device_train_batch_size = 1, 
    gradient_accumulation_steps = 4,  
    num_generations = 2,  
    
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 50,
    save_steps = 5,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs_qwen3_4B_grpo_qat",
    save_total_limit = 3,

    # For optional training + evaluation
    fp16_full_eval = True,
    per_device_eval_batch_size = 2, 
    eval_accumulation_steps = 1,
    eval_strategy = "steps",
    eval_steps = 5,

    metric_for_best_model="eval_loss", # Early stopping config
    fp16 = False,   
    bf16 = True,  
)

from transformers import EarlyStoppingCallback

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,
        judge_thinking_and_answer_alignment
    ],
    args = training_args,
    # For optional training + evaluation
    train_dataset = train_dataset,
    eval_dataset = val_dataset.select(range(1)), # Chỉ eval 1 record để tiết kiệm VRAM
)
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 3,     # How many steps we will wait if the eval loss doesn't decrease
                                     # For example the loss might increase, but decrease after 3 steps
    early_stopping_threshold = 0.0,  # Can set higher - sets how much loss should decrease by until
                                     # we consider early stopping. For eg 0.01 means if loss was
                                     # 0.02 then 0.01, we consider to early stop the run.
)
trainer.add_callback(early_stopping_callback)
print(f"✅ Process đang ở device {trainer.accelerator.device}")

if __name__ == "__main__":
    # === CLEAR MEMORY BEFORE TRAINING ===
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 Cleared cache. VRAM before train: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    trainer.train()