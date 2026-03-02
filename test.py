import os
import shutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# 베이스라인 설정
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# 데이터 길이 설정
NUM_CALIBRATION_SAMPLES = 128   
MAX_SEQUENCE_LENGTH = 512      

# 저장 경로 설정
SUBMIT_DIR = "./submit"
MODEL_DIR = os.path.join(SUBMIT_DIR, "model")


# 1. 모델 & 토크나이저 로드
print(">>> [1/4] 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. 데이터 준비 (MANTA-1M 데이터셋 사용)
print(">>> [2/4] 학습용 데이터 준비 중...")
ds = load_dataset(DATASET_ID, split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
    }

ds = ds.map(preprocess)

# 3. GPTQ 양자화 수행 (핵심!)
print(f">>> [3/4] 모델 경량화 시작...")

recipe = [
    GPTQModifier(
        scheme="W4A16",  # 4비트 양자화
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"], # 성능 보호를 위해 앞뒤 레이어는 제외
        dampening_frac=0.01,
    )
]

# 양자화 실행
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# 4. 저장 및 압축 (제출용 파일 생성)
print(">>> [4/4] 결과물 저장 및 압축 중...")

# 기존 폴더 청소
if os.path.exists(SUBMIT_DIR):
    shutil.rmtree(SUBMIT_DIR)
os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 저장 (vLLM 호환 압축 저장)
model.save_pretrained(MODEL_DIR, save_compressed=True)
tokenizer.save_pretrained(MODEL_DIR)

# submit 폴더의 내용물(model 폴더)을 압축합니다.
shutil.make_archive("submission", "zip", root_dir=SUBMIT_DIR)

print("완료")