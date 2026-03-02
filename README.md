# LG-Aimers-8th-Model-Lightweight-Project
# LG Aimers 해커톤: EXAONE 1.2B 모델 경량화 프로젝트

LG EXAONE 4.0 1.2B 모델을 GPTQ 기법을 사용하여 4비트로 경량화하고 최적화한 프로젝트입니다.

# 성과
- Baseline 달성: 0.5879점 (Baseline 약 0.5점 대비 17% 향상)
- 추론 시간 단축: 20분 제한 → 10분 30초로 약 50% 단축
- 경량화: 2.4GB → 800MB (W4A16 Quantization 적용)

# 사용 기술
- Model: LGAI-EXAONE/EXAONE-4.0-1.2B
- Method: GPTQ (Samples: 128, Seq Len: 512)
- Environment: Local PC (CPU Inference Optimization)

# 라이선스 고지
본 프로젝트는 LG AI Research의 EXAONE 모델을 기반으로 합니다. 
라이선스 및 용량 문제로 모델 가중치(Weights) 파일은 포함되어 있지 않습니다.
