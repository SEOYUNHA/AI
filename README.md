# AI

🚀 SSAFY AI Portfolio: 첨단 AI 기술 및 Python 실전 구현 역량  
본 포트폴리오는 대규모 언어 모델(LLM), 전이 학습(Transfer Learning), AI 에이전트 개발, 데이터 분석 및 머신러닝 기초에 이르기까지, 현대 AI 개발에 필수적인 핵심 기술을 Python을 활용하여 직접 구현한 프로젝트 모음입니다.

---

## 🌟 핵심 역량 요약

| 영역 | 핵심 기술 및 경험 | 주요 파일 예시 |
|------|------------------|---------------|
| Generative AI & LLM | PEFT(LoRA/Unsloth)를 이용한 경량화 모델 파인튜닝, LLM 기반 합성 데이터 생성, Text-to-Image 모델(Stable Diffusion) 및 평가(CLIP) 활용 | PEFT(파라미터_효율적_튜닝)_Text2SQL_과제2.ipynb, 합성_데이터_실습.ipynb, 데이터셋_생성및파인튜닝.ipynb |
| AI Agent & RAG | ReAct 및 RAG 구조를 이용한 복잡한 질의 응답 에이전트 개발 및 Upstage API 활용, Gradio 기반 서비스 배포 및 시각화 | ReAct기반_Customer_Service_AI_에이전트개발.ipynb, RAG_기반_Customer_Service_AI_에이전트_개발 (1).ipynb, 에이전트_서비스_및_시각화_실습1.ipynb |
| 딥러닝/전이 학습 | ResNet, ViT(HuggingFace) 등 사전 학습 모델을 활용한 전이 학습 및 리니어 프로빙, PyTorch/PyTorch Lightning 기반 MLP 구현 | Transfer_Learning_기반의_CNN_모델_학습.ipynb, MLP_구현.ipynb, MLP 구현2.ipynb |
| Python & 데이터 분석 | NumPy를 활용한 선형 회귀/로지스틱 회귀 모델의 밑바닥 구현, Pandas/Seaborn을 이용한 탐색적 데이터 분석(EDA) 및 전처리 | AI를_위한_Python.ipynb, NumPy를_이용한_선형_회귀_모델_구현.ipynb, 데이터_EDA_및_모델_학습.ipynb |

---

## 📂 주요 프로젝트 상세 구현 내역

### 1. 경량화 모델 파인튜닝 (PEFT, Unsloth, LoRA)
**기술**: PEFT (Parameter-Efficient Fine-Tuning) 기법 중 LoRA를 활용하여 대규모 언어 모델을 특정 태스크(Text-to-SQL)에 맞게 효율적으로 튜닝했습니다.  
**강조 역량**: LLM의 파인튜닝 워크플로우에 대한 깊은 이해와 리소스 효율성을 고려한 모델 최적화 능력을 입증합니다.  
**관련 파일**:
- PEFT(파라미터_효율적_튜닝)_Text2SQL_과제2.ipynb
- PEFT(파라미터_효율적_튜닝)_Unsloth_실습.ipynb

---

### 2. AI 에이전트 및 서비스 개발 (ReAct, RAG, Gradio)
**기술**: ReAct (Reasoning and Acting) 프레임워크를 기반으로 고객 서비스 AI 에이전트를 개발했습니다.  
RAG (검색 증강 생성) 기술을 결합하여 외부 지식을 활용한 정확한 답변 생성을 구현했습니다.  
**강조 역량**: 복잡한 문제 해결을 위한 논리적 추론 과정 설계 (프롬프트 엔지니어링), 외부 시스템 연동 (Tool Learning), 그리고 Gradio를 이용한 웹 서비스 프로토타입 배포 능력을 보여줍니다.  
**관련 파일**:
- ReAct기반_Customer_Service_AI_에이전트개발.ipynb
- RAG_기반_Customer_Service_AI_에이전트_개발 (1).ipynb
- 에이전트_서비스_및_시각화_실습1.ipynb

---

### 3. 컴퓨터 비전 (전이 학습, HuggingFace ViT)
**기술**: 전이 학습의 핵심 개념인 리니어 프로빙을 ResNet-18 모델에 적용하고, HuggingFace의 Vision Transformer (ViT) 모델을 활용하여 최신 트랜스포머 기반의 이미지 인식 추론을 실습했습니다.  
**강조 역량**: CNN부터 최신 Transformer 구조까지 다양한 딥러닝 모델 아키텍처에 대한 실무적 이해와 사전 학습된 가중치 활용 능력을 어필합니다.  
**관련 파일**:
- Transfer_Learning_기반의_CNN_모델_학습.ipynb
- 이미지_생성_및_평가와_모델_학습1.ipynb

---

### 4. Python 기반 AI 기초 모델 구현 및 데이터 분석
**기술**: NumPy만 사용하여 선형/로지스틱 회귀 모델의 손실 함수, 경사 하강법 등 핵심 수식을 직접 구현했습니다.  
또한, Pandas와 Seaborn을 이용한 체계적인 EDA 및 전처리 워크플로우를 구축했습니다.  
**강조 역량**: Python 라이브러리 활용 능력과 더불어, 머신러닝 알고리즘의 동작 원리를 밑바닥부터 이해하고 구현할 수 있는 탄탄한 기초 역량을 보여줍니다.  
**관련 파일**:
- NumPy를_이용한_선형_회귀_모델_구현.ipynb
- 데이터_EDA_및_모델_학습.ipynb
- AI를_위한_Python.ipynb

---

## 🛠️ 기술 스택 (Tech Stack)

| 구분 | 주요 라이브러리 |
|------|----------------|
| LLM/파인튜닝 | Hugging Face ecosystem (Transformers, Datasets), PEFT (LoRA), Unsloth, Gradio, Upstage API |
| 딥러닝 | PyTorch, PyTorch Lightning, Torchvision |
| 머신러닝/통계 | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |
| 에이전트/RAG | Langchain/LangGraph (혹은 유사 프레임워크), VectorDB (ChromaDB 등), Embedding models |

---
