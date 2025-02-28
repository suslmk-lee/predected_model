# Predicted Model Project

이 프로젝트는 Python을 사용한 예측 모델 개발을 위한 환경입니다.

## 환경 설정 및 구성

### 시스템 요구사항
- Python 3.9.6 이상
- macOS 환경

### 프로젝트 구조
- `venv/`: Python 가상 환경
- `src/`: 소스 코드 디렉토리
  - `model.py`: 예측 모델 클래스 구현
  - `generate_sample_data.py`: 샘플 데이터 생성 스크립트
- `data/`: 데이터 파일 디렉토리
  - `sample_data.csv`: 생성된 샘플 데이터
- `notebooks/`: Jupyter 노트북 디렉토리
  - `model_example.ipynb`: 모델 사용 예제 노트북
- `requirements.txt`: 필요한 Python 패키지 목록

## 설치 및 설정 방법

### 1. 가상 환경 생성
```bash
# 프로젝트 디렉토리에서 가상 환경 생성
python3 -m venv venv
```

### 2. 가상 환경 활성화
```bash
# macOS/Linux
source venv/bin/activate
```

### 3. 필요한 패키지 설치
```bash
# 가상 환경이 활성화된 상태에서
pip install -r requirements.txt
```

### 4. Jupyter 노트북 설치 (선택사항)
```bash
# 가상 환경이 활성화된 상태에서
pip install jupyter
```

## 사용 방법

### 샘플 데이터 생성
```bash
# 가상 환경이 활성화된 상태에서
python src/generate_sample_data.py
```

### 예측 모델 실행
```bash
# 가상 환경이 활성화된 상태에서
python src/model.py
```

### Jupyter 노트북 실행
```bash
# 가상 환경이 활성화된 상태에서
jupyter notebook
```
또는 특정 노트북 실행:
```bash
jupyter notebook notebooks/model_example.ipynb
```

## 주요 기능

### 예측 모델 클래스 (`model.py`)
- 데이터 로드 및 전처리
- 모델 학습 및 평가
- 결과 시각화
- 새로운 데이터에 대한 예측

### 샘플 데이터 생성 (`generate_sample_data.py`)
- 선형 관계를 가진 샘플 데이터 생성
- CSV 파일로 저장

### Jupyter 노트북 예제 (`model_example.ipynb`)
- 데이터 탐색 및 시각화
- 모델 학습 및 평가
- 새로운 데이터에 대한 예측 예제

## 개발 환경 설정 과정

이 프로젝트는 다음과 같은 과정으로 설정되었습니다:

1. Python 버전 확인
   ```bash
   python3 --version
   # Python 3.9.6
   ```

2. 가상 환경 생성
   ```bash
   python3 -m venv venv
   ```

3. 가상 환경 활성화
   ```bash
   source venv/bin/activate
   ```

4. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

5. 프로젝트 디렉토리 구조 생성
   ```bash
   mkdir -p src data notebooks
   ```

6. 샘플 데이터 생성
   ```bash
   python src/generate_sample_data.py
   ```

7. Jupyter 설치 (선택사항)
   ```bash
   pip install jupyter
   ```

## 모델 개선 방향

1. 특성 선택 및 엔지니어링
2. 하이퍼파라미터 튜닝
3. 다양한 모델 시도 (Random Forest, Gradient Boosting 등)
4. 교차 검증 적용

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.


```sh
docker build -t 44ce789b-kr1-registry.container.nhncloud.com/container-platform-registry/predicted_model .
docker push 44ce789b-kr1-registry.container.nhncloud.com/container-platform-registry/predicted_model
```

```sh
kubectl create secret docker-registry ncr-secret \
  --docker-server=44ce789b-kr1-registry.container.nhncloud.com/container-platform-registry \
  --docker-username=1aXCA1Oj0FqA8OMhalOUhK2b \
  --docker-password=3czGgxo7Xlpk5ojY7v4d \
  -n suslmk-ns
```
