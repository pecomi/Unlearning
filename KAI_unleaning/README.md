# KAI Unlearning Research Framework


### 1. 환경 설정

```bash
# Conda 환경 생성 (권장)
conda env create -f environment.yml
conda activate kai-unlearning

# 또는 pip 설치
pip install -r requirements.txt
```

### 2. 설정

`config_{pretrain/full_class/label_based/random}.yaml` 파일에서 필요한 파라미터를 수정


### 3. 실행

```bash
# 전체 파이프라인 실행 (학습 -> Unlearning -> 평가)
python src/main.py --config config.yaml --mode full

# 단계별 실행
python src/main.py --config config.yaml --mode train
python src/main.py --config config.yaml --mode unlearn
python src/main.py --config config.yaml --mode evaluate
```

## 지원하는 구성 요소

### 데이터셋
- CIFAR-10 (기본)
- CIFAR-100
- 확장 가능: TOFU, MUSE, Fashion MNIST

### 모델
- ResNet-18 (기본)
- ResNet-34
- 확장 가능: LLM, Regression 모델

### Unlearning 알고리즘
- SSD (Selective Synaptic Dampening)
- 확장 가능: SCRUB, Influence-based, Gradient-ascent

## 평가 메트릭

### Model Utility
- **Retain Accuracy**: 보존 데이터에 대한 정확도
- **Test Accuracy**: 테스트 데이터에 대한 정확도

### Forget Quality
- **Forget Accuracy**: 망각 데이터에 대한 정확도 (낮을수록 좋음)

## 확장 방법

### 새로운 데이터셋 추가

```python
# src/data/mydataset.py
from .base import BaseDataset

class MyDataset(BaseDataset):
    name = "mydataset"

    def load(self):
        # 데이터 로드 구현
        pass

    def create_splits(self, ...):
        # 데이터 분할 구현
        pass

# src/data/factory.py에서 등록
DatasetFactory.register("mydataset", MyDataset)
```

### 새로운 모델 추가

```python
# src/models/mymodel.py
from .base import BaseModel

class MyModel(BaseModel):
    name = "mymodel"

    def create_model(self):
        # 모델 생성 구현
        pass

# src/models/factory.py에서 등록
ModelFactory.register("mymodel", MyModel)
```

### 새로운 Unlearning 알고리즘 추가

```python
# src/unlearning/myalgorithm.py
from .base import BaseUnlearning

class MyUnlearning(BaseUnlearning):
    name = "myalgorithm"

    def unlearn(self, forget_loader, retain_loader, **kwargs):
        # Unlearning 알고리즘 구현
        pass

# src/unlearning/factory.py에서 등록
UnlearningFactory.register("myalgorithm", MyUnlearning)
```

## 참고 문헌

- **SSD**: Golatkar, A., Achille, A., & Soatto, S. (2020). "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks."

## 라이선스

MIT License
