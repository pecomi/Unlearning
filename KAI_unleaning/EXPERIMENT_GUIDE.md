## 실험 단계

### 1단계: Baseline Pretrain (한 번만 실행)

모든 태스크가 공유할 단일 baseline 모델을 학습합니다.

```bash
python src/main.py --config config_pretrain.yaml --mode train
```

**결과**: `./checkpoints/baseline/trained_model.pth` 생성

---

### 2단계: 각 태스크별 Unlearning

같은 baseline 모델에서 각 태스크에 맞는 forget/retain split을 적용하여 unlearning을 수행합니다.

#### 2.1 Full-Class Forgetting

모든 클래스를 forgetting (all_labels strategy)

```bash
# Unlearning
python src/main.py --config config_full_class.yaml --mode unlearn

# Retrain (Gold Standard)
python src/main.py --config config_full_class.yaml --mode retrain

# Compare
python src/main.py --config config_full_class.yaml --mode compare
```

#### 2.2 Label-Based (Single-Class) Forgetting

특정 클래스(예: class 4)를 forgetting

```bash
# Unlearning
python src/main.py --config config_label_based.yaml --mode unlearn

# Retrain (Gold Standard)
python src/main.py --config config_label_based.yaml --mode retrain

# Compare
python src/main.py --config config_label_based.yaml --mode compare
```

**다른 클래스 실험**: `config_label_based.yaml`에서 `forget_labels: [4]`를 원하는 클래스로 변경

#### 2.3 Random Sample Forgetting

무작위 10% 샘플을 forgetting

```bash
# Unlearning
python src/main.py --config config_random.yaml --mode unlearn

# Retrain (Gold Standard)
python src/main.py --config config_random.yaml --mode retrain

# Compare
python src/main.py --config config_random.yaml --mode compare
```

---

## Config 파일 설명

| 파일 | 용도 | 주요 설정 |
|------|------|-----------|
| `config_pretrain.yaml` | Baseline 학습 | `checkpoint_dir: ./checkpoints/baseline` |
| `config_full_class.yaml` | Full-class forgetting | `split_strategy: all_labels` |
| `config_label_based.yaml` | Single-class forgetting | `split_strategy: label_based`, `forget_labels: [4]` |
| `config_random.yaml` | Random sample forgetting | `split_strategy: random`, `forget_ratio: 0.1` |