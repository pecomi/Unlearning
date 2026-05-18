#!/bin/bash
# KAI Unlearning - 모든 태스크 실행 스크립트
# 세 가지 forgetting 태스크를 순차적으로 실행 (단일 GPU 환경)

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_header() {
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
}

# 시작 시간 기록
START_TIME=$(date +%s)

echo ""
log_header "KAI Unlearning - 전체 실험 실행"
log_info "시작 시간: $(date)"
echo ""

# ============================================================================
# STEP 0: 사전 체크
# ============================================================================
log_step "0. 사전 체크"

# Baseline 모델 존재 확인
# Note: Pretrain mode는 timestamp를 사용하지 않고 직접 저장됨
if [ ! -f "./checkpoints/baseline/trained_model.pth" ]; then
    log_warning "Baseline 모델이 존재하지 않습니다. Pretrain을 먼저 실행합니다."
    NEED_PRETRAIN=true
else
    log_success "Baseline 모델 확인 완료: ./checkpoints/baseline/trained_model.pth"
    NEED_PRETRAIN=false
fi

echo ""

# ============================================================================
# STEP 1: Baseline Pretrain (필요한 경우만)
# ============================================================================
if [ "$NEED_PRETRAIN" = true ]; then
    log_header "STEP 1: Baseline Pretrain"
    log_info "모든 태스크가 공유할 단일 baseline 모델을 학습합니다..."
    echo ""

    python src/main.py --config config_pretrain.yaml --mode train

    log_success "Baseline pretrain 완료!"
    echo ""
fi

# ============================================================================
# STEP 2: Unlearning (각 태스크 순차 실행)
# ============================================================================
log_header "STEP 2: Unlearning Phase"
log_info "각 태스크별로 unlearning을 실행합니다..."
echo ""

# 2.1 Full-class Forgetting
log_step "2.1 Full-class Forgetting - Unlearning"
python src/main.py --config config_full_class.yaml --mode unlearn
log_success "Full-class unlearning 완료"
echo ""

# 2.2 Label-based Forgetting
log_step "2.2 Label-based Forgetting - Unlearning"
python src/main.py --config config_label_based.yaml --mode unlearn
log_success "Label-based unlearning 완료"
echo ""

# 2.3 Random Sample Forgetting
log_step "2.3 Random Sample Forgetting - Unlearning"
python src/main.py --config config_random.yaml --mode unlearn
log_success "Random sample unlearning 완료"
echo ""

# ============================================================================
# STEP 3: Retrain (Gold Standard)
# ============================================================================
log_header "STEP 3: Retrain Phase (Gold Standard)"
log_info "각 태스크별로 retain set만으로 재학습합니다..."
echo ""

# 3.1 Full-class Forgetting
log_step "3.1 Full-class Forgetting - Retrain"
python src/main.py --config config_full_class.yaml --mode retrain
log_success "Full-class retrain 완료"
echo ""

# 3.2 Label-based Forgetting
log_step "3.2 Label-based Forgetting - Retrain"
python src/main.py --config config_label_based.yaml --mode retrain
log_success "Label-based retrain 완료"
echo ""

# 3.3 Random Sample Forgetting
log_step "3.3 Random Sample Forgetting - Retrain"
python src/main.py --config config_random.yaml --mode retrain
log_success "Random sample retrain 완료"
echo ""

# ============================================================================
# STEP 4: Comparison
# ============================================================================
log_header "STEP 4: Comparison Phase"
log_info "Unlearned vs Retrained 모델을 비교합니다..."
echo ""

# 4.1 Full-class Forgetting
log_step "4.1 Full-class Forgetting - Compare"
python src/main.py --config config_full_class.yaml --mode compare
log_success "Full-class comparison 완료"
echo ""

# 4.2 Label-based Forgetting
log_step "4.2 Label-based Forgetting - Compare"
python src/main.py --config config_label_based.yaml --mode compare
log_success "Label-based comparison 완료"
echo ""

# 4.3 Random Sample Forgetting
log_step "4.3 Random Sample Forgetting - Compare"
python src/main.py --config config_random.yaml --mode compare
log_success "Random sample comparison 완료"
echo ""

# ============================================================================
# 완료
# ============================================================================
log_header "모든 실험 완료!"

# 총 소요 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_success "총 소요 시간: ${HOURS}시간 ${MINUTES}분 ${SECONDS}초"
log_info "완료 시간: $(date)"
echo ""

log_info "결과 위치:"
echo "  - Baseline: ./checkpoints/baseline/"
echo "  - Full-class: ./runs/full_class/"
echo "  - Label-based: ./runs/label_based/"
echo "  - Random: ./runs/random/"
echo ""

log_info "WandB에서 결과 확인: https://wandb.ai/<your-entity>/kai-unlearning"
echo ""
