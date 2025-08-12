#!/bin/bash

# GPU 순차 할당 초기화
current_gpu=0  # 전역 변수로 초기화

# GPU 상태 확인 함수 (GPU 개수를 확인하여 순차적으로 순회)
function get_next_gpu() {
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)  # GPU 개수 확인
    local next_gpu=$((current_gpu % gpu_count))  # 현재 GPU의 인덱스 계산
    echo $next_gpu
}

# 훈련 프로세스 함수
function train_process() {
    subject=$1
    gpu_num=$2
    seed=$3

    echo "Starting training: subject=$subject, gpu_num=$gpu_num, seed=$seed"
    # Python 프로세스를 실행하면서 환경 변수 직접 전달
    python training.py --target_subject=$subject --gpu_num=$gpu_num --seed=$seed

    if [ $? -eq 0 ]; then
        echo "Completed training: subject=$subject, gpu_num=$gpu_num, seed=$seed"
    else
        echo "Failed training: subject=$subject, gpu_num=$gpu_num, seed=$seed"
    fi
}

# 훈련 시드 및 subject 설정
seeds=(2 12 22 32 42)  # 여러 시드 값
subject_list=(0 1 2 3 4 5 6 7 8)  # subject 리스트

# 메인 스크립트
for seed in "${seeds[@]}"; do
    for subject in "${subject_list[@]}"; do
        gpu_num=$(get_next_gpu)  # GPU를 순차적으로 선택
        echo "Assigned GPU: $gpu_num for subject: $subject and seed: $seed"  # 디버깅 로그 추가
        current_gpu=$((current_gpu + 1))  # 전역 변수 업데이트
        train_process $subject $gpu_num $seed &  # 비동기로 작업 실행
        sleep 2  # 작업 간 시간 차를 둬 GPU 과부하 방지
        # wait
    done
    wait # 모든 시드에 대한 훈련이 완료될 때까지 대기
done

# 모든 백그라운드 작업 완료 대기
wait
echo "All tasks completed."