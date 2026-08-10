#!/usr/bin/env bash
# 호환용 진입점. 신규 학습은 K-Fold 대신 고정 split 파이프라인을 사용한다.
set -e
cd "$(dirname "$0")/../.."
echo "[안내] K-Fold 일괄 실행은 중단되었습니다. scripts/keras/run_fixed_split.sh를 실행합니다."
exec ./scripts/keras/run_fixed_split.sh "$@"
