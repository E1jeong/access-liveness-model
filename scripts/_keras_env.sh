# run_keras_*.sh 공용 프리앰블 — source 전용, 직접 실행하지 않는다.
# 사용: source scripts/_keras_env.sh "실행"   (또는 "실행됩니다" / "변환됩니다")
# 호출 스크립트가 이미 repo 루트로 cd한 뒤에 source해야 한다.

export LD_LIBRARY_PATH="$(find .venv-tf/lib -path "*/nvidia/*/lib" -type d | tr '\n' ':')"

_keras_env_msg="${1:-실행}"

echo "=== GPU 상태 확인 ==="
.venv-tf/bin/python - <<EOF
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpus if gpus else '없음 (CPU로 ${_keras_env_msg})'}")
EOF

echo ""

unset _keras_env_msg
