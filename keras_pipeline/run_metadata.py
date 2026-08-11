"""학습 실행 기록(run metadata) 생성.

"이 .keras 파일이 어떤 설정·어떤 데이터로 나왔는가"를 JSON 한 장으로 재구성할 수 있게 한다.
모델 파일 자체에는 하이퍼파라미터도 데이터 구성도 남지 않기 때문에, 이 기록이 없으면
몇 주 뒤 산출물끼리 비교가 불가능해진다.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from classes import CLASS_NAMES
from utils import collect_split_items


# 실행 식별자: UTC 시각 + 모델 타입 (예: 20260807T100157Z_dual).
# 로컬 시간이 아니라 UTC를 쓰는 이유는 여러 머신(회사 PC/서브 노트북)의 기록을
# 시간순으로 정렬할 때 타임존 차이로 순서가 뒤집히지 않게 하기 위해서다.
def make_run_id(model_type):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{model_type}"


# split별 파일 목록의 SHA-256 지문. "그때 그 데이터가 지금 이 데이터와 같은가"를
# 나중에 한 줄 비교로 확인할 수 있게 한다(파일이 추가/삭제/재분배되면 해시가 바뀐다).
def split_hashes(data_dir):
    root = Path(data_dir)
    hashes = {}
    for split in ("train", "validation", "test"):
        items = collect_split_items(data_dir, split)
        # 절대경로가 아니라 data_dir 기준 상대경로로 바꾼다 →
        # 머신이 달라져도(회사 PC vs 서브 노트북) 같은 데이터면 같은 해시가 나온다.
        payload = [
            (str(Path(rgb).relative_to(root)), str(Path(ir).relative_to(root)), label)
            for rgb, ir, label in items
        ]
        # separators로 공백을 제거해 직렬화 형태를 고정한다(포맷이 흔들리면 해시도 흔들린다).
        # 목록 순서는 collect_split_items가 정렬해 주므로 실행마다 동일하다.
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        # 파일 '내용'이 아니라 '목록'의 해시다. 내용 중복 검사는 utils의 누수 검증이 담당한다.
        hashes[split] = hashlib.sha256(encoded).hexdigest()
    return hashes


# 학습 종료 직후 tf_train.main()이 호출한다.
def write_run_metadata(path, run_id, config, data_dir, best_checkpoint, best_metrics):
    metadata = {
        "run_id": run_id,
        "config": config,                        # CLI 인자 전체(= 하이퍼파라미터)
        "class_map": CLASS_NAMES,                # 인덱스↔클래스 대응. 앱 해석과 대조할 때 쓴다
        "split_hashes": split_hashes(data_dir),  # 데이터 구성 지문
        "best_checkpoint": str(best_checkpoint), # 이 기록이 설명하는 모델 파일
        "best_validation_metrics": best_metrics, # ACER 최저 에폭의 지표
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 + 끝 개행: git diff로 읽기 좋게. ensure_ascii=False로 한글도 그대로 저장.
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
