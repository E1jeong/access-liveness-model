import os
import random
import numpy as np
import hashlib
import json
from common.classes import BASE_CLASS_NAMES, BONA_FIDE_CLASS_INDICES, CLASS_NAMES, CLASS_MAPPING

FIXED_SPLITS = ("train", "validation", "test")
REQUIRED_SPLITS = ("train", "validation")
OPTIONAL_SPLITS = ("test",)
LIVE_CONDITIONS = ("high", "medium", "off")
ENVIRONMENT_CONDITIONS = ("indoor", "outdoor")


# 한 클래스 폴더 안의 subject 디렉터리 목록을 번호순으로 반환한다.
# live는 high/medium/off, 나머지 클래스는 indoor/outdoor 조건 폴더를 사용한다.
# 기존 비-live 평면 구조도 마이그레이션 전까지 허용하지만 새 구조와 섞지는 않는다.
def _sort_subject_dirs(cat_path, category):
    prefix = f"{category}_"
    conditions = LIVE_CONDITIONS if category == "live" else ENVIRONMENT_CONDITIONS
    root_dirs = [
        d for d in os.listdir(cat_path)
        if os.path.isdir(os.path.join(cat_path, d))
    ]
    direct_subjects = [d for d in root_dirs if d.startswith(prefix)]
    condition_dirs = [condition for condition in conditions if condition in root_dirs]
    unexpected_dirs = [
        d for d in root_dirs
        if d not in conditions and d not in direct_subjects
    ]
    if unexpected_dirs:
        raise ValueError(
            f"지원하지 않는 조건 폴더가 있습니다: {cat_path} -> {sorted(unexpected_dirs)}"
        )
    if direct_subjects and condition_dirs:
        raise ValueError(f"평면/조건별 구조가 섞여 있습니다: {cat_path}")
    if category == "live" and direct_subjects:
        raise ValueError(f"live subject는 품질 조건 폴더 아래에 있어야 합니다: {cat_path}")

    if direct_subjects:
        try:
            return sorted(direct_subjects, key=lambda x: int(x[len(prefix):]))
        except ValueError:
            return sorted(direct_subjects)

    subdirs = []
    for condition in conditions:
        condition_path = os.path.join(cat_path, condition)
        if not os.path.isdir(condition_path):
            continue
        nested_dirs = [
            d for d in os.listdir(condition_path)
            if os.path.isdir(os.path.join(condition_path, d))
        ]
        unexpected_subjects = [d for d in nested_dirs if not d.startswith(prefix)]
        if unexpected_subjects:
            raise ValueError(
                f"지원하지 않는 subject 폴더가 있습니다: "
                f"{condition_path} -> {sorted(unexpected_subjects)}"
            )
        matched = [os.path.join(condition, d) for d in nested_dirs]
        try:
            matched = sorted(matched, key=lambda x: int(os.path.basename(x)[len(prefix):]))
        except ValueError:
            matched = sorted(matched)
        subdirs.extend(matched)
    return subdirs



# 한 subject 안의 프레임 폴더("0", "1", "2", ...)를 숫자 순으로 정렬한다.
# 이름이 숫자가 아니면 예외를 잡고 문자열 정렬로 물러선다(순서만 결정적이면 된다).
def _sort_frame_dirs(subject_path):
    subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
    try:
        return sorted(subdirs, key=lambda x: int(x))
    except ValueError:
        return sorted(subdirs)


def collect_split_items(data_dir="dataset/raw", split="train"):
    """고정 split 하나에서 (cropRGB, cropIR, label) 항목을 수집한다.

    디렉터리 구조:
      live: data_dir/{split}/live/{high,medium,off}/{subject}/{frame}/cropRGB.bmp
      기타: data_dir/{split}/{category}/{indoor,outdoor}/{subject}/{frame}/cropRGB.bmp
            (기존 평면 {category}/{subject}/{frame}도 지원)
    반환: [(rgb_path, ir_path, label_int), ...]

    결과 순서는 클래스 → subject 번호 → frame 번호 순으로 완전히 결정적이다.
    이 결정성이 run_metadata의 split 해시와 증강 시드 재현성의 전제가 된다.
    (대신 클래스별로 뭉쳐 나오므로, 학습용으로 쓰기 전에 반드시 셔플해야 한다.)

    누락은 조용히 넘기지 않고 전부 예외로 던진다 — 데이터 일부가 빠진 채로 학습이
    시작되면 나중에 지표만 보고는 원인을 찾을 수 없기 때문이다.
    """
    if split not in FIXED_SPLITS:
        raise ValueError(f"split은 {FIXED_SPLITS} 중 하나여야 합니다: {split}")

    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"고정 split 디렉터리가 없습니다: {split_dir}")

    items = []
    # 원본 12개 폴더를 그대로 읽고 촬영 조건으로 24개 출력 라벨을 결정한다.
    for category in BASE_CLASS_NAMES:
        cat_path = os.path.join(split_dir, category)
        if not os.path.isdir(cat_path):
            raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")

        # live 품질 조건과 나머지 클래스의 촬영 환경 조건을 모두 훑는다.
        subdirs = _sort_subject_dirs(cat_path, category)
        if not subdirs:
            raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")

        category_items = []
        for subject_dir in subdirs:
            condition = os.path.dirname(subject_dir)
            class_name = category + ("_outdoor" if condition in ("outdoor", "off") else "")
            category_items.extend(gather_frame_items(cat_path, [subject_dir], CLASS_MAPPING[class_name]))
        if not category_items:
            raise ValueError(f"프레임이 비어 있습니다: {cat_path}")
        items.extend(category_items)

    return items


def subsample_split_items(items, max_samples, seed=42):
    """평가용으로 클래스 순 정렬 목록에서 대표 subsample을 뽑는다.

    collect_split_items 결과는 클래스별로 뭉쳐 있어서, 앞에서부터 자른 접두사는
    앞쪽 클래스만 평가하는 셈이 된다. 고정 시드로 라벨 순서와 라벨별 목록을 셔플한 뒤
    라운드로빈으로 뽑아 모든 클래스가 최대한 균등하게 섞이도록 하고,
    같은 입력에는 항상 같은 부분집합이 나오게 한다.
    """
    if max_samples >= len(items):
        return list(items)

    rng = random.Random(seed)
    by_label = {}
    for item in items:
        by_label.setdefault(item[2], []).append(item)
    labels = sorted(by_label)
    rng.shuffle(labels)
    for values in by_label.values():
        rng.shuffle(values)

    selected = []
    positions = {label: 0 for label in labels}
    while len(selected) < max_samples:
        added = False
        for label in labels:
            if positions[label] >= len(by_label[label]):
                continue
            selected.append(by_label[label][positions[label]])
            positions[label] += 1
            added = True
            if len(selected) == max_samples:
                break
        if not added:
            break
    return selected


def is_optional_split_empty(data_dir, split):
    """선택적 split(test)이 없거나 하위 디렉터리가 전혀 없는 빈 상태인지 확인한다."""
    if split not in OPTIONAL_SPLITS:
        return False
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        return True
    return not any(
        os.path.isdir(os.path.join(split_dir, d)) for d in os.listdir(split_dir)
    )


def validate_fixed_split_coverage(data_dir="dataset/raw"):
    """고정 split의 완전성과 frame/content/metadata 누수를 검사한다.

    subject 폴더 번호는 촬영 조건과 split별 독립 ID이며 인물 신원 근거가 아니다.
    명시적인 metadata groupId 계약이 생기기 전에는 폴더명이나 정렬 순서로 동일 인물을
    추론하지 않고, 명시적으로 준비된 fixed split 배치를 신뢰한다.

    학습 시작 직전에 training/train.py의 main()이 호출한다. 같은 프레임이 train과
    validation에 동시에 있으면 검증 지표가 부풀려져 실제 성능보다 좋아 보인다.
    그런 결과로 모델을 선택하면 현장에서 그대로 실패하므로, 아래 세 겹으로 검사하고
    하나라도 걸리면 예외를 던져 학습 자체를 막는다.

      1) realpath leakage — 심볼릭 링크를 풀었을 때 같은 파일을 가리키는가
      2) content leakage  — 경로가 달라도 파일 내용(MD5)이 같은가 (복사본 탐지)
      3) metadata leakage — meta.json의 session/video ID가 두 split에 걸쳐 있는가

    반환: {split: 프레임 수} — 호출부가 로그로 출력한다.
    """
    split_items = {}

    # 필수 split(train, validation)과 존재하는 선택적 split(test)의 프레임 목록을 읽어 둔다.
    for split in FIXED_SPLITS:
        if is_optional_split_empty(data_dir, split):
            continue
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"고정 split 디렉터리가 없습니다: {split_dir}")

        for category in BASE_CLASS_NAMES:
            cat_path = os.path.join(split_dir, category)
            if not os.path.isdir(cat_path):
                raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")
            subdirs = _sort_subject_dirs(cat_path, category)
            if not subdirs:
                raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")
        split_items[split] = collect_split_items(data_dir, split)
        if split in REQUIRED_SPLITS:
            present = {label for _, _, label in split_items[split]}
            missing = [name for label, name in enumerate(CLASS_NAMES) if label not in present]
            if missing:
                raise ValueError(f"{split} split에 클래스 프레임이 없습니다: {missing}")

    # 1. Realpath leakage 검사
    # split을 심볼릭 링크로 구성한 경우, 경로 문자열은 달라도 실체가 같은 파일일 수 있다.
    # realpath로 링크를 모두 풀어 정규화한 뒤 집합 교집합으로 비교한다.
    resolved_paths = {}
    for split, items in split_items.items():
        paths = {
            os.path.realpath(path)
            for rgb_path, ir_path, _ in items
            for path in (rgb_path, ir_path)
        }
        # 이미 처리한 split들과만 비교하면 되므로(교집합은 대칭) 중복 비교가 없다.
        for other_split, other_paths in resolved_paths.items():
            overlap = paths & other_paths
            if overlap:
                example = sorted(overlap)[0]
                raise ValueError(
                    f"{other_split}/{split} split에 동일 프레임 실경로가 있습니다: {example}"
                )
        resolved_paths[split] = paths

    # 2. Content MD5 Hash Leakage 검사
    # 1번(realpath)을 통과해도, 파일을 물리적으로 복사해 두 split에 넣었다면 잡히지 않는다.
    # 그래서 내용 자체를 해시해 비교한다. 여기서는 보안이 아니라 중복 탐지가 목적이라
    # 속도가 빠른 MD5로 충분하다. 전 이미지 내용을 읽으므로 데이터 크기에 비례해 비용이 든다.
    hash_to_split_paths = {}
    for split, items in split_items.items():
        for rgb_path, ir_path, _ in items:
            for file_path in (rgb_path, ir_path):
                real_p = os.path.realpath(file_path)
                if not os.path.exists(real_p):
                    continue
                # MD5 해시 계산
                # 64KB씩 나눠 읽어 대용량 파일에서도 메모리를 일정하게 유지한다.
                hasher = hashlib.md5()
                with open(real_p, 'rb') as f:
                    for chunk in iter(lambda: f.read(65536), b''):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()

                # 같은 해시가 '다른' split에서 이미 나왔다면 동일 내용 프레임이 양쪽에 있다는 뜻.
                # 같은 split 안의 중복은 누수가 아니므로 통과시킨다.
                if file_hash in hash_to_split_paths:
                    existing_split, existing_path = hash_to_split_paths[file_hash]
                    if existing_split != split:
                        raise ValueError(
                            f"Content hash leakage detected! "
                            f"File in split '{split}' ({file_path}) has identical content to "
                            f"file in split '{existing_split}' ({existing_path})."
                        )
                else:
                    hash_to_split_paths[file_hash] = (split, file_path)

    # 3. Explicit Manifest / Metadata Leakage 검사 (meta.json)
    # 같은 촬영 세션·같은 영상에서 뽑은 서로 다른 프레임은 파일 내용이 달라 1~2번을 모두
    # 통과하지만, 사실상 같은 데이터다(조명·표정·배경이 거의 동일). meta.json이 있으면
    # session/video ID로 그 관계를 직접 확인한다.
    # device와 attack_medium은 원래 여러 split에 공유되는 게 정상이라 로그만 남긴다.
    session_to_splits = {}
    video_to_splits = {}
    device_to_splits = {}
    attack_medium_to_splits = {}

    for split, items in split_items.items():
        for rgb_path, _, _ in items:
            frame_dir = os.path.dirname(rgb_path)
            meta_path = os.path.join(frame_dir, "meta.json")
            # meta.json은 선택 사항이다. 없으면 이 검사만 건너뛴다(1~2번은 이미 통과한 상태).
            if os.path.exists(meta_path):
                # 단, 존재하는 meta.json이 깨져 있으면 session/video 검사를 우회하는 셈이므로
                # 조용히 넘기지 않고 검증 자체를 실패시킨다.
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ValueError(f"meta.json이 손상되었습니다: {meta_path} ({e})") from e
                if not isinstance(meta_data, dict):
                    raise ValueError(
                        f"meta.json은 JSON object여야 합니다: {meta_path} ({type(meta_data).__name__})"
                    )

                session_val = None
                video_val = None
                device_val = None
                attack_val = None

                # 키 표기가 파일마다 제각각이라(session_id / sessionId / session)
                # 소문자로 낮춘 뒤 알려진 별칭들을 모두 받아 준다.
                for k, v in meta_data.items():
                    k_lower = k.lower()
                    if k_lower in ("session", "session_id", "sessionid"):
                        session_val = v
                    elif k_lower in ("video", "video_id", "videoid"):
                        video_val = v
                    elif k_lower in ("device", "device_id", "deviceid"):
                        device_val = v
                    elif k_lower in ("attack_medium", "attackmedium", "attack_type", "attacktype"):
                        attack_val = v

                # Session leakage 검사
                if session_val is not None:
                    session_to_splits.setdefault(session_val, {}).setdefault(split, []).append(meta_path)
                # Video leakage 검사
                if video_val is not None:
                    video_to_splits.setdefault(video_val, {}).setdefault(split, []).append(meta_path)
                # Device 정보 수집 (로깅)
                if device_val is not None:
                    device_to_splits.setdefault(device_val, {}).setdefault(split, []).append(meta_path)
                # Attack Medium 정보 수집 (로깅)
                if attack_val is not None:
                    attack_medium_to_splits.setdefault(attack_val, {}).setdefault(split, []).append(meta_path)

    # Session, Video Leakage 에러 처리
    for session_val, splits_dict in session_to_splits.items():
        if len(splits_dict) > 1:
            example_paths = {s: paths[0] for s, paths in splits_dict.items()}
            raise ValueError(
                f"Session leakage detected! Session '{session_val}' found in multiple splits: "
                f"{sorted(splits_dict.keys())}. Example paths: {example_paths}"
            )

    for video_val, splits_dict in video_to_splits.items():
        if len(splits_dict) > 1:
            example_paths = {s: paths[0] for s, paths in splits_dict.items()}
            raise ValueError(
                f"Video leakage detected! Video '{video_val}' found in multiple splits: "
                f"{sorted(splits_dict.keys())}. Example paths: {example_paths}"
            )

    # 로깅 출력 (Device, Attack Medium)
    for device_val, splits_dict in device_to_splits.items():
        if len(splits_dict) > 1:
            print(f"[Metadata Info] Device '{device_val}' is shared across splits: {sorted(splits_dict.keys())}")
    for attack_val, splits_dict in attack_medium_to_splits.items():
        if len(splits_dict) > 1:
            print(f"[Metadata Info] Attack Medium '{attack_val}' is shared across splits: {sorted(splits_dict.keys())}")

    return {split: len(split_items.get(split, ())) for split in FIXED_SPLITS}


def gather_frame_items(cat_path, subdirs_list, label):
    """subject 폴더 목록에서 (rgb_path, ir_path, label) 튜플 리스트를 수집한다.

    subject 폴더 순서(호출부에서 정렬됨) × 프레임 번호 오름차순으로 쌓으므로 결과가 결정적이다.
    """
    gathered = []
    for sd in subdirs_list:
        subject_path = os.path.join(cat_path, sd)
        for frame_id in _sort_frame_dirs(subject_path):
            frame_path = os.path.join(subject_path, frame_id)
            # 학습에 실제로 쓰는 것은 얼굴 크롭 두 장뿐이다.
            rgb_path = os.path.join(frame_path, "cropRGB.bmp")
            ir_path = os.path.join(frame_path, "cropIR.bmp")
            # 원본 전체 프레임은 학습에 쓰지 않지만 존재 여부는 확인한다.
            # 네 파일이 한 세트여야 정상 수집된 프레임이고, 하나라도 없으면
            # 데이터 준비 단계가 중간에 끊긴 것이므로 조용히 넘기지 않고 실패시킨다.
            raw_rgb_path = os.path.join(frame_path, "RGB.bmp")
            raw_ir_path = os.path.join(frame_path, "IR.bmp")
            required = [rgb_path, ir_path, raw_rgb_path, raw_ir_path]
            if not all(os.path.exists(p) for p in required):
                raise FileNotFoundError(f"필수 BMP 파일이 누락되었습니다: {frame_path}")
            gathered.append((rgb_path, ir_path, int(label)))
    return gathered


def quantize_for_tflite(arr, detail):
    """인터프리터의 텐서 정보에 따라 ``arr``를 int8/uint8로 양자화한다.

    float 자료형이면 값 변환 없이 float32 형식으로만 맞춘다.
    """
    dtype = detail['dtype']
    if dtype not in (np.int8, np.uint8):
        return arr.astype(np.float32)
    scale, zero_point = detail['quantization']
    if scale == 0.0:
        scale = 1.0
    q = np.round(arr / scale) + zero_point
    info = np.iinfo(dtype)
    return np.clip(q, info.min, info.max).astype(dtype)


def dequantize_from_tflite(arr, detail):
    dtype = detail['dtype']
    if dtype not in (np.int8, np.uint8):
        return arr.astype(np.float32)
    scale, zero_point = detail['quantization']
    return (arr.astype(np.float32) - zero_point) * scale


def calculate_validation_metrics(labels, preds):
    """혼동 행렬, 클래스별 Recall, APCER/BPCER/ACER를 계산한다.

    학습(AcerCheckpoint)과 TFLite 평가(evaluate_tflite.py)가 같은 이 함수를 쓴다
    → 두 단계의 숫자를 그대로 비교할 수 있다.

    12-클래스 분류 결과를 bona-fide vs attack 2진 관점으로 눌러서 본다.
      bona-fide = live, dental_white, dental_black
      attack    = 나머지 9개 클래스
      APCER = 공격을 bona-fide로 통과시킨 비율          ← 보안 사고에 직결
      BPCER = bona-fide를 attack으로 거부한 비율            ← 사용성 저하
      ACER  = 두 값의 단순 평균 (현재 체크포인트 선택 기준, 낮을수록 좋음)

    구체 예) 검증셋에 bona-fide 100장, attack 900장이 있고 attack 중 9장을
    bona-fide로, bona-fide 중 5장을 attack으로 틀렸다면
      APCER = 9/900 = 0.01, BPCER = 5/100 = 0.05, ACER = (0.01+0.05)/2 = 0.03.
    같은 상황에서 accuracy는 986/1000 = 0.986으로 아주 좋아 보이지만,
    ACER는 BPCER 쪽 문제를 그대로 드러낸다. 이것이 accuracy 대신 ACER로 모델을 고르는 이유다.

    같은 pass 집합 또는 attack 집합 안에서 세부 클래스를 혼동하는 것은
    APCER/BPCER에 영향을 주지 않는다. 그 세부는 혼동행렬로 본다.
    """
    num_classes = len(CLASS_NAMES)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    if labels.ndim != 1 or preds.ndim != 1:
        raise ValueError("labels와 preds는 1차원이어야 합니다.")
    if len(labels) != len(preds):
        raise ValueError("labels와 preds의 길이가 같아야 합니다.")
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError(f"labels는 0 이상 {num_classes - 1} 이하여야 합니다.")
    if np.any((preds < 0) | (preds >= num_classes)):
        raise ValueError(f"preds는 0 이상 {num_classes - 1} 이하여야 합니다.")

    # 행=정답, 열=예측. 대각선이 맞힌 것, 나머지가 틀린 것.
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        confusion_matrix[int(label), int(pred)] += 1

    # 클래스별 recall = 해당 행의 대각 원소 / 행 합계.
    # 그 클래스 샘플이 하나도 없으면 0.0으로 둔다(0으로 나누지 않기 위함).
    recalls = []
    for class_idx in range(num_classes):
        total = confusion_matrix[class_idx, :].sum()
        correct = confusion_matrix[class_idx, class_idx]
        recalls.append(float(correct / total) if total > 0 else 0.0)

    bona_fide_labels = np.isin(labels, BONA_FIDE_CLASS_INDICES)
    bona_fide_preds = np.isin(preds, BONA_FIDE_CLASS_INDICES)
    attack_labels = ~bona_fide_labels
    total_bona_fide = int(bona_fide_labels.sum())
    total_attacks = int(attack_labels.sum())
    # 정답은 attack인데 pass 집합으로 예측 → 공격 통과
    apcer_errors = int((bona_fide_preds & attack_labels).sum())
    # 정답은 bona-fide인데 attack으로 예측 → 정상 사용자 거부
    bpcer_errors = int(((~bona_fide_preds) & bona_fide_labels).sum())

    apcer = apcer_errors / total_attacks if total_attacks > 0 else 0.0
    bpcer = bpcer_errors / total_bona_fide if total_bona_fide > 0 else 0.0
    # 단순 평균이라 bona-fide/attack 샘플 수 불균형의 영향을 받지 않는다.
    acer = (apcer + bpcer) / 2.0
    return confusion_matrix, recalls, apcer, bpcer, acer
