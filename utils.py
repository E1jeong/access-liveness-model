import os
import random
import numpy as np
import hashlib
import json
from classes import CLASS_NAMES, CLASS_MAPPING

FIXED_SPLITS = ("train", "validation", "test")


# subject 폴더 정렬 키. "live_10"에서 뒤쪽 숫자를 뽑아 숫자 순으로 세운다
# (문자열 정렬이면 live_10 < live_2가 되어 그룹핑 규칙이 어긋난다).
# 튜플 첫 원소 0/1은 "숫자 해석 성공/실패" 구분 — 형식이 다른 폴더는 뒤로 몰아 이름순 정렬한다.
def _subject_sort_key(subject):
    basename = os.path.basename(subject)
    try:
        return (0, int(basename.rsplit("_", 1)[1]))
    except (IndexError, ValueError):
        return (1, basename)


# 한 클래스 폴더 안의 subject 디렉터리 목록을 번호순으로 반환한다.
# live만 high/medium 하위 폴더를 한 단계 더 갖고 있어 분기가 필요하다
# (반환값도 "high/live_1"처럼 하위 폴더를 포함한 상대 경로가 된다).
def _sort_subject_dirs(cat_path, category):
    prefix = f"{category}_"
    if category == "live":
        subdirs = []
        for subdir in ["high", "medium"]:
            sub_path = os.path.join(cat_path, subdir)
            if os.path.exists(sub_path):
                matched = [
                    os.path.join(subdir, d) for d in os.listdir(sub_path)
                    if os.path.isdir(os.path.join(sub_path, d)) and d.startswith(prefix)
                ]
                try:
                    matched = sorted(matched, key=lambda x: int(os.path.basename(x)[len(prefix):]))
                except ValueError:
                    matched = sorted(matched)
                subdirs.extend(matched)
        return subdirs
    else:
        subdirs = [
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d)) and d.startswith(prefix)
        ]
        try:
            return sorted(subdirs, key=lambda x: int(x[len(prefix):]))
        except ValueError:
            return sorted(subdirs)



# 한 subject 안의 프레임 폴더("0", "1", "2", ...)를 숫자 순으로 정렬한다.
# 이름이 숫자가 아니면 예외를 잡고 문자열 정렬로 물러선다(순서만 결정적이면 된다).
def _sort_frame_dirs(subject_path):
    subdirs = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
    try:
        return sorted(subdirs, key=lambda x: int(x))
    except ValueError:
        return sorted(subdirs)


def _group_subject_dirs(subdirs, category):
    # subdirs는 이미 정렬되어 있는 상태입니다. (_sort_subject_dirs의 리턴값)
    # subdirs를 동일 인물(Group)로 묶습니다.
    # 누수 검증에서 "같은 물리 인물"의 단위를 정하는 함수라, 정렬 순서가 바뀌면
    # spoof 쪽 그룹 구성이 통째로 달라진다는 점에 주의.
    groups = {}
    if category == "live":
        # live의 경우: high/live_1, medium/live_1 등이 있으므로 basename에서 숫자를 추출하여 그룹화
        for sd in subdirs:
            basename = os.path.basename(sd)  # 폴더명 예: "live_1"
            try:
                group_key = int(basename.split("_")[1])
            except (IndexError, ValueError):
                group_key = basename  # 숫자를 읽지 못하면 폴더 이름을 대신 쓴다
            groups.setdefault(group_key, []).append(sd)
    else:
        # 그 외 spoof: subject 목록을 정렬한 순서대로 2개씩 묶음
        for i, sd in enumerate(subdirs):
            group_key = i // 2
            groups.setdefault(group_key, []).append(sd)
    return groups


def _split_kfold_subjects(subdirs, k_folds, fold_idx, seed, category):
    groups = _group_subject_dirs(subdirs, category)

    # 그룹 키 정렬 리스트 생성 (일관성 보장)
    group_keys = sorted(list(groups.keys()))
    # 그룹 키들을 셔플합니다.
    random.Random(seed).shuffle(group_keys)

    # 그룹 키들을 k_folds로 나눕니다.
    folds_keys = [group_keys[i::k_folds] for i in range(k_folds)]

    # 각 fold의 실제 subdir 목록을 만듭니다.
    folds = []
    for f_keys in folds_keys:
        fold_subdirs = []
        for k in f_keys:
            fold_subdirs.extend(groups[k])
        folds.append(fold_subdirs)
        
    val_subdirs = folds[fold_idx]
    train_subdirs = [sd for i, fold in enumerate(folds) if i != fold_idx for sd in fold]
    return train_subdirs, val_subdirs, folds


def collect_split_items(data_dir="dataset/raw", split="train"):
    """고정 split 하나에서 (cropRGB, cropIR, label) 항목을 수집한다.

    디렉터리 구조: data_dir/{split}/{category}/{subject}/{frame}/cropRGB.bmp
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
    # CLASS_MAPPING은 CLASS_NAMES 순서를 그대로 따르므로 label 값이 인덱스와 일치한다.
    for category, label in CLASS_MAPPING.items():
        cat_path = os.path.join(split_dir, category)
        if not os.path.isdir(cat_path):
            raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")

        # live는 high/medium 하위 폴더까지 훑고, spoof는 평면 구조다(_sort_subject_dirs 참고).
        subdirs = _sort_subject_dirs(cat_path, category)
        if not subdirs:
            raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")

        category_items = gather_frame_items(cat_path, subdirs, label)
        if not category_items:
            raise ValueError(f"프레임이 비어 있습니다: {cat_path}")
        items.extend(category_items)

    return items


def validate_fixed_split_coverage(data_dir="dataset/raw"):
    """고정 split의 완전성과 subject/frame/content/metadata 누수를 검사한다.

    live는 high/medium의 동일 번호를 같은 인물로 본다. spoof 클래스는 기존
    Group K-Fold 계약과 동일하게 전체 subject 정렬 순서에서 연속 두 폴더를
    같은 물리 인물로 본다.

    학습 시작 직전에 tf_train.main()이 호출한다. 같은 인물/같은 프레임이 train과
    validation에 동시에 있으면 검증 지표가 부풀려져 실제 성능보다 좋아 보인다.
    그런 결과로 모델을 선택하면 현장에서 그대로 실패하므로, 아래 네 겹으로 검사하고
    하나라도 걸리면 예외를 던져 학습 자체를 막는다.

      1) subject leakage  — 같은 물리 인물이 두 split에 걸쳐 있는가
      2) realpath leakage — 심볼릭 링크를 풀었을 때 같은 파일을 가리키는가
      3) content leakage  — 경로가 달라도 파일 내용(MD5)이 같은가 (복사본 탐지)
      4) metadata leakage — meta.json의 session/video ID가 두 split에 걸쳐 있는가

    반환: {split: 프레임 수} — 호출부가 로그로 출력한다.
    """
    split_subjects = {split: {} for split in FIXED_SPLITS}
    split_items = {}

    # 먼저 세 split의 subject 목록과 프레임 목록을 모두 읽어 둔다(검사는 그다음).
    for split in FIXED_SPLITS:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"고정 split 디렉터리가 없습니다: {split_dir}")

        for category in CLASS_MAPPING:
            cat_path = os.path.join(split_dir, category)
            if not os.path.isdir(cat_path):
                raise FileNotFoundError(f"클래스 디렉터리가 없습니다: {cat_path}")
            subdirs = _sort_subject_dirs(cat_path, category)
            if not subdirs:
                raise ValueError(f"subject 폴더가 비어 있습니다: {cat_path}")
            split_subjects[split][category] = subdirs

        split_items[split] = collect_split_items(data_dir, split)

    # 1. Subject leakage 검사
    for category in CLASS_MAPPING:
        # 그룹핑(특히 spoof의 "연속 2개 = 동일 인물") 규칙은 정렬 순서에 의존하므로,
        # split별로 따로 묶으면 안 되고 세 split을 합친 전체 목록에서 한 번에 묶어야 한다.
        all_subdirs = sorted({
            subject
            for split in FIXED_SPLITS
            for subject in split_subjects[split][category]
        }, key=_subject_sort_key)
        groups = _group_subject_dirs(all_subdirs, category)
        # subject 폴더 → 그룹(물리 인물) 키 역인덱스
        subject_to_group = {
            subject: group_key
            for group_key, subjects in groups.items()
            for subject in subjects
        }

        # 그룹별로 "어느 split에 등장했는가"를 모은다.
        group_splits = {}
        for split in FIXED_SPLITS:
            for subject in split_subjects[split][category]:
                group_key = subject_to_group[subject]
                group_splits.setdefault(group_key, set()).add(split)

        # 한 그룹이 2개 이상의 split에 걸쳐 있으면 그 인물의 얼굴을 학습에서 이미 본 것이다.
        overlaps = {
            group_key: sorted(splits)
            for group_key, splits in group_splits.items()
            if len(splits) > 1
        }
        if overlaps:
            raise ValueError(
                f"{category} 클래스의 동일 물리 subject가 여러 split에 있습니다: {overlaps}"
            )

    # 2. Realpath leakage 검사
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

    # 3. Content MD5 Hash Leakage 검사
    # 2번(realpath)을 통과해도, 파일을 물리적으로 복사해 두 split에 넣었다면 잡히지 않는다.
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

    # 4. Explicit Manifest / Metadata Leakage 검사 (meta.json)
    # 같은 촬영 세션·같은 영상에서 뽑은 서로 다른 프레임은 파일 내용이 달라 1~3번을 모두
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
            # meta.json은 선택 사항이다. 없으면 이 검사만 건너뛴다(1~3번은 이미 통과한 상태).
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                except Exception:
                    # 깨진 meta.json 하나 때문에 전체 검증을 멈추지는 않는다.
                    continue

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

    return {split: len(items) for split, items in split_items.items()}


def validate_kfold_coverage(data_dir="dataset/raw", k_folds=5, seed=42):
    for category in CLASS_MAPPING.keys():
        cat_path = os.path.join(data_dir, category)
        if not os.path.exists(cat_path):
            continue

        subdirs = _sort_subject_dirs(cat_path, category)
        if len(subdirs) < k_folds:
            raise ValueError(
                f"{category} 클래스의 subject 폴더 수({len(subdirs)})가 K({k_folds})보다 적습니다."
            )

        _, _, folds = _split_kfold_subjects(subdirs, k_folds, 0, seed, category)
        seen = [sd for fold in folds for sd in fold]
        assert len(seen) == len(set(seen)), \
            f"{category} 클래스의 fold validation subject가 서로 겹칩니다."
        assert set(seen) == set(subdirs), \
            f"{category} 클래스의 fold validation subject가 전체 subject를 덮지 못합니다."


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

    10-클래스 분류 결과를 live vs spoof 2진 관점으로 눌러서 보는 것이 핵심이다.
      APCER = 스푸핑을 live(0)로 통과시킨 비율          ← 보안 사고에 직결
      BPCER = 진짜 사람을 spoof로 거부한 비율            ← 사용성 저하
      ACER  = 두 값의 단순 평균 (현재 체크포인트 선택 기준, 낮을수록 좋음)

    구체 예) 검증셋에 live 100장, spoof 900장이 있고 spoof 중 9장을 live로,
    live 중 5장을 spoof로 틀렸다면
      APCER = 9/900 = 0.01, BPCER = 5/100 = 0.05, ACER = (0.01+0.05)/2 = 0.03.
    같은 상황에서 accuracy는 986/1000 = 0.986으로 아주 좋아 보이지만,
    ACER는 BPCER 쪽 문제를 그대로 드러낸다. 이것이 accuracy 대신 ACER로 모델을 고르는 이유다.

    스푸핑 종류(print/mask/...)를 서로 혼동하는 것은 APCER/BPCER에 영향을 주지 않는다.
    "spoof를 다른 spoof로 분류"해도 결국 거부되기 때문. 그 세부는 혼동행렬로 본다.
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

    # 인덱스 0 = live, 1 이상 = 전부 spoof (classes.py의 CLASS_NAMES 순서에 의존).
    live_mask = labels == 0
    spoof_mask = labels != 0
    total_live = int(live_mask.sum())
    total_spoof = int(spoof_mask.sum())
    # 정답은 spoof인데 live로 예측 → 공격 통과
    apcer_errors = int(((preds == 0) & spoof_mask).sum())
    # 정답은 live인데 spoof(어떤 종류든)로 예측 → 정상 사용자 거부
    bpcer_errors = int(((preds != 0) & live_mask).sum())

    apcer = apcer_errors / total_spoof if total_spoof > 0 else 0.0
    bpcer = bpcer_errors / total_live if total_live > 0 else 0.0
    # 단순 평균이라 클래스 불균형(spoof 9 : live 1)의 영향을 받지 않는다.
    acer = (apcer + bpcer) / 2.0
    return confusion_matrix, recalls, apcer, bpcer, acer
