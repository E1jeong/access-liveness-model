"""3D 깊이 지도(Pseudo Depth Map) 실시간 생성 모듈.

출입통제 안티스푸핑 모델의 Multi-Task Auxiliary 지도학습을 위해
입력 이미지 및 라벨에 대응하는 14x14 크기의 3D 깊이 지도를 메모리 상에서 실시간 생성합니다.

깊이 지도 규칙 (ISO/IEC PAD 물리 모델 기반):
  1) live (0): 코를 정점으로 하는 3D 볼록 타원 곡면 (0.0 ~ 1.0)
  2) dental_white (10), dental_black (11): 상안부(이마/눈/미간) 3D 곡면 + 하안부(마스크) 감쇄 (0.0 ~ 1.0)
  3) curved_* (6, 7, 8, 9): 1차원 원통형 곡면 (최대 높이 0.25)
  4) flat spoof (1, 2, 3, 4, 5): 완전 평면 Z=0 (모든 픽셀 0.0)
"""
import cv2
import numpy as np


# 14x14 기본 템플릿 사전 생성 (학습 시 CPU 오버헤드 극소화)
def _build_base_templates(size=(14, 14)):
    h, w = size
    y = np.linspace(-1, 1, h)
    x = np.linspace(-1, 1, w)
    X, Y = np.meshgrid(x, y)

    # 1. Live Face 3D Template
    center_y = -0.05
    R = np.sqrt(X**2 + ((Y - center_y) / 1.2)**2)
    dist_nose = np.sqrt(X**2 + (Y - 0.05)**2)

    face_dome = np.clip(1.0 - R**1.8, 0.0, 1.0)
    nose_peak = 0.35 * np.exp(- (dist_nose**2) / (2 * 0.20**2))
    forehead = 0.15 * np.exp(- (X**2 + (Y + 0.4)**2) / (2 * 0.30**2))
    cheeks = 0.15 * (np.exp(- ((X - 0.35)**2 + Y**2) / (2 * 0.25**2)) +
                     np.exp(- ((X + 0.35)**2 + Y**2) / (2 * 0.25**2)))

    live_depth = face_dome * 0.5 + nose_peak + forehead + cheeks
    live_depth = np.clip(live_depth, 0.0, 1.0)
    live_depth[R > 0.90] = 0.0

    # 2. Dental Mask 3D Template (하안부 마스크 감쇄)
    dental_depth = live_depth.copy()
    mask_region = Y > 0.1
    dental_depth[mask_region] = dental_depth[mask_region] * 0.25

    # 3. Curved Spoof Template (1D 원통형 곡면)
    cylinder = np.clip(1.0 - (x**2) * 0.8, 0.0, 1.0) * 0.25
    curved_depth = np.tile(cylinder, (h, 1)).astype(np.float32)

    # 4. Flat Spoof Template (완전 평면 0)
    flat_depth = np.zeros(size, dtype=np.float32)

    return {
        "live": live_depth.astype(np.float32),
        "dental": dental_depth.astype(np.float32),
        "curved": curved_depth.astype(np.float32),
        "flat": flat_depth.astype(np.float32),
    }


_CACHED_TEMPLATES = _build_base_templates((14, 14))


def generate_pseudo_depth_map(label_idx, size=(14, 14), flip=0, angle=0.0):
    """지정된 클래스 라벨 및 기하 증강(flip/angle)에 맞춰 3D 깊이 지도를 생성한다.

    반환: shape (14, 14, 1), float32 [0.0, 1.0]
    """
    if size == (14, 14):
        templates = _CACHED_TEMPLATES
    else:
        templates = _build_base_templates(size)

    # 클래스에 따른 기본 템플릿 선택
    if label_idx == 0:
        base = templates["live"].copy()
    elif label_idx in (10, 11):
        base = templates["dental"].copy()
    elif label_idx in (6, 7, 8, 9):
        base = templates["curved"].copy()
    else:
        # 평면 스푸핑은 기하 변환을 거쳐도 항상 0
        return np.zeros((size[0], size[1], 1), dtype=np.float32)

    # 공간 증강 동기화 (이미지 회전/반전과 동일하게 회전)
    if flip == 1:
        base = cv2.flip(base, 1)

    if abs(angle) > 1e-3:
        h, w = size
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        base = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    base = np.clip(base, 0.0, 1.0)
    return np.expand_dims(base.astype(np.float32), axis=-1)
