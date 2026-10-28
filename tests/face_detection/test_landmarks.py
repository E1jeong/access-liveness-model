import hashlib
import json

import pytest
import torch
from PIL import Image

from common.landmark_metadata import capture_metadata_digest

from face_detection_pipeline.data.dataset import FaceDetectionDataset, collate_face_detection_batch
from face_detection_pipeline.models.yunet import build_yunet_n
from face_detection_pipeline.training.loss import YuNetCriterion
from face_detection_pipeline.training.train import parse_args, train


NAMES = ('eyeImageLeft', 'eyeImageRight', 'noseTip', 'mouthImageLeft', 'mouthImageRight')


def write_frame(root, *, enabled=5, detected=True):
    frame = root / 'live' / '1'
    frame.mkdir(parents=True)
    Image.new('RGB', (96, 128), (20, 30, 40)).save(frame / 'RGB.bmp')
    metadata = {'schemaVersion': 2, 'rgb': {
        'width': 96, 'height': 128, 'faceRect': [20, 20, 75, 110],
    }}
    label = {
        'schemaVersion': 1, 'image': 'RGB.bmp', 'imageWidth': 96, 'imageHeight': 128,
        'coordinateSystem': 'original_image_pixels',
        'status': 'extracted' if detected else 'not_detected',
        'imageSha256': hashlib.sha256((frame / 'RGB.bmp').read_bytes()).hexdigest(),
        'captureSha256': capture_metadata_digest(metadata),
        'points': {name: {'x': x if detected else None, 'y': y if detected else None,
                          'useForTraining': detected and i < enabled}
                   for i, (name, x, y) in enumerate(zip(NAMES, [30, 60, 45, 32, 58], [40, 40, 60, 85, 85]))},
    }
    metadata['landmarks'] = label
    (frame / 'meta.json').write_text(json.dumps(metadata))
    return frame, label


@pytest.mark.parametrize('change', ['nan', 'bool', 'bounds', 'flag', 'hash', 'coordinates', 'names'])
def test_landmark_dataset_rejects_invalid_selected_labels(tmp_path, change):
    frame, label = write_frame(tmp_path)
    point = label['points'][NAMES[0]]
    if change == 'nan':
        point['x'] = float('nan')
    elif change == 'bool':
        point['x'] = True
    elif change == 'bounds':
        point['x'] = 96
    elif change == 'flag':
        point['useForTraining'] = 'true'
    elif change == 'hash':
        label['imageSha256'] = 'bad'
    elif change == 'coordinates':
        label['coordinateSystem'] = 'crop_pixels'
    else:
        label['points'].pop(NAMES[0])
    metadata = json.loads((frame / 'meta.json').read_text())
    metadata['landmarks'] = label
    (frame / 'meta.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        FaceDetectionDataset(tmp_path, image_size=(64, 64), landmarks=True)[0]


def test_mixed_targets_keep_boxes_and_ignore_disabled_coordinates(tmp_path):
    for i, enabled in enumerate([5, 2, 0]):
        frame, label = write_frame(tmp_path / str(i), enabled=enabled, detected=enabled != 0)
        if enabled == 2:
            label['points']['noseTip']['x'] = None
            metadata = json.loads((frame / 'meta.json').read_text())
            metadata['landmarks'] = label
            (frame / 'meta.json').write_text(json.dumps(metadata))
    dataset = FaceDetectionDataset(tmp_path, image_size=(64, 64), landmarks=True)
    images, boxes, points, masks = collate_face_detection_batch([dataset[i] for i in range(3)])
    assert images.shape == (3, 3, 64, 64)
    assert [box.shape for box in boxes] == [(1, 4)] * 3
    assert [int(mask.sum()) for mask in masks] == [5, 2, 0]
    assert torch.allclose(points[0][0, 0], torch.tensor([23., 20.]))
    assert (points[1][~masks[1]] == 0).all()


def test_zoom_uses_same_transform_for_boxes_and_points(tmp_path):
    write_frame(tmp_path)
    original = FaceDetectionDataset(tmp_path, image_size=(64, 64), landmarks=True)[0]
    torch.manual_seed(9)
    zoomed = FaceDetectionDataset(tmp_path, image_size=(64, 64), landmarks=True,
                                  zoom_probability=1)[0]
    _, box, points, mask = original
    _, zoom_box, zoom_points, zoom_mask = zoomed
    ratio = (zoom_box[0, 2:] - zoom_box[0, :2]) / (box[0, 2:] - box[0, :2])
    expected = (points - box[:, None, :2]) * ratio + zoom_box[:, None, :2]
    assert torch.allclose(zoom_points[zoom_mask], expected[mask], atol=1e-5)


def test_landmark_loss_masks_gradients_and_handles_no_labels():
    model = build_yunet_n(landmarks=True)
    outputs = model(torch.randn(3, 3, 64, 64))
    for level in outputs.landmark_regressions:
        level.retain_grad()
    boxes = [torch.tensor([[12., 10., 52., 56.]])] * 3
    points = [torch.tensor([[[20., 20.], [40., 20.], [30., 30.], [22., 45.], [38., 45.]]])] * 3
    masks = [torch.ones(1, 5, dtype=torch.bool),
             torch.tensor([[True, True, False, False, False]]), torch.zeros(1, 5, dtype=torch.bool)]
    losses = YuNetCriterion()(outputs, boxes, points, masks)
    assert losses['landmark'] > 0
    assert all(torch.isfinite(value) for value in losses.values())
    losses['total'].backward()
    assert any(level.grad[0].abs().sum() > 0 for level in outputs.landmark_regressions)
    for level in outputs.landmark_regressions:
        assert (level.grad[1, 4:] == 0).all()
        assert (level.grad[2] == 0).all()
    outputs = model(torch.randn(3, 3, 64, 64))
    losses = YuNetCriterion()(outputs, boxes, points, [torch.zeros_like(mask) for mask in masks])
    assert losses['landmark'] == 0
    losses['total'].backward()


def test_landmark_training_and_resume(tmp_path):
    data = tmp_path / 'data'
    write_frame(data / 'train', enabled=2)
    write_frame(data / 'validation')
    options = ['--landmarks', '--data-dir', str(data), '--output-dir', str(tmp_path / 'model'),
               '--image-width', '64', '--image-height', '64', '--batch-size', '1',
               '--num-workers', '0', '--device', 'cpu', '--zoom-probability', '0']
    path = train(parse_args(options + ['--epochs', '1', '--run-id', 'landmark_smoke']))
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['model'] == 'yunet_n_landmarks'
    assert checkpoint['landmark_names'] == list(NAMES)
    assert checkpoint['validation_metrics']['landmark'] > 0
    path = train(parse_args(options + ['--epochs', '2', '--resume', str(path)]))
    latest = path.with_name('latest_yunet_n_64x64.pth')
    assert torch.load(latest, weights_only=True)['epoch'] == 2
    with pytest.raises(ValueError, match='model'):
        train(parse_args([arg for arg in options if arg != '--landmarks'] +
                         ['--epochs', '3', '--resume', str(latest)]))


@pytest.mark.parametrize('height,width', [(320, 320), (448, 256)])
def test_landmark_head_shapes(height, width):
    model = build_yunet_n(landmarks=True).eval()
    with torch.inference_mode():
        outputs = model(torch.zeros(1, 3, height, width))
    for level, stride in zip(outputs.landmark_regressions, model.strides):
        assert level.shape == (1, 10, height // stride, width // stride)


def test_landmark_loss_all_empty_boxes():
    outputs = build_yunet_n(landmarks=True)(torch.zeros(1, 3, 64, 64))
    losses = YuNetCriterion()(outputs, [torch.empty(0, 4)], [torch.empty(0, 5, 2)],
                             [torch.empty(0, 5, dtype=torch.bool)])
    assert losses['landmark'] == 0
    assert losses['objectness'] > 0
    losses['total'].backward()


def test_landmark_loss_exact_grid_encoding_and_masked_nan():
    from face_detection_pipeline.models.yunet import YuNetLandmarkOutputs
    from face_detection_pipeline.training.loss import _AssignResult

    cls = [torch.zeros(1, 1, 1, 1, requires_grad=True)]
    regressions = [torch.zeros(1, 10, 1, 1, requires_grad=True)]
    outputs = YuNetLandmarkOutputs(cls, cls, [torch.zeros(1, 4, 1, 1)], regressions)
    points = torch.full((1, 5, 2), float('nan'))
    points[0, 0] = torch.tensor([8., 16.])
    criterion = YuNetCriterion(strides=(8,))
    criterion.assigner.assign = lambda *args: _AssignResult(torch.tensor([1]), torch.tensor([1.]))
    losses = criterion(outputs, [torch.tensor([[0., 0., 32., 32.]])], [points],
                       [torch.tensor([[True, False, False, False, False]])])
    # Grid origin (0,0), stride 8: encoded target (1,2); mean Smooth L1 = 1.
    assert losses['landmark'].item() == pytest.approx(1.)
    losses['total'].backward()
    assert torch.isfinite(regressions[0].grad).all()
    assert (regressions[0].grad[:, 2:] == 0).all()


def test_landmark_training_rejects_dataset_without_selected_points(tmp_path):
    data = tmp_path / 'data'
    write_frame(data / 'train', detected=False)
    write_frame(data / 'validation', detected=False)
    args = parse_args(['--landmarks', '--data-dir', str(data), '--output-dir', str(tmp_path / 'model'),
                       '--image-width', '64', '--image-height', '64', '--epochs', '1',
                       '--batch-size', '1', '--num-workers', '0', '--device', 'cpu'])
    with pytest.raises(ValueError, match='at least one selected'):
        train(args)
    assert not list((tmp_path / 'model').rglob('*.pth'))


def test_exporter_rejects_incomplete_landmark_contract_before_conversion(tmp_path):
    from face_detection_pipeline.export.convert import load_checkpoint

    path = tmp_path / 'landmarks.pth'
    torch.save({'model': 'yunet_n_landmarks'}, path)
    with pytest.raises(ValueError, match='landmark contract'):
        load_checkpoint(path)
