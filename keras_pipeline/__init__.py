"""Keras/TensorFlow 학습 및 TFLite 변환 파이프라인.

이 파일은 폴더를 파이썬 패키지로 만드는 표시자다. 실행되는 코드는 없고
docstring만 들어 있으며, 있기 때문에 아래가 성립한다.

    from keras_pipeline.data.dataset import make_dataset
    python -m keras_pipeline.training.train

모듈 구성:
    data            입력 규격·깊이 지도·tf.data 파이프라인
    models          백본·손실·모델 조립
    training        학습 진입점·산출물 경로·실행 기록
    export          TFLite 변환·NPU 동등성 검증·매니페스트
    contracts       입출력 텐서 규격 검사
"""
