"""Keras/TensorFlow training and TFLite conversion pipeline.

이 파일은 폴더를 파이썬 패키지로 만드는 표시자다. 실행되는 코드는 없고
docstring만 들어 있으며, 있기 때문에 아래가 성립한다.

    from keras_pipeline.tf_dataset import make_dataset   # 폴더=패키지, 파일=모듈
    python -m keras_pipeline.tf_train                    # 학습 실행 방식

모듈 구성:
    spec            입력 규격 상수 (아래 모듈들이 공유하는 계약)
    tf_dataset      tf.data 입력 파이프라인 (디코딩·증강·정규화)
    tf_model        MobileNetV2 모델 정의
    tf_train        학습 진입점
    convert_keras_to_tflite   .keras → .tflite (float / int8 / npu_int8)
    export_validator          NPU용 재조립·동등성 검증·매니페스트 작성
    model_signature           입출력 텐서 규격 검사
    artifact_paths  산출물 파일명 규칙
    run_metadata    학습 실행 기록
"""
