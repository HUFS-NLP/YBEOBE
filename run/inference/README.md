## 1. 모델 다운로드
- 링크: https://drive.google.com/file/d/1cRyBTH3vuFYAlRrR9L_TFjQClchDOfo_/view?usp=sharing

## 2. 모델 압축 해제
- 소스코드가 설치된 폴더에서 위 링크를 압축해제하면 models 폴더가 생김
- 해당 models 폴더를 https://github.com/JuaeKim54/YBEOBE 아래에 위치시킴

## 3. 아래 코드 실행
- train_and_inference_testEmo.py
- train_and_inference_찌리리공.py
- train_and_inference_꼬렛_v1.py
- train_and_inference_꼬렛_v2.py

- 실행시 test_EMO, 찌리리공, 꼬렛_v1, 꼬렛_v2 모델들이 학습되고, 추론된 결과 jsonl 파일들이 outputs 폴더 안에 생성됨

## 4. inference.py 코드 실행
- 실행시 각 단일모델의 추론 결과 jsonl 파일들이 outputs 폴더 안에 생성됨

## 5. ensemble.py 코드 실행
- 실행 시 단계별로 앙상블 실행 및 결과 jsonl 파일들이 outputs 폴더 안에 생성됨

## 6. (1) outputs 폴더의 힐볼.jsonl 제출 시 순위표(리더보드)에 성적 반영
## 6. (2) outputs 폴더의 뮤츠_v2.jsonl 제출 시 순위표(리더보드)에 성적 반영

