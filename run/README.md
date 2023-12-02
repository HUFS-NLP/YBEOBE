## BiLSTM + attention
- target을 query, key와 value를 text로 하는 scaled dot-product attention 도입

## threshold 최적화
- 각각 감정 라벨에 대해 개별적인 threshold 값 설정

## ASL loss
- 클래스 불균형 해소 위해 도입한 손실 함수

## SpanEMO
- 텍스트와 감정 클래스 세트를 모두 입력으로 받아 전통적인 다중 레이블 분류 작업을 새로운 span-prediction 문제로 전환
- label correlation aware 손실 함수 도입


### refernece
- scaled dot-product attention (https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- ASL loss (https://github.com/Alibaba-MIIL/ASL/tree/main)
- SpanEMO (https://github.com/hasanhuz/SpanEmo)
