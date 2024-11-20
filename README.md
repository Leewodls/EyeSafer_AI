# EyeSafer
- 인공지능 기반의 인구 밀집도 측정 및 경고 시스템
- 안전 사고 예방 및 효율적인 인력 배치를 목적으로 설계된 프로젝트 
- YOLOv5 모델을 활용하여 실시간 객체 검출 및 밀집도 계산을 수행하며, 경고 알림 기능을 제공

  ## 목표
- 안전 사고 방지:
  - 밀집도가 높은 지역을 실시간으로 탐지하여 경고 알림 제공.
- 효율적인 자원 관리:
  - 불필요한 인력 배치를 방지하고, 필요한 곳에만 배치.
- 객체 밀집도 계산:
  - 영상 데이터에서 객체(사람)를 검출하고, 면적 대비 밀집도를 계산.

## 기능
- 객체 검출(Object Detection)
  - YOLOv5를 활용하여 영상에서 사람 객체를 검출.
  - Flask를 사용한 웹 애플리케이션으로 실시간 분석 제공.
- 객체 밀집도 계산
  - 객체가 특정 면적에 3명 이상 모일 경우 경고 메시지와 시각적 알림(빨간 박스) 출력.

- 경고 알림
  - 기준치를 초과하는 밀집도가 탐지되면 실시간 경고 메시지 출력.

# Tech

## 언어
<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/HTML-3766AB?style=flat-square&logo=HTML&logoColor=white"/>

## 프레임워크 및 라이브러리
<img src="https://img.shields.io/badge/Flask-00599C?style=flat-square&logo=Flask&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-00599C?style=flat-square&logo=OpenCV&logoColor=white"/> <img src="https://img.shields.io/badge/YOLOv5-00599C?style=flat-square&logo=YOLOv5&logoColor=white"/>
- AI 모델: YOLOv5
- 데이터셋: COCO 데이터셋, 군중 밀집 데이터셋(crowd-counting-dataset-w3o7w)
- 웹 프레임워크: Flask

## 협업 도구
<img src="https://img.shields.io/badge/Git-00599C?style=flat-square&logo=Git&logoColor=white"/> <img src="https://img.shields.io/badge/Slack-00599C?style=flat-square&logo=Slack&logoColor=white"/>

# Output
`Original Video`  `Roboflow 2.0 Object Detection(Fast)` `Inference YOLOv5 Inference`  `Heatmap`

![GIFMaker_me (2)](https://github.com/hanghae-hackathon/EyeSafer_AI/assets/44021629/e192154f-c64e-49c8-a4b7-f8776067a314) 
![GIFMaker_me (1)](https://github.com/hanghae-hackathon/EyeSafer_AI/assets/44021629/5b251cd1-0aa9-4dd1-acac-383817474459) 
![GIFMaker_me](https://github.com/hanghae-hackathon/EyeSafer_AI/assets/44021629/b6036f1d-184c-42b4-bcb0-44ff129ac7ad) 
![model(roboflow) test video](https://github.com/hanghae-hackathon/EyeSafer_AI/assets/145883892/2e77d088-6a48-4991-88d8-d0b40812cc5c)
