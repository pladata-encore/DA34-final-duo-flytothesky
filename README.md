# Gemmini: 세대간 언어 번역기

![logo](img/E8D18C5B-0C80-4873-8F9E-73CFBC5C7A69.png)

> Gemmini는 세대 간 언어 차이를 극복하고 원활한 소통을 돕는 AI 기반 번역 서비스입니다. 각 세대의 특징적인 표현과 은어를 상대방이 이해하기 쉬운 언어로 번역해 이해할 수 있게 만들어 줘요!

## 팀 구성원

김성준(<ksjhshl@naver.com>) / 최준환(<wnsksl0527@gmail.com>)

## 활용 데이터

- aihub: [연령대별 특징적 발화(은어·속어 등) 음성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71320)

| No | 항목 | 설명 | 타입 | (예시) |
|---|---|---|---|---|
| 1 | DataSet | 데이터셋 | String | 연령대별특징적발화음성 |
| 2 | Version | 데이터셋 버전 | String | 1.0 |
| 3 | Date | 녹음날짜 | String | 20220523 |
| 4 | MediaUrl | 녹취된 음원의 URL | String | 대화/20대/26.일상/20_26_42785278_220906_0001.wav |
| 5 | Category | 연령대분류 [10대/20대/30대/40대/50대이상] | String | 20대 |
| 6 | Subcategory | 주제별 분류 [게임,SNS,교육,일상,군대 등] | String | 일상 |
| 7 | DialogPlace | 발화장소 [집/휴게실/사무실/야외/실내/실외] | String | 사무실 |
| 8 | DialogVoiceType | [독백/대면/음성통화/영상통화] | String | 대면대화 |
| 9 | SpeakerNumber | 화자수 [1명/2명] | String | 2 |
| 10 | RecDevice | [스마트폰, 컴퓨터, 녹음장치] | String | 녹음장치 |
| 11 | RecLen | 전체 녹음 시간(초) | number | 60 |
| 12 | AudioResolution | 오디오레졸류션 | Object | {} |
| 12-1 | BitDepth | 16bit | String | 16bit |
| 12-2 | SampleRate | 44.1kHz | String | 44.1kHz |
| 13 | Speakers | 화자 정보 (화자 목록) | Array | [] |
| 13-1 | Speaker | 화자코드 | String | 4278 |
| 13-2 | Gender | [남성/여성] | String | 남성 |
| 13-3 | Locate | [수도권/강원/충청/전라/경상] | String | 수도권 |
| 13-4 | Agegroup | 화자연령대 [10/20/30/40/50] | String | 30 |
| 14 | Dialogs | 전사 데이터 목록 (대화 목록) | Array | [] |
| 14-1 | Speaker | 화자코드 | String | 4278 |
| 14-2 | Speakertext | 은어 속어가 포함된 발화문장 | String | 오늘 점심은 뭐로 할까? 점메추 좀~ |
| 14-3 | TextConvert | 은어 속어를 풀어쓴 발화문장 | String | 오늘 점심은 뭐로 할까? 점심메뉴추천 좀~ |
| 14-4 | StartTime | 발화 시작 시간 | number | 1 |
| 14-5 | EndTime | 발화 종료 시간 | number | 4 |
| 14-6 | SpeakTime | 화자 발화한 발성시간(초) | number | 3 |
| 14-7 | WordInfo | 특정적단어에 대한 세부정보 | Array | [] |
| 14-7-1 | Word | 특징적 발화 단어 | String | 점메추 |
| 14-7-2 | WordType | 단어유형분류[은어/속어/유행어/줄임말 등] | String | 줄임말 |
| 14-7-3 | WordStructure | 단어구조분류[파생어,합성어,혼성어, 축약어 등] | String | 축약어 |
| 14-7-4 | WordDefine | 특정단어의 뜻풀이 | String | 점심 메뉴 추천 |
| 14-7-5 | WordFell | 감정의 반응을 표시 [긍정/부정/중립] | String | 중립 |
| 14-7-6 | WordMean | 감정의 세부 항목(의도 분류) 긍정[좋음/선의] 부정[싫어함/화남] 중립[감정없음] | String | 궁금 |
| 14-7-1 | Word | 특징적 발화 단어 | String | 점메추 |
| 14-7-2 | WordType | 단어유형분류[은어/속어/유행어/줄임말 등] | String | 줄임말 |
| 14-7-3 | WordStructure | 단어구조분류[파생어,합성어,혼성어, 축약어 등] | String | 축약어 |
| 14-7-4 | WordDefine | 특정단어의 뜻풀이 | String | 점심 메뉴 추천 |
| 14-7-5 | WordFell | 감정의 반응을 표시 [긍정/부정/중립] | String | 중립 |
| 14-7-6 | WordMean | 감정의 세부 항목(의도 분류) 긍정[좋음/선의] 부정[싫어함/화남] 중립[감정없음] | String | 궁금 |

## 주요 기능

- 세대별 언어 스타일 분석 및 변환
- 실시간 채팅 Q&A
- 음성 인식을 통한 채팅 지원
- 다국어 지원

## 기술 스택

- langchain
- llama-index
- open-web-ui
- pipelines

## 활용 모델

Open LLM:

- open-ai-gpt3.5-turbo
- xionic-ko-llama-3-70b

Closed LLM:

- llama-3-Korean-Bllossom-70B-gguf-Q4_K_M
- llama3-alphako-8b-q8
- llama3-alphako-8b-q5-k-m
- llama3-instruct-8b
- llama-3-Open-Ko-8B-Q4_K_M

## 아키텍처

![아키텍처](img/0BBBB15C-881E-4960-8B91-171C8C082D6A.png)
<br>

# 시작하기

## 파이썬 요구 사항

- Python 3.11+
- TensorFlow 2.0+
- PyTorch 1.8+

## 환경 구성

multi-modal

```sh
pip install -r requirements-multi-modal.txt
```

chat-qa

```sh
pip install -r requirements-chat-qa.txt
```

## Multi Modal App (gradio)

![gradio](img/gradio_final.png)
```
```

## Chat QA App (open-web-ui)

### Ollama / Local LLM 설치

- 플랫폼에 맞게 Ollama 설치: <https://ollama.com/download>
- `model/README.md`를 참고하여 로컬 모델 Load

### Docker 실행

- open web ui

```sh
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

- pipelines

```sh
docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
```

### 프롬프트 구성

### 모델 비교

모델을 자원의 한계와 정성적 평가 방식으로 진행함

![모델비교](img/model-comparison.png)

1. 세대 간 언어 번역의 경우, 시대적 맥락과 문화적 뉘앙스가 중요하여 단순한 수치 지표로 평가하기 어려움

2. 창의적 표현이나 의역의 적절성을 객관적인 수치로 나타내기 어려워 정량적 평가만으로는 번역의 질을 완전히 반영하기 어렵기 때문
