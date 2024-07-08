# DA34-final-duo-flytothesky
최준환 김성준
OpenWebUI와 AIHUB 데이터셋을 활용한 은어 순화어 번역 서비스
이 레포지토리는 OpenWebUI와 AIHUB 데이터셋을 활용하여 은어를 순화어로 번역하는 LLM(대형 언어 모델) 기반의 서비스를 구축하는 방법을 소개합니다.

목차
소개
기능
설치
사용법
데이터셋
모델 학습
기여 방법
라이선스
소개
이 프로젝트는 OpenWebUI와 AIHUB의 데이터셋을 활용하여 은어를 순화어로 번역하는 대형 언어 모델(LLM) 기반의 서비스를 구축하기 위한 종합적인 솔루션을 제공합니다. 이 서비스는 텍스트를 입력받아 은어를 감지하고 적절한 순화어로 번역해줍니다.

기능
은어 감지: 입력된 텍스트에서 은어 감지
순화어 번역: 감지된 은어를 적절한 순화어로 번역
사용자 정의 학습: 사용자의 데이터셋으로 모델 재학습
웹 인터페이스: 모델과 상호 작용하기 위한 사용자 친화적인 웹 인터페이스
설치
필요 조건
Python 3.8 이상
Git
Virtualenv
설치 방법
레포지토리 클론:

bash
코드 복사
git clone https://github.com/yourusername/slang-translation-service.git
cd slang-translation-service
가상 환경 생성 및 활성화:

bash
코드 복사
virtualenv venv
source venv/bin/activate
필요한 패키지 설치:

bash
코드 복사
pip install -r requirements.txt
OpenWebUI 설치:

OpenWebUI GitHub 페이지의 지침을 따라 OpenWebUI를 설치하고 설정합니다.

사용법
웹 인터페이스 실행
OpenWebUI 서버 시작:

bash
코드 복사
openwebui-server
서비스 실행:

bash
코드 복사
python app.py
브라우저를 열고 http://localhost:5000에 접속하여 웹 인터페이스에 접근합니다.

모델과의 상호작용
웹 인터페이스 또는 제공된 API 엔드포인트를 통해 모델과 상호작용할 수 있습니다.

은어 순화어 번역
은어를 순화어로 번역하려면 /translate 엔드포인트를 사용합니다:

bash
코드 복사
curl -X POST http://localhost:5000/translate -d '{"text": "이거 완전 대박이다!"}'
데이터셋
AIHUB의 다양한 데이터셋을 사용하여 모델을 학습시킬 수 있습니다. 데이터셋을 다운로드하고 준비하는 방법은 AIHUB 사이트를 참고하세요.

모델 학습
자신만의 데이터셋을 사용하여 모델을 학습시키려면, train.py 스크립트를 사용합니다:

bash
코드 복사
python train.py --dataset_path /path/to/your/dataset
기여 방법
기여를 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 등을 통해 기여할 수 있습니다.

라이선스
이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.


기여 방법
기여를 환영합니다! 버그 리포트, 기능 요청, 풀 리퀘스트 등을 통해 기여할 수 있습니다.

라이선스
이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.
