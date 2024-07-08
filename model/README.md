| 모델 파일 크기가 매우 커 직접 다운로드 필요 아래의 지침에 따라 실행하면 ollama 로컬 모델 생성/적용 가능

1. huggingface 모델 다운로드 : [EEVE모델링크]https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF

2. Ollama 실행

```sh
ollama create EEVE-Korean-10.8B -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile
```

3. Ollama 모델 확인

```sh
ollama list
```

4. Ollama 모델 실행

```sh
ollama run EEVE-Korean-10.8B:latest
```
