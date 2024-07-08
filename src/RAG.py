#!/usr/bin/env python
# coding: utf-8

# # Google Colab으로 오픈소스 LLM 구동하기

# ## 1단계 - LLM 양자화에 필요한 패키지 설치
# - bitsandbytes: Bitsandbytes는 CUDA 사용자 정의 함수, 특히 8비트 최적화 프로그램, 행렬 곱셈(LLM.int8()) 및 양자화 함수에 대한 경량 래퍼
# - PEFT(Parameter-Efficient Fine-Tuning): 모델의 모든 매개변수를 미세 조정하지 않고도 사전 훈련된 PLM(언어 모델)을 다양한 다운스트림 애플리케이션에 효율적으로 적용 가능
# - accelerate: PyTorch 모델을 더 쉽게 여러 컴퓨터나 GPU에서 사용할 수 있게 해주는 도구
# 

# In[ ]:


#양자화에 필요한 패키지 설치
get_ipython().system('pip install -q -U bitsandbytes')
get_ipython().system('pip install -q -U git+https://github.com/huggingface/transformers.git00-')
get_ipython().system('pip install -q -U git+https://github.com/huggingface/peft.git')
get_ipython().system('pip install -q -U git+https://github.com/huggingface/accelerate.git')


# ## 2단계 - 트랜스포머에서 BitsandBytesConfig를 통해 양자화 매개변수 정의하기
# 
# 
# * load_in_4bit=True: 모델을 4비트 정밀도로 변환하고 로드하도록 지정
# * bnb_4bit_use_double_quant=True: 메모리 효율을 높이기 위해 중첩 양자화를 사용하여 추론 및 학습
# * bnd_4bit_quant_type="nf4": 4비트 통합에는 2가지 양자화 유형인 FP4와 NF4가 제공됨. NF4 dtype은 Normal Float 4를 나타내며 QLoRA 백서에 소개되어 있습니다. 기본적으로 FP4 양자화 사용
# * bnb_4bit_compute_dype=torch.bfloat16: 계산 중 사용할 dtype을 변경하는 데 사용되는 계산 dtype. 기본적으로 계산 dtype은 float32로 설정되어 있지만 계산 속도를 높이기 위해 bf16으로 설정 가능
# 
# 

# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# ## 3단계 - 경량화 모델 로드하기

# 이제 모델 ID를 지정한 다음 이전에 정의한 양자화 구성으로 로드합니다.

# 약 6분 소요

# In[ ]:


model_id = "allganize/Llama-3-Alpha-Ko-8B-Evo"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")


# In[ ]:


print(model)


# ## 4단계 - 잘 실행되는지 확인

# 약 1분 소요

# In[ ]:


device = "cuda:0"

messages = [
    {"role": "system", "content": "당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요."},
    {"role": "user", "content": "은행의 기준 금리에 대해서 설명해줘"}
]


encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
model_inputs = encodeds.to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generated_ids = model.generate(model_inputs, max_new_tokens=512, eos_token_id=terminators, do_sample=True, repetition_penalty=1.05,)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


# 이번에는 RAG로 실험해보자.

# In[ ]:


search_result = '''쏘카(대표 이재웅)가 해군(해군참모총장 심승섭 대장)과 공유경제 활성화 및 업무 효율 향상을 위한 업무 협약을 체결했다. 해군은 국군 중 최초로 법인용 차량 공유 서비스 ‘쏘카 비즈니스’를 도입한다. 쏘카와 해군은 지난 5일 서울 해군 재경근무지원대대 회의실에서 김남희 쏘카 신규사업본부장과 조정권 해군본부 군수차장과 김은석 해군본부 수송과장 등 주요 관계자가 참석한 가운데 상호협력과 발전을 위한 업무 협약식을 체결했다. 양 기관은 △해군본부 임직원의 업무 이동 효율성 향상 △공유 차량을 활용한 해군 본부 및 부대 주차난 해소 △공유 차량 이용 활성화 및 확대를 위해 적극 협력키로 했다. 쏘카는 해군 장병과 군무원을 대상으로 법인용 차량 공유 서비스 ‘쏘카 비즈니스’를 제공한다. 해군 장병들과 군무원은 업무 이동 시 전국 쏘카존에 있는 1만 2천여 대의 차량을 이용할 수 있다. 특히, 출장 시에는 전국 74개 시군의 KTX, 기차역, 버스터미널, 공항 등 대중교통과 교통 편의시설 거점이 연결된 260여 개 쏘카존을 통해 효율적인 이동이 가능해진다. 쏘카와 해군은 우선 올해까지 해군본부를 대상으로 ‘쏘카 비즈니스’ 시범 적용을 거친 후 내년부터는 해군 전 부대로 확대할 계획이다. 그전까지 일반 사병들에게는 별도로 월별 할인 혜택과 특전을 제공, 휴가와 외출 시에도 합리적인 가격으로 쏘카를 이용할 수 있도록 지원한다. 조정권 해군본부 군수차장은 “해군은 장병들의 업무 이동 편의성을 향상시키는 동시에 사기진작과 복리 증진을 위해 차량 공유 서비스 기업과 업무협약을 체결했다”며 “전문기관, 업체와의 협력을 통해 새로운 기술을 해군 수송업무에 도입해 해군이 그려나가는 ‘스마트 해군’ 건설에 한 걸음 더 다가갈 수 있도록 노력하겠다”고 말했다. 김남희 쏘카 신규사업본부장은 “일반 기업체 외에도 군이나 지자체, 공공기관 등에서도 법인용 차량 공유 서비스에 대한 수요가 꾸준히 늘어나고 있다”며 “업무 이용 패턴과 특성에 맞는 서비스 인프라와 라인업을 지속해서 확대해 나갈 것”이라고 말했다.'''

user_prompt = '검색 결과:\n' + search_result + '\n\n질문: 해군이 쏘카와 도입하는 서비스는?'

messages = [
    {"role": "system", "content": "당신은 주어진 검색 결과와 사용자의 질문을 바탕으로 답변하는 어시스턴트입니다. 검색 결과로 답변할 수 없는 내용이면 답변할 수 없다고 하세요."},
    {"role": "user", "content": user_prompt}
]

encodeds = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
model_inputs = encodeds.to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generated_ids = model.generate(model_inputs, max_new_tokens=512, eos_token_id=terminators, do_sample=True, repetition_penalty=1.05,)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


# ## 5단계- RAG 시스템 결합하기

# In[ ]:


# pip install시 utf-8, ansi 관련 오류날 경우 필요한 코드
import locale
def getpreferredencoding(do_setlocale = True):
   return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# In[ ]:


get_ipython().system('pip -q install langchain pypdf chromadb sentence-transformers faiss-gpu langchain_community')


# In[ ]:


from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.chains import LLMChain

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=True,
    max_new_tokens=300,
)

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 주어진 검색 결과와 사용자의 질문을 바탕으로 답변하는 어시스턴트입니다. 검색 결과로 답변할 수 없는 내용이면 답변할 수 없다고 하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

검색 결과: {context}

질문: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# Create llm chain
 llm_chain = LLMChain(llm=koplatyi_llm, prompt=prompt)


# In[ ]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough


# In[ ]:


from google.colab import drive

# Google 드라이브를 마운트합니다.
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('pip install jq')


# In[ ]:


# JSON 파일들이 있는 드라이브 내 디렉터리 경로
directory_path = '/content/drive/MyDrive/02.라벨링데이터' #to do

# 모든 JSON 파일을 저장할 리스트
all_documents = []

# 디렉터리 내 모든 파일 리스트
all_files = os.listdir(directory_path)

# JSON 파일 필터링
json_files = [filename for filename in all_files if filename.endswith('.json')]

# 처음 100개의 JSON 파일만 선택
selected_files = json_files[:]

# 선택된 파일에 대해 반복
for filename in selected_files:
    file_path = os.path.join(directory_path, filename)

    # JSON 파일을 UTF-8 인코딩으로 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # 문서 내용 디코딩 (필요한 경우)
        page_content = json.dumps(data, ensure_ascii=False)

        # Document 객체로 변환
        doc = Document(page_content=page_content, metadata={"source": file_path})
        all_documents.append(doc)



# In[ ]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(all_documents)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

db = FAISS.from_documents(texts, hf)
retriever = db.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k': 3}
                        )


# In[ ]:


rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)


# In[ ]:


rag_chain


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


i.metadata


# In[ ]:


result = rag_chain.invoke("귀염뽀짝이 뭐게?")

for i in result['context']:
    print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']}  \n\n")

print('--' * 50)
print(f"\n답변: {result['text']}")

