#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://api.aihub.or.kr/api/aihubshell.do
get_ipython().system('curl -O https://api.aihub.or.kr/api/aihubshell.do # URL을 실제로 대체')
get_ipython().system('ls -l')
get_ipython().system('mv aihubshell.do aihubshell')
get_ipython().system('chmod +x aihubshell')
import os

os.environ['AIHUB_ID'] = 'ksjhshl@naver.com'  # 실제 AIHUB ID로 변경
os.environ['AIHUB_PW'] = 'Sjgb*2358'  # 실제 AIHUB 비밀번호로 변경



# In[ ]:


# 데이터셋 목록 조회
get_ipython().system('./aihubshell list')

# 특정 데이터셋 파일 정보 조회
datasetkey = 'your_dataset_key'  # 실제 datasetkey로 변경
get_ipython().system('./aihubshell list -datasetkey $datasetkey')

# 데이터셋 다운로드
get_ipython().system('./aihubshell download -datasetkey $datasetkey')

