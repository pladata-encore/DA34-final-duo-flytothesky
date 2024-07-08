#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
#   <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
#   <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> Join Discord if you need help + support us if you can!
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://github.com/unslothai/unsloth#installation-instructions---conda).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save) (eg for Llama.cpp).
# 
# **[NEW] Llama-3 8b is trained on a crazy 15 trillion tokens! Llama-2 was 2 trillion.**

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import zipfile
import os

# Google Drive의 폴더 경로
folder_path = '/content/drive/MyDrive/02.라벨링데이터' #todo

# 폴더 내의 모든 파일 리스트 가져오기
file_list = os.listdir(folder_path)

# ZIP 파일 찾기 및 풀기
for file_name in file_list:
    if file_name.endswith('.zip'):
        file_path = os.path.join(folder_path, file_name)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)  # ZIP 파일을 동일한 폴더에 풉니다
            print(f'Extracted {file_name}')


# In[ ]:


import json
import os

# JSON 파일이 있는 디렉토리
json_files_dir = '/content/drive/MyDrive/02.라벨링데이터'

# 디렉토리 내의 JSON 파일 목록 가져오기
json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

# JSON 파일의 숫자 출력
print(f"디렉토리 내의 JSON 파일 개수: {len(json_files)}")

# 첫 번째 JSON 파일의 내용 출력
if json_files:
    first_file_path = os.path.join(json_files_dir, json_files[0])
    with open(first_file_path, 'r', encoding='utf-8') as first_file:
        first_file_content = json.load(first_file)

    # 첫 번째 파일의 내용 출력
    print(f"첫 번째 JSON 파일 ({json_files[0]}) 내용:")
    print(json.dumps(first_file_content, ensure_ascii=False, indent=4))
else:
    print("디렉토리에 JSON 파일이 없습니다.")


# In[ ]:


import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# JSON 파일들이 있는 디렉토리 경로
json_files_dir = '/content/drive/MyDrive/02.라벨링데이터'

# 포맷 문자열
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Tokenizer EOS 토큰
EOS_TOKEN = '<EOS>'  # tokenizer.eos_token # Must add EOS_TOKEN

# 변환 함수
def formatting_prompts_func(data):
    formatted_data = []
    for dialog in data["Dialogs"]:
        text = alpaca_prompt.format(
            "Translate the following slang to a more standard form.",
            dialog["SpeakerText"],
            dialog["TextConvert"]
        ) + EOS_TOKEN
        formatted_data.append({"text": text})
    return formatted_data

# JSON 파일을 처리하는 함수
def process_json_file(filename):
    json_file_path = os.path.join(json_files_dir, filename)
    try:
        # JSON 파일 열기
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 변환 작업 수행
        return formatting_prompts_func(data)
    except ValueError as e:
        print(f'Error reading {json_file_path}: {e}')
        return []

# 사용 가능한 CPU 코어 수 확인
num_cores = multiprocessing.cpu_count()
print(f'사용 가능한 CPU 코어 수: {num_cores}')

# 모든 JSON 파일 순회 및 병렬 처리
if __name__ == "__main__":
    json_files = [f for f in os.listdir(json_files_dir) if f.endswith('.json')]

    formatted_data = []
    batch_size = 100  # 주기적으로 저장할 배치 크기
    output_file_path = '/content/formatted_data_partial.json'

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_file = {executor.submit(process_json_file, filename): filename for filename in json_files}

        for i, future in enumerate(as_completed(future_to_file)):
            result = future.result()
            formatted_data.extend(result)

            # 주기적으로 데이터를 파일에 저장
            if (i + 1) % batch_size == 0:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_data, f, indent=2, ensure_ascii=False)
                print(f'Partial data saved up to {i + 1} files.')

    # 최종 결과 저장
    output_file_path = '/content/drive/MyDrive/formatted_data'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)

    print(f'The formatted data has been saved to {output_file_path}')

    # 변환된 데이터 파일 열기 및 출력
    with open(output_file_path, 'r', encoding='utf-8') as f:
        formatted_data = json.load(f)

    # JSON 데이터 출력
    print(json.dumps(formatted_data, indent=2, ensure_ascii=False))


# In[ ]:


len(formatted_data)
output_file_path = '/content/formatted_data_final.json'
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=2, ensure_ascii=False)

print(f'The formatted data has been saved to {output_file_path}')

# 변환된 데이터 파일 열기 및 출력
with open(output_file_path, 'r', encoding='utf-8') as f:
    formatted_data = json.load(f)


# In[ ]:


len(formatted_data)


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import torch\nmajor_version, minor_version = torch.cuda.get_device_capability()\n# Must install separately since Colab has torch 2.2.1, which breaks packages\n!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"\nif major_version >= 8:\n    # Use this for new GPUs like Ampere, Hopper GPUs (RTX 30xx, RTX 40xx, A100, H100, L40)\n    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes\nelse:\n    # Use this for older GPUs (V100, Tesla T4, RTX 20xx)\n    !pip install --no-deps xformers trl peft accelerate bitsandbytes\npass\n')


# * We support Llama, Mistral, CodeLlama, TinyLlama, Vicuna, Open Hermes etc
# * And Yi, Qwen ([llamafied](https://huggingface.co/models?sort=trending&search=qwen+llama)), Deepseek, all Llama, Mistral derived archs.
# * We support 16bit LoRA or 4bit QLoRA. Both 2x faster.
# * `max_seq_length` can be set to anything, since we do automatic RoPE Scaling via [kaiokendev's](https://kaiokendev.github.io/til) method.
# * [**NEW**] With [PR 26037](https://github.com/huggingface/transformers/pull/26037), we support downloading 4bit models **4x faster**! [Our repo](https://huggingface.co/unsloth) has Llama, Mistral 4bit models.

# In[ ]:


from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/mistral-7b-bnb-4bit",
#     "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
#     "unsloth/llama-2-7b-bnb-4bit",
#     "unsloth/gemma-7b-bnb-4bit",
#     "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
#     "unsloth/gemma-2b-bnb-4bit",
#     "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
#     "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
# ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# We now add LoRA adapters so we only need to update 1 to 10% of all parameters!

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# <a name="Data"></a>
# ### Data Prep
# We now use the Alpaca dataset from [yahma](https://huggingface.co/datasets/yahma/alpaca-cleaned), which is a filtered version of 52K of the original [Alpaca dataset](https://crfm.stanford.edu/2023/03/13/alpaca.html). You can replace this code section with your own data prep.
# 
# **[NOTE]** To train only on completions (ignoring the user's input) read TRL's docs [here](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).
# 
# **[NOTE]** Remember to add the **EOS_TOKEN** to the tokenized output!! Otherwise you'll get infinite generations!
# 
# If you want to use the `ChatML` template for ShareGPT datasets, try our conversational [notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing).
# 
# For text completions like novel writing, try this [notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing).

# In[ ]:





# In[ ]:


from datasets import Dataset
dataset = formatted_data
#from datasets import load_dataset
#dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
#dataset = dataset.map(formatting_prompts_func, batched = True,)

# datasets.Dataset 형식으로 변환
dataset = Dataset.from_dict({"text": [entry["text"] for entry in formatted_data]})

# 데이터 확인
print(dataset)
print(formatted_data[1:100])


# <a name="Train"></a>
# ### Train the model
# Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. We also support TRL's `DPOTrainer`!

# In[ ]:


from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# In[ ]:


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


trainer_stats = trainer.train()


# In[ ]:


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can change the instruction and input - leave the output blank!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[ ]:


# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Translate the following slang to a more standard form.", # instruction
        "오대기할거요", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[ ]:


model.save_pretrained("lora_model") # Local saving
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving


# In[ ]:


from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 모델을 Google Drive에 저장
model.save_pretrained("/content/drive/MyDrive/lora_model")


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# You can also use Hugging Face's `AutoModelForPeftCausalLM`. Only use this if you do not have `unsloth` installed. It can be hopelessly slow, since `4bit` model downloading is not supported, and Unsloth's **inference is 2x faster**.

# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in `llama.cpp` or a UI based system like `GPT4All`. You can install GPT4All by going [here](https://gpt4all.io/index.html).

# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/u54VK8m8tk) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Zephyr DPO 2x faster [free Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)
# 2. Llama 7b 2x faster [free Colab](https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing)
# 3. TinyLlama 4x faster full Alpaca 52K in 1 hour [free Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)
# 4. CodeLlama 34b 2x faster [A100 on Colab](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing)
# 5. Mistral 7b [free Kaggle version](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook)
# 6. We also did a [blog](https://huggingface.co/blog/unsloth-trl) with 🤗 HuggingFace, and we're in the TRL [docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth)!
# 7. `ChatML` for ShareGPT datasets, [conversational notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing)
# 8. Text completions like novel writing [notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)
# 
# <div class="align-center">
#   <a href="https://github.com/unslothai/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/u54VK8m8tk"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://ko-fi.com/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Kofi button.png" width="145"></a></a> Support our work if you can! Thanks!
# </div>
