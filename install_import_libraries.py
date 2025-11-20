# LlamaIndex, как высказалась мой куратор, этот фреймворк как вьючная змея, и ей нужны правильные зависимости, установлю через файл requirements.txt

%%writefile requirements.txt
transformers>=4.42.0
llama_index==0.10.46
pyvis==0.3.2
Ipython==7.34.0
langchain==0.2.5
pypdf==4.2.0
langchain_community==0.2.5
llama-index-llms-huggingface==0.2.3
llama-index-embeddings-huggingface==0.2.2
llama-index-embeddings-langchain==0.1.2
langchain-huggingface==0.0.3
sentencepiece==0.1.99
accelerate==0.31.0
bitsandbytes==0.43.1
peft==0.11.1
llama-index-readers-wikipedia==0.1.4
wikipedia==1.4.0
triton<=3.0 # Ещё необходимый модуль для загрузки модели

huggingface-hub==0.23.3
torch==2.3.1
packaging==24.1
pyyaml==6.0.1
requests==2.31.0
tqdm==4.66.4
filelock==3.14.0
regex==2024.5.15
typing-extensions==4.12.2
safetensors==0.4.3
tokenizers==0.19.1

# Устанавовка библиотек через файл 
!pip install -r requirements.txt

# Импортируем классы и функции из модуля llama_index.core
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.core import StorageContext

# Импортируем классы для работы с эмбеддингами и языковыми моделями из HuggingFace
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# Импортируем классы и функции из библиотеки PEFT для адаптации моделей
from peft import PeftModel, PeftConfig

# Импортируем классы и функции из библиотеки Transformers для работы с языковыми моделями
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

# Импортируем библиотеку PyTorch для работы с тензорами и моделями
import torch

# Импортируем класс для работы с эмбеддингами из Langchain
from llama_index.embeddings.langchain import LangchainEmbedding

# Импортируем класс для работы с эмбеддингами из Langchain и Hugging Face
from langchain_huggingface import HuggingFaceEmbeddings

