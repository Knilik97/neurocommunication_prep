# Конектимся к платформе HuggingFace
from huggingface_hub import login
import os

# Получение токена из секретов Colab
hf_token = os.getenv("huggingface")

# Аутентификация
login(token=hf_token)

# Вспомогательная ф-ция для модели
def messages_to_prompt(messages):
    # Инициализируем пустую строку для хранения итогового промпта
    prompt = ""

    # Проходим по каждому сообщению в списке сообщений
    for message in messages:
        # Если роль сообщения 'system', добавляем его в промпт с соответствующими тегами
        if message.role == 'system':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        # Если роль сообщения 'user', добавляем его в промпт с соответствующими тегами
        elif message.role == 'user':
            prompt += f"<s>{message.role}\n{message.content}</s>\n"
        # Если роль сообщения 'bot', добавляем только начальный тег для бота
        elif message.role == 'bot':
            prompt += f"<s>bot\n"

    # Проверяем, начинается ли промпт с системного сообщения, если нет, добавляем пустое системное сообщение
    if not prompt.startswith("<s>system\n"):
        prompt = "<s>system\n</s>\n" + prompt

    # Добавляем финальный начальный тег для бота в конец промпта
    prompt = prompt + "<s>bot\n"

    # Возвращаем итоговый промпт
    return prompt

def completion_to_prompt(completion):
    # Формируем промпт из завершения, добавляя теги для системного и пользовательского сообщений
    return f"<s>system\n</s>\n<s>user\n{completion}</s>\n<s>bot\n"

# Загрузка модели saiga_7b и квантование
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
#для настройки и управления параметрами обучения моделей машинного обучения с использованием методов эффективного обучения (Parameter-Efficient Fine-Tuning, PEFT).
from peft import PeftConfig, PeftModel

# Определяем параметры квантования, иначе модель не выполниться в колабе
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Задаем имя модели
MODEL_NAME = "IlyaGusev/saiga_mistral_7b"

# Создание конфига, соответствующего методу PEFT (в нашем случае LoRA)
config = PeftConfig.from_pretrained(MODEL_NAME)

# Загружаем базовую модель, ее имя берем из конфига для LoRA
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,          # идентификатор модели
    quantization_config=quantization_config, # параметры квантования
    torch_dtype=torch.float16,               # тип данных
    device_map="auto"                        # автоматический выбор типа устройства
)

# Загружаем LoRA модель
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)

# Переводим модель в режим инференса
# Можно не переводить, но явное всегда лучше неявного
model.eval()

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Загружаем конфигурацию генерации из предобученной модели.
# MODEL_NAME - это строка, содержащая имя модели, из которой мы хотим получить конфигурацию.
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

# Указываем путь для сохранения квантованной модели
SAVE_PATH = "quantized_model_saiga"

# Сохраняем модель с настройками квантования
model.save_pretrained(
    SAVE_PATH,
    safe_serialization=False  # Важно для сохранения квантованной версии
)

# Сохраняем токенизатор
tokenizer.save_pretrained(SAVE_PATH)

# Передача в класс ранее объявленные вспомогательные функции
llm = HuggingFaceLLM(
    model=model,             # модель
    model_name=MODEL_NAME,   # идентификатор модели
    tokenizer=tokenizer,     # токенизатор
    max_new_tokens=generation_config.max_new_tokens, # параметр необходимо использовать здесь, и не использовать в generate_kwargs, иначе ошибка двойного использования
    model_kwargs={"quantization_config": quantization_config}, # параметры квантования
    generate_kwargs = {   # параметры для инференса
      "bos_token_id": generation_config.bos_token_id, # токен начала последовательности
      "eos_token_id": generation_config.eos_token_id, # токен окончания последовательности
      "pad_token_id": generation_config.pad_token_id, # токен пакетной обработки (указывает, что последовательность ещё не завершена)
      "no_repeat_ngram_size": generation_config.no_repeat_ngram_size,
      "repetition_penalty": generation_config.repetition_penalty,
      "temperature": generation_config.temperature,
      "do_sample": True,
      "top_k": 50,
      "top_p": 0.95
    },
    messages_to_prompt=messages_to_prompt,     # функция для преобразования сообщений к внутреннему формату
    completion_to_prompt=completion_to_prompt, # функции для генерации текста
    device_map="auto",                         # автоматически определять устройство
)

# Загрузка текстового файла (.txt)

# Чтение содержимого текстового файла
file_path = '/content/NeyroWork.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read(500)  # Прочитает первые 100 символов
print(text)

from llama_index.core import SimpleDirectoryReader
import os

file_path = '/content/reader/NeyroWork.txt'

# Проверяем существование файла
if os.path.exists(file_path):
    print("Файл найден")
    # Правильно загружаем документ через ридер
    documents = SimpleDirectoryReader(
        input_files=[file_path]  # указываем список файлов
    ).load_data()
else:
    print(f"Файл не найден по пути: {file_path}")

# Импортируем класс HuggingFaceEmbeddings из модуля langchain_huggingface
from langchain_huggingface import HuggingFaceEmbeddings

# Создаем объект embed_model, используя класс LangchainEmbedding
# В качестве параметра передаем объект HuggingFaceEmbeddings, который инициализируется с моделью
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Установка модели языковой модели (LLM)
Settings.llm = llm

# Установка модели для встраивания (embedding model)
Settings.embed_model = embed_model

# Установка размера фрагмента (chunk size) для обработки данных
Settings.chunk_size = 512

# Импортируем необходимые классы из библиотеки llama_index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Создаем индекс из документов
# Предполагается, что 'documents' - это переменная, содержащая список документов для индексации
index = VectorStoreIndex.from_documents(documents)

# Подготавливаем движок для выполнения запросов к индексу
query_engine = index.as_query_engine()

response = query_engine.query("Кто такой грузчик ? .")
print(response)
