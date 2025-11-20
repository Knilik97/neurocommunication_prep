# Базовая фильтрация запросов
def validate_query(query):
    # Проверяем длину запроса
    if len(query.strip()) < 3:
        raise ValueError("Запрос слишком короткий")

    # Проверяем запрещенные слова
    banned_words = ["взломать", "украсть", "незаконный", "наркотики, личные данные"]
    if any(word in query.lower() for word in banned_words):
        raise ValueError("Недопустимый запрос")

    # Проверяем тип запроса
    if not isinstance(query, str):
        raise TypeError("Запрос должен быть строкой")

    return query


# Создаем безопасный запрос
user_query = "Можно ли взломать вашу компанию?"

# Обрабатываем через фильтр
response = validate_query(user_query)

print(response)

# Задаю системный промпт
system_prompt = """
Ты — корпоративный ассистент.
* Отвечай только на основе предоставленного контекста
* Если информации нет — говори: "Данных нет"
* Не придумывай информацию
* Соблюдай деловой стиль общения
* Избегай сложных терминов без объяснения
"""

# Улучшенный запрос с фильтрацией
def safe_query(query):
    try:
        # Валидируем запрос
        validated_query = validate_query(query)

        # Добавляем системный промпт
        response = query_engine.query(
            f"{system_prompt}\n\n{validated_query}"
        )

        return response

    except Exception as e:
        return f"Ошибка обработки запроса: {str(e)}"

# Создаем безопасный запрос
user_query = "Расскажи про должностные обязанности грузчика"

# Обрабатываем через фильтр
response = safe_query(user_query)

print(response)
