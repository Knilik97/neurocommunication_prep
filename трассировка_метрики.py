import logging  # Импортируем модуль для работы с логированием

# Настраиваем базовую конфигурацию логирования
# Уровень INFO означает, что будут записываться все сообщения уровня INFO и выше
logging.basicConfig(level=logging.INFO)

def log_query(query):
    """
    Функция для логирования входящего запроса

    Параметры:
    query (str) - входящий запрос для обработки

    Возвращает:
    str - оригинальный запрос после логирования
    """
    # Записываем информацию о полученном запросе в лог-файл
    # Используем форматированную строку для более читаемого вывода
    logging.info(f"Получен запрос: {query}")

    # Возвращаем оригинальный запрос для дальнейшей обработки
    return query


def analyze_metrics():
    # Получаем текущие метрики из системы мониторинга
    metrics = monitoring.get_metrics()

    # Записываем информацию об анализе метрик в лог
    logging.info(f"Анализ метрик:")

    # Выводим основные показатели
    logging.info(f"Всего запросов: {metrics['total_requests']}")
    logging.info(f"Среднее время обработки: {metrics['avg_response_time']:.2f} сек")
    logging.info(f"Процент ошибок: {metrics['error_rate']*100:.2f}%")

    # Дополнительный анализ с предупреждениями
    # Проверяем время обработки запросов
    if metrics['avg_response_time'] > 5:
        logging.warning("Высокое время обработки запросов")

    # Проверяем процент ошибок
    if metrics['error_rate'] > 0.05:
        logging.warning("Высокий процент ошибок")


import logging
from datetime import datetime
import time

# Класс для мониторинга запросов и ошибок
class Monitoring:
    def __init__(self):
        # Инициализация структуры метрик для хранения статистики
        self.metrics = {
            "requests": [],          # Список всех обработанных запросов
            "response_times": [],   # Время отклика каждого запроса
            "errors": 0              # Количество ошибок
        }

    # Метод для регистрации успешного запроса
    def log_request(self, prompt, response, start_time):
        try:
            # Если ответ является объектом с полем content, извлекаем содержимое
            if hasattr(response, 'content'):
                response_data = response.content.decode('utf-8')  # Преобразование байтов в строку
            # Если ответ имеет поле text, используем его непосредственно
            elif hasattr(response, 'text'):
                response_data = response.text
            # Иначе представляем ответ строкой
            else:
                response_data = str(response)

            # Вычисление времени обработки запроса
            end_time = time.time()       # Текущее время завершения обработки
            processing_time = end_time - start_time  # Продолжительность обработки

            # Добавляем запись о запросе в список запросов
            self.metrics["requests"].append({
                "prompt": prompt,           # Запрос пользователя
                "response": response_data, # Ответ системы
                "timestamp": datetime.now(), # Точное время обработки
                "processing_time": processing_time # Длительность обработки
            })

            # Сохранение времени обработки отдельно
            self.metrics["response_times"].append(processing_time)

            # Логирование успешности обработки запроса
            logging.info(f"Запрос: {prompt[:100]}...")      # Краткий вывод запроса
            logging.info(f"Ответ: {response_data[:100]}...") # Краткий вывод ответа
            logging.info(f"Время обработки: {processing_time:.2f} сек") # Информация о длительности обработки

        except Exception as e:
            # Регистрация ошибки при сохранении запроса
            logging.error(f"Ошибка при логировании: {str(e)}")

    # Метод для записи ошибки
    def log_error(self, error):
        self.metrics["errors"] += 1     # Увеличиваем счётчик ошибок
        logging.error(f"Ошибка: {str(error)}") # Логируем ошибку в журнал

    # Получение текущих метрик мониторинга
    def get_metrics(self):
        return {
            "total_requests": len(self.metrics["requests"]), # Всего запросов
            "avg_response_time": sum(self.metrics["response_times"]) / len(self.metrics["response_times"]) if self.metrics["response_times"] else 0, # Среднее время отклика
            "error_rate": self.metrics["errors"] / len(self.metrics["requests"]) if self.metrics["requests"] else 0 # Частота ошибок
        }

# Функция для обработки запроса с мониторингом
def process_query(prompt):
    start_time = time.time()  # Начало отсчета времени обработки
    try:
        # Выполняем обработку запроса движком
        response = query_engine.query(prompt)
        # Фиксируем успешный запрос
        monitoring.log_request(prompt, response, start_time)
        return response
    except Exception as e:
        # Если произошла ошибка, фиксируем её
        monitoring.log_error(e)
        return "Произошла ошибка при обработке запроса"


# Пример использования
monitoring = Monitoring()

# Добавляем функцию вывода метрик
def show_metrics():
    metrics = monitoring.get_metrics()
    print("\n--- Метрики системы ---")
    print(f"Общее количество запросов: {metrics['total_requests']}")
    print(f"Среднее время обработки: {metrics['avg_response_time']:.2f} сек")
    print(f"Процент ошибок: {metrics['error_rate']*100:.2f}%")

    # Дополнительная статистика
    if metrics['total_requests'] > 0:
        print(f"Последнее время обработки: {monitoring.metrics['response_times'][-1]:.2f} сек")
        print(f"Всего ошибок: {monitoring.metrics['errors']}")
    else:
        print("Нет данных для отображения")

# Изменяем основной цикл для добавления возможности просмотра метрик

if __name__ == "__main__":
    while True:
        # Получаем ввод от пользователя
        user_input = input("\nВведите запрос (или 'метрики' для просмотра статистики) или 'Спасибо' для выхода: ")

        # Проверяем команду для просмотра метрик
        if user_input.lower() == 'метрики':
            show_metrics()
            continue  # Возвращаемся к началу цикла

        # Проверяем команду для выхода
        if user_input.lower() == 'спасибо':
            print("До свидания!")
            break  # Выходим из бесконечного цикла

        # Обрабатываем обычный запрос
        response = process_query(user_input)
        print(response)

        # Выполняем анализ метрик каждые 10 запросов
        if len(monitoring.metrics['requests']) % 10 == 0:
            analyze_metrics()

