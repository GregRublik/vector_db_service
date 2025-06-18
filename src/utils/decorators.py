import time
from functools import wraps


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Функция '{func.__name__}' выполнена за {end-start:.4f} секунд")
        return result
    return wrapper
