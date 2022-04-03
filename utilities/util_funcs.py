from time import perf_counter
import functools

def timer(func):
    """Timer decorator"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func_val = func(*args, **kwargs)
        end = perf_counter()
        total_time = end - start
        return (func_val, total_time)
    return wrapper