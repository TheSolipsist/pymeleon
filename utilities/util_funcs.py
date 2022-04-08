from time import perf_counter
import functools

def timer(func):
    """
    Timer decorator
    
    Causes a function to return the tuple (return_value, total_time):
        return_value:   the function's return value
        total_time:     the time it took for the function to execute
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func_val = func(*args, **kwargs)
        end = perf_counter()
        total_time = end - start
        return (func_val, total_time)
    return wrapper