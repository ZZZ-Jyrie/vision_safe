import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        execution_time = (end_time - start_time) * 1000
        print(f"函数 {func.__name__} 运行时间: {execution_time:.6f} 豪秒")
        return result
    return wrapper