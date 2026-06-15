import time
from functools import wraps

# Registro global de tiempo acumulado
function_times = {
    "marginalizar": 0.0,
    "funcion_submodular": 0.0,
    "bipartir": 0.0,
    "distribucion": 0.0,
    "emd_efecto": 0.0
}

def track_time(name):
    """
    Decorador para acumular el tiempo de ejecución de una función en `function_times`.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            if name in function_times:
                function_times[name] += duration
            return result
        return wrapper
    return decorator

def reset_times():
    """Reinicia todos los contadores de tiempo."""
    for k in function_times:
        function_times[k] = 0.0
