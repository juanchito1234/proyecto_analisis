import time

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ncube_cache_hits = 0
        self.ncube_cache_misses = 0
        self.ncube_subset_hits = 0
        self.calls_marginalizar = 0
        self.calls_funcion_submodular = 0
        self.calls_bipartir = 0
        self.calls_distribucion = 0
        self.calls_emd = 0

        self.time_marginalizar = 0.0
        self.time_funcion_submodular = 0.0
        self.time_bipartir = 0.0
        self.time_distribucion = 0.0
        self.time_emd = 0.0

global_metrics = MetricsTracker()

def timer_decorator(metric_time_attr, metric_calls_attr):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            setattr(global_metrics, metric_time_attr, getattr(global_metrics, metric_time_attr) + duration)
            setattr(global_metrics, metric_calls_attr, getattr(global_metrics, metric_calls_attr) + 1)
            return result
        return wrapper
    return decorator
