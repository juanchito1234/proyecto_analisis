import matplotlib.pyplot as plt
import os
import psutil
from src.middlewares.metrics import global_metrics

def plot_metrics(N: int, total_time: float, peak_ram_mb: float, loss: float):
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Gráfico de Métricas Generales (Rendimiento)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Rendimiento QNodes (N={N}) - Exact Math Optimization', fontsize=16)
    
    # Datos de Caché
    hits_exact = global_metrics.ncube_cache_hits
    hits_subset = global_metrics.ncube_subset_hits
    misses = global_metrics.ncube_cache_misses
    
    total_calls = hits_exact + hits_subset + misses
    reuse_pct = ((hits_exact + hits_subset) / total_calls * 100) if total_calls > 0 else 0
    
    cache_labels = ['Exact Hits', 'Subset Hits', 'Hard Misses']
    cache_values = [hits_exact, hits_subset, misses]
    cache_colors = ['#2ca02c', '#1f77b4', '#d62728']
    
    axs[0].bar(cache_labels, cache_values, color=cache_colors)
    axs[0].set_title(f'Uso de Caché (Reuso: {reuse_pct:.2f}%)')
    axs[0].set_ylabel('Cantidad de Peticiones')
    for i, v in enumerate(cache_values):
        axs[0].text(i, v + (max(cache_values)*0.01), str(v), ha='center', fontweight='bold')
        
    # Datos Generales de Tiempo y RAM
    stats_labels = ['Tiempo Total (s)', 'RAM Pico (MB)', 'Pérdida (φ)']
    stats_values = [total_time, peak_ram_mb, loss]
    
    axs[1].bar(stats_labels, stats_values, color=['#ff7f0e', '#9467bd', '#e377c2'])
    axs[1].set_title('Métricas Globales')
    for i, v in enumerate(stats_values):
        axs[1].text(i, v + (max(stats_values)*0.01), f'{v:.2f}', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'results/plots/performance_N{N}.png', dpi=150)
    plt.close()
    
    # 2. Gráfico de Tiempos Acumulados por Función
    plt.figure(figsize=(10, 6))
    times_labels = ['Marginalizar', 'F. Submodular', 'Bipartir', 'Distribución', 'EMD']
    times_values = [
        global_metrics.time_marginalizar,
        global_metrics.time_funcion_submodular,
        global_metrics.time_bipartir,
        global_metrics.time_distribucion,
        global_metrics.time_emd
    ]
    
    # Limpiar valores negativos o ruidosos de time.perf_counter en wrappers
    times_values = [max(0, t) for t in times_values]
    
    bars = plt.bar(times_labels, times_values, color='#17becf')
    plt.title(f'Tiempos Acumulados por Función (N={N})', fontsize=14)
    plt.ylabel('Tiempo Total (s)')
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(times_values)*0.01), f'{yval:.2f}s', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'results/plots/times_N{N}.png', dpi=150)
    plt.close()
