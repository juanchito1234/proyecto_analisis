import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

# Root of GeoMIP
GEOMIP_ROOT = Path(__file__).resolve().parents[4]

def generar_graficos(n_size, k_size, total_time, peak_ram_mb, hits, misses, cache_porcentaje, function_times):
    """
    Genera dos gráficos PNG en results/plots/:
    1. performance_N{N}_K{K}.png
    2. temporal_N{N}_K{K}.png
    """
    plots_dir = GEOMIP_ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # === 1. performance_N{N}.png ===
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # We will display the data as a table in the plot
    table_data = [
        ["Métrica", "Valor"],
        ["Tiempo total (s)", f"{total_time:.4f}"],
        ["RAM pico (MB)", f"{peak_ram_mb:.4f}"],
        ["Cache hits", str(hits)],
        ["Cache misses", str(misses)],
        ["Porcentaje de reutilización (%)", f"{cache_porcentaje:.2f}"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    plt.title(f"Performance N={n_size} K={k_size}", fontsize=16, fontweight='bold', pad=20)
    
    perf_path = plots_dir / f"performance_N{n_size}_K{k_size}.png"
    plt.savefig(perf_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # === 2. temporal_N{N}_K{K}.png ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    funciones = list(function_times.keys())
    tiempos = list(function_times.values())
    
    # Calcular "Otros"
    suma_tiempos = sum(tiempos)
    otros = max(0.0, total_time - suma_tiempos)
    funciones.append("Otros")
    tiempos.append(otros)
    
    y_pos = np.arange(len(funciones))
    bars = ax.barh(y_pos, tiempos, align='center', color='skyblue', edgecolor='black')
    
    ax.set_yticks(y_pos, labels=funciones)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Tiempo Acumulado (segundos)')
    ax.set_title(f"Tiempo acumulado por función (N={n_size} K={k_size})", fontsize=14, fontweight='bold')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f}s',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')
                    
    # Prevent cutting off labels
    plt.tight_layout()
    
    temp_path = plots_dir / f"temporal_N{n_size}_K{k_size}.png"
    plt.savefig(temp_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n[Plotter] Gráficos generados en: {plots_dir}")
