import os
import time
import numpy as np
from pathlib import Path

def generar_tpm(n_nodos: int, version: str):
    """
    Genera una matriz TPM aleatoria para pruebas de escalabilidad y la guarda
    en los directorios de muestras de QNodes y GeoMIP.
    
    Args:
        n_nodos (int): Tamaño de la red (número de nodos).
        version (str): Sufijo identificador de la red (ej: 'AUTO25').
    """
    print(f"\n--- Iniciando generación de TPM para N={n_nodos} (Versión: {version}) ---")
    
    # 1. Rutas de destino
    root_dir = Path(__file__).parent.resolve()
    qnodes_dir = root_dir / "QNodes" / "src" / ".samples"
    geomip_dir = root_dir / "GeoMIP" / "data" / "samples"
    
    # Crear directorios si no existen
    qnodes_dir.mkdir(parents=True, exist_ok=True)
    geomip_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombres de archivo esperados por los Managers
    filename = f"N{n_nodos}{version}.csv"
    qnodes_path = qnodes_dir / filename
    geomip_path = geomip_dir / filename
    
    # 2. Generar matriz TPM aleatoria
    num_estados = 1 << n_nodos  # 2^n_nodos
    total_size_gb = (num_estados * n_nodos) / (1024**3)
    
    print(f"Dimensiones de la matriz: {num_estados} x {n_nodos}")
    print(f"Memoria RAM estimada: {total_size_gb:.4f} GB")
    
    start_time = time.time()
    print("Generando estados aleatorios...")
    # Matriz determinista simulando TPMs causales (como N10A)
    # Se usa int8 para minimizar uso de RAM
    tpm = np.random.randint(2, size=(num_estados, n_nodos), dtype=np.int8)
    print(f"Generación completada en {time.time() - start_time:.2f} segundos.")
    
    # 3. Guardar matriz en disco
    print("Guardando archivos CSV...")
    
    # Función auxiliar para guardar y medir
    def guardar_y_medir(path, matrix):
        t0 = time.time()
        # Se usa ',' como delimitador, que es el estandar en los archivos de muestra (N3A, N6A, etc)
        np.savetxt(path, matrix, delimiter=',', fmt="%d")
        t1 = time.time()
        size_gb = os.path.getsize(path) / (1024**3)
        print(f" - {path.name} guardado en {path.parent.name}/ ({size_gb:.4f} GB) en {t1 - t0:.2f} s")

    guardar_y_medir(qnodes_path, tpm)
    guardar_y_medir(geomip_path, tpm)
    
    print("\n¡Generación y guardado exitosos!")
    print("Para ejecutar en el proyecto principal:")
    print(f'   K = 2\n   N = {n_nodos}\n   VERSION = "{version}"\n')
    
    return qnodes_path, geomip_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generador automático de TPMs")
    parser.add_argument("--nodos", type=int, default=16, help="Número de nodos")
    parser.add_argument("--version", type=str, default="AUTO", help="Versión identificadora (ej: AUTO)")
    args = parser.parse_args()
    
    generar_tpm(args.nodos, args.version)
