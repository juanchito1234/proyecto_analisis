import os
import sys
from pathlib import Path

# Añadir las rutas para poder importar los Managers
root_dir = Path(__file__).parent.resolve()

# Aseguramos de importar el manager de QNodes para probar la carga
sys.path.insert(0, str(root_dir / "QNodes"))

from generador_escalabilidad import generar_tpm

def test_escalabilidad_generacion_y_carga():
    """
    Verifica que la matriz se genera correctamente y que puede
    ser leída por el Manager de QNodes.
    """
    N = 16 # Usar 16 para la prueba
    VERSION = "TESTGEN"
    
    # 1. Generar la TPM
    q_path, g_path = generar_tpm(n_nodos=N, version=VERSION)
    
    assert q_path.exists(), "El archivo para QNodes no se creó"
    assert g_path.exists(), "El archivo para GeoMIP no se creó"
    
    # 2. Cargar usando Manager de QNodes
    from src.controllers.manager import Manager as QManager
    from src.models.base.application import aplicacion
    
    # Ajustamos la configuracion como si estuvieramos en main.py
    aplicacion.set_pagina_red_muestra(VERSION)
    estado_inicial = "1" + ("0" * (N - 1))
    
    # Como estamos corriendo desde la raíz, pasamos la ruta base explícitamente
    qnodes_samples_dir = root_dir / "QNodes" / "src" / ".samples"
    gestor = QManager(estado_inicial=estado_inicial, ruta_base=qnodes_samples_dir)
    
    print(f"\n[QNodes] Probando carga de red desde {gestor.tpm_filename}...")
    matriz_cargada = gestor.cargar_red()
    
    assert matriz_cargada is not None, "La matriz no pudo ser cargada"
    assert matriz_cargada.shape == (2**N, N), f"Shape incorrecto: {matriz_cargada.shape}"
    print(f"[QNodes] Carga exitosa. Shape verificado: {matriz_cargada.shape}")
    
    # 3. Cleanup para no dejar basura de test
    os.remove(q_path)
    os.remove(g_path)
    print("\nTest exitoso y archivos limpiados.")

if __name__ == "__main__":
    test_escalabilidad_generacion_y_carga()
