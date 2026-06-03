import pytest
import numpy as np
from src.models.core.system import System
from src.models.geometry.transition_geometry import TransitionGeometry


@pytest.fixture
def dummy_system():
    """
    Crea un sistema mínimo de 3 nodos para validación geométrica.
    Matriz de transición aleatoria o uniforme (8x3).
    """
    tpm = np.full((8, 3), 0.5)
    estado_inicio = np.array([0, 0, 0], dtype=np.int8)
    return System(tpm=tpm, estado_inicio=estado_inicio)


def test_construccion_rutas_hamming(dummy_system):
    """
    Valida que los caminos de Hamming se generen correctamente
    y alcancen la cobertura completa para un sistema de 3 variables.
    """
    estado_ini = dummy_system.estado_inicial
    estado_fin = np.array([1, 1, 1], dtype=np.int8)
    
    tg = TransitionGeometry(dummy_system, estado_ini, estado_fin)
    tg.build_geometry()
    
    paths = tg.get_paths()
    
    # 3 variables binarias implican 4 niveles (0, 1, 2, 3)
    assert len(paths) == 4
    
    # Cantidad de combinaciones por distancia Hamming: C(n, k)
    assert len(paths[0]) == 1  # 0 bit diferentes
    assert len(paths[1]) == 3  # 1 bit diferente
    assert len(paths[2]) == 3  # 2 bits diferentes
    assert len(paths[3]) == 1  # 3 bits diferentes


def test_calculo_tabla_costos(dummy_system):
    """
    Valida que el cálculo de la tabla de transiciones compute
    un array de costos por cada dimensión futura en el sistema.
    """
    estado_ini = dummy_system.estado_inicial
    estado_fin = np.array([1, 1, 1], dtype=np.int8)
    
    tg = TransitionGeometry(dummy_system, estado_ini, estado_fin)
    tg.build_geometry()
    
    tabla = tg.get_transition_table()
    
    # Debe tener entradas, en particular la del camino completo
    assert len(tabla) > 0
    key = (tuple(estado_ini), tuple(estado_fin))
    
    # Solicitamos un costo específico para forzar la inicialización si fuera lazy
    costos = tg.get_cost(estado_ini, estado_fin)
    
    # Debe devolver un valor para cada nodo futuro
    assert len(costos) == len(dummy_system.indices_ncubos)
    
    # En un sistema uniforme todos los costos base deberían ser procesados numéricamente (no Nulos)
    for c in costos:
        assert c is not None


def test_simetria_costos(dummy_system):
    """
    Valida que el cálculo geométrico del costo tx(i, j) sea direccionalmente
    consistente. tx(A->B) no necesariamente es igual a tx(B->A) en el hipercubo
    si las probabilidades son asimétricas, pero aquí validamos que no lance error
    y respete los límites de variables.
    """
    estado_ini = dummy_system.estado_inicial
    estado_fin = np.array([1, 1, 1], dtype=np.int8)
    
    tg = TransitionGeometry(dummy_system, estado_ini, estado_fin)
    
    # Forzamos lazy evaluation en ambos sentidos de la diagonal del hipercubo
    costo_ida = tg.get_cost(estado_ini, estado_fin)
    costo_vuelta = tg.get_cost(estado_fin, estado_ini)
    
    assert len(costo_ida) == 3
    assert len(costo_vuelta) == 3


def test_integridad_nodos(dummy_system):
    """
    Garantiza que la instancia reconozca todos los n-cubos que maneja el sistema.
    """
    estado_ini = dummy_system.estado_inicial
    estado_fin = np.array([1, 1, 1], dtype=np.int8)
    tg = TransitionGeometry(dummy_system, estado_ini, estado_fin)
    
    assert len(tg.idx_ncubos) == 3
    assert tg._flat_data.shape == (3, 8)
