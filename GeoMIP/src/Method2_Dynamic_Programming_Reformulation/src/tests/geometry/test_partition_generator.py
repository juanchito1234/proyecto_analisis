import pytest
import numpy as np
from src.models.core.system import System
from src.models.geometry.transition_geometry import TransitionGeometry
from src.models.partitions.k_partition_generator import KPartitionGenerator
from src.constants.base import ACTUAL, EFECTO

@pytest.fixture
def dummy_system_generator():
    """
    Crea un sistema mínimo de 3 nodos y su geometría asociada
    para probar el generador de particiones de manera aislada.
    """
    tpm = np.full((8, 3), 0.5)
    estado_inicio = np.array([0, 0, 0], dtype=np.int8)
    sys = System(tpm=tpm, estado_inicio=estado_inicio)
    
    estado_fin = np.array([1, 1, 1], dtype=np.int8)
    tg = TransitionGeometry(sys, estado_inicio, estado_fin)
    tg.build_geometry()
    
    return sys, tg, tg.get_paths(), estado_fin

def _check_cobertura_completa(particion, n_nodos):
    """
    Función auxiliar para verificar que una partición cubre todos
    los nodos futuros sin repeticiones.
    """
    nodos_encontrados = set()
    for bloque in particion:
        for tipo, nodo in bloque:
            if tipo == EFECTO:
                nodos_encontrados.add(nodo)
    return len(nodos_encontrados) == n_nodos

def test_generador_k2(dummy_system_generator):
    """
    Validar que para k=2 el generador devuelve al menos las
    bi-particiones clásicas sembradas por un solo efecto.
    """
    sys, tg, paths, estado_fin = dummy_system_generator
    
    # Logger puede ser None en pruebas aisladas
    generator = KPartitionGenerator(sys, tg, paths, estado_fin, logger=None)
    candidatos = generator.identificar_particiones_candidatas(k=2)
    
    assert len(candidatos) > 0
    # Toda partición devuelta para k=2 debe tener exactamente 2 bloques
    for particion in candidatos:
        assert len(particion) == 2
        assert _check_cobertura_completa(particion, 3)

def test_generador_k3(dummy_system_generator):
    """
    Validar que el motor puede segmentar en k=3. 
    Para un sistema de 3 nodos, la única tri-partición de efectos
    es asignar cada nodo a un bloque diferente.
    """
    sys, tg, paths, estado_fin = dummy_system_generator
    generator = KPartitionGenerator(sys, tg, paths, estado_fin, logger=None)
    
    candidatos = generator.identificar_particiones_candidatas(k=3)
    
    assert len(candidatos) > 0
    for particion in candidatos:
        assert len(particion) == 3
        assert _check_cobertura_completa(particion, 3)

def test_generador_k_invalido(dummy_system_generator):
    """
    Verificar el comportamiento cuando se solicita un k mayor 
    a la cantidad de nodos. En este caso 3 nodos, si pedimos k=4,
    es imposible segmentar 3 efectos en 4 bloques no vacíos.
    """
    sys, tg, paths, estado_fin = dummy_system_generator
    generator = KPartitionGenerator(sys, tg, paths, estado_fin, logger=None)
    
    candidatos = generator.identificar_particiones_candidatas(k=4)
    # Ya que no puede asignar efectos al 4to bloque de manera que quede no-vacío,
    # el resultado de candidatos válidos debe ser 0.
    assert len(candidatos) == 0

def test_generador_sin_repetidos(dummy_system_generator):
    """
    Garantizar que el generador utiliza correctamente la firma
    hashable para no arrojar particiones duplicadas en la lista de candidatos.
    """
    sys, tg, paths, estado_fin = dummy_system_generator
    generator = KPartitionGenerator(sys, tg, paths, estado_fin, logger=None)
    
    candidatos = generator.identificar_particiones_candidatas(k=2)
    
    # Transformamos cada candidato a la firma que usa el evaluador
    firmas = set()
    for particion in candidatos:
        firma = tuple(sorted([tuple(sorted(list(bloque))) for bloque in particion]))
        firmas.add(firma)
    
    assert len(firmas) == len(candidatos)
