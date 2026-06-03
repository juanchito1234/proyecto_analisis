import pytest
import numpy as np
from src.models.core.system import System
from src.models.partitions.partition_evaluator import PartitionEvaluator
from src.constants.base import ACTUAL, EFECTO

@pytest.fixture
def dummy_system_eval():
    """
    Sistema determinista mínimo para asegurar que la EMD 
    pueda ser no cero si hay un particionamiento lesivo.
    """
    tpm = np.full((8, 3), 0.5)
    estado_inicio = np.array([0, 0, 0], dtype=np.int8)
    return System(tpm=tpm, estado_inicio=estado_inicio)

def mock_split_partition_by_time(partition):
    """
    Función helper idéntica a la empleada por KGeometric
    para convertir el modelo abstracto al modelo concreto
    (alcance, mecanismo).
    """
    resultado = []
    for bloque in partition:
        alcance = []
        mecanismo = []
        for tipo, nodo in bloque:
            if tipo == EFECTO: 
                alcance.append(nodo)
            elif tipo == ACTUAL: 
                mecanismo.append(nodo)
        resultado.append((np.array(sorted(alcance), dtype=np.int8), 
                          np.array(sorted(mecanismo), dtype=np.int8)))
    return resultado

def test_evaluacion_particion_valida(dummy_system_eval):
    """
    Garantizar que una partición bi-partita válida retorne la
    pérdida (float) y la distribución marginal del sistema
    particionado.
    """
    dist_original = dummy_system_eval.distribucion_marginal()
    evaluator = PartitionEvaluator(dummy_system_eval, dist_original)
    
    particion = [
        {(EFECTO, 0), (EFECTO, 1), (ACTUAL, 0), (ACTUAL, 1)},
        {(EFECTO, 2), (ACTUAL, 2)}
    ]
    
    perdida, dist_part = evaluator.evaluate_partition(particion, mock_split_partition_by_time)
    
    # Earth Mover's Distance no puede ser negativa
    assert perdida >= 0
    # Al particionar 3 nodos, el tensor de distribuciones sigue siendo para 3 variables
    assert len(dist_part) == len(dist_original)
    
def test_memoizacion_cache_reutilizado(dummy_system_eval):
    """
    Valida la optimización: si se pide evaluar la misma estructura 
    dos veces, no debe re-calcularse ni modificar el cache superior a 1.
    """
    dist_original = dummy_system_eval.distribucion_marginal()
    evaluator = PartitionEvaluator(dummy_system_eval, dist_original)
    
    particion = [
        {(EFECTO, 0), (ACTUAL, 0)},
        {(EFECTO, 1), (EFECTO, 2), (ACTUAL, 1), (ACTUAL, 2)}
    ]
    
    # Primera evaluación
    evaluator.evaluate_partition(particion, mock_split_partition_by_time)
    stats1 = evaluator.cache_stats()
    assert stats1["particiones"] == 1
    
    # Segunda evaluación idéntica
    evaluator.evaluate_partition(particion, mock_split_partition_by_time)
    stats2 = evaluator.cache_stats()
    
    # El cache no debe haber crecido, la clave fue interceptada
    assert stats2["particiones"] == 1

def test_consistencia_ejecuciones(dummy_system_eval):
    """
    Comprobar el determinismo: la EMD tras evaluar, limpiar caché 
    y volver a evaluar DEBE arrojar exactamente el mismo float.
    """
    dist_original = dummy_system_eval.distribucion_marginal()
    evaluator = PartitionEvaluator(dummy_system_eval, dist_original)
    
    particion = [
        {(EFECTO, 0), (ACTUAL, 0)},
        {(EFECTO, 1), (EFECTO, 2), (ACTUAL, 1), (ACTUAL, 2)}
    ]
    
    perdida1, _ = evaluator.evaluate_partition(particion, mock_split_partition_by_time)
    
    evaluator.clear_cache()
    # Verifica que el limpiar cache funcione
    assert evaluator.cache_stats()["particiones"] == 0
    
    perdida2, _ = evaluator.evaluate_partition(particion, mock_split_partition_by_time)
    
    assert perdida1 == perdida2

def test_particion_invalida_vacia(dummy_system_eval):
    """
    Robusted: enviar una partición vacía debe arrojar un error semántico.
    """
    dist_original = dummy_system_eval.distribucion_marginal()
    evaluator = PartitionEvaluator(dummy_system_eval, dist_original)
    
    with pytest.raises(ValueError):
        evaluator.evaluate_partition([], mock_split_partition_by_time)
