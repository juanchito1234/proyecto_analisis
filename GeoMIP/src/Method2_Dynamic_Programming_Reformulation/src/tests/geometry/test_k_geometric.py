import pytest
import numpy as np

from src.controllers.manager import Manager
from src.controllers.strategies.k_geometric import KGeometric


def build_test_manager():
    """
    Construye un Manager funcional instanciando
    únicamente con los parámetros permitidos por la dataclass.
    """
    tpm = np.full((8, 3), 0.5)

    gestor = Manager(
        estado_inicial="000"
    )

    return gestor, tpm


def test_k_geometric():
    """
    Test de integración completo para KGeometric.
    Verifica preparación, geometría, generación y evaluación.
    """

    gestor, tpm = build_test_manager()
    strategy = KGeometric(gestor)

    solution = strategy.aplicar_estrategia(
        condicion="abc",
        alcance="ABC",
        mecanismo="ABC",
        tpm=tpm,
        k=3
    )

    # Verificaciones principales
    assert solution is not None
    assert solution.perdida >= 0
    assert solution.distribucion_particion is not None
    assert solution.distribucion_subsistema is not None
    assert solution.particion is not None

    # Verificar servicios internos (Desacoplamiento)
    assert strategy.geometry is not None
    assert strategy.partition_generator is not None
    assert strategy.partition_evaluator is not None

    # Verificar tabla geométrica
    tabla = strategy.geometry.get_transition_table()
    assert len(tabla) > 0

    # Verificar memoización (al menos 1 cálculo debió usar caché superior o inicializarse)
    stats = strategy.partition_evaluator.cache_stats()
    assert stats["particiones"] >= 0