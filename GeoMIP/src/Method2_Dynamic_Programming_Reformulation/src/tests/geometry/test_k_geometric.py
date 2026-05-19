import numpy as np

from src.controllers.manager import Manager

from src.controllers.strategies.k_geometric import (
    KGeometric
)


def build_test_manager():
    """
    Construye un Manager mínimo
    funcional para pruebas unitarias
    de KGeometric.

    Aquí debes adaptar únicamente
    los parámetros reales que exige
    tu clase Manager.
    """

    """
    TPM mínima binaria 3x3.

    Esta TPM es solamente un ejemplo.
    Usa una TPM válida de tu sistema.
    """

    tpm = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])

    """
    IMPORTANTE:
    ------------

    Debes reemplazar esto por la
    forma REAL en que inicializas
    Manager dentro de tu proyecto.
    """

    gestor = Manager(
        estado_inicial="000",
        sistema=tpm,
        pagina="test"
    )

    return gestor, tpm


def test_k_geometric():
    """
    Test de integración completo
    para KGeometric.

    Este test verifica:

    - preparación del subsistema
    - geometría de transición
    - generación de candidatos
    - evaluación EMD
    - memoización
    - solución final
    """

    gestor, tpm = (
        build_test_manager()
    )

    strategy = KGeometric(
        gestor
    )

    solution = (
        strategy.aplicar_estrategia(
            condicion="",
            alcance="ABC",
            mecanismo="ABC",
            tpm=tpm,
            k=3
        )
    )

    """
    Verificaciones principales.
    """

    assert solution is not None

    assert solution.perdida >= 0

    assert (
        solution.distribucion_particion
        is not None
    )

    assert (
        solution.distribucion_subsistema
        is not None
    )

    assert solution.particion is not None

    """
    Verificar servicios internos.
    """

    assert strategy.geometry is not None

    assert (
        strategy.partition_generator
        is not None
    )

    assert (
        strategy.partition_evaluator
        is not None
    )

    """
    Verificar tabla geométrica.
    """

    tabla = (
        strategy.geometry
        .get_transition_table()
    )

    assert len(tabla) > 0

    """
    Verificar memoización.
    """

    stats = (
        strategy.partition_evaluator
        .cache_stats()
    )

    assert stats["particiones"] > 0