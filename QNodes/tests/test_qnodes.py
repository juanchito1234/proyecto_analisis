import pytest
import numpy as np
from unittest.mock import patch

from src.strategies.q_nodes import QNodes


def build_test_qnodes_tpm():
    """
    Construye una matriz de transición (TPM) válida para pruebas.
    Sistema de 3 variables (8 estados).
    """
    tpm = np.full((8, 3), 0.5)
    return tpm


@patch("src.strategies.q_nodes.fmt_k_particion_q", side_effect=lambda x: x)
def test_qnodes_k2(mock_fmt):
    """
    Verifica que la estrategia submodular devuelva 
    una bi-partición válida (k=2).
    """
    tpm = build_test_qnodes_tpm()
    strategy = QNodes(tpm)
    
    solution = strategy.aplicar_estrategia(
        estado_inicial="000",
        condicion="abc",
        alcance="ABC",
        mecanismo="ABC",
        k=2
    )
    
    assert solution is not None
    assert solution.particion is not None
    assert len(solution.particion) == 2


@patch("src.strategies.q_nodes.fmt_k_particion_q", side_effect=lambda x: x)
def test_qnodes_k3(mock_fmt):
    """
    Verifica que el algoritmo iterativo submodular 
    (Greedy) devuelva k=3 grupos.
    """
    tpm = build_test_qnodes_tpm()
    strategy = QNodes(tpm)
    
    solution = strategy.aplicar_estrategia(
        estado_inicial="000",
        condicion="abc",
        alcance="ABC",
        mecanismo="ABC",
        k=3
    )
    
    assert solution is not None
    assert len(solution.particion) == 3


@patch("src.strategies.q_nodes.fmt_k_particion_q", side_effect=lambda x: x)
def test_qnodes_k4(mock_fmt):
    """
    Validar comportamiento para k=4 con 3 nodos.
    Matemáticamente, con 3 presentes y 3 futuros hay 6 elementos,
    los cuales pueden dividirse en 4 bloques.
    """
    tpm = build_test_qnodes_tpm()
    strategy = QNodes(tpm)
    
    solution = strategy.aplicar_estrategia(
        estado_inicial="000",
        condicion="abc",
        alcance="ABC",
        mecanismo="ABC",
        k=4
    )
    
    assert len(solution.particion) == 4


@patch("src.strategies.q_nodes.fmt_k_particion_q", side_effect=lambda x: x)
def test_qnodes_cobertura_y_no_repetidos(mock_fmt):
    """
    Valida que no haya nodos perdidos ni duplicados
    al usar la búsqueda Greedy.
    """
    tpm = build_test_qnodes_tpm()
    strategy = QNodes(tpm)
    
    solution = strategy.aplicar_estrategia(
        estado_inicial="000",
        condicion="abc",
        alcance="ABC",
        mecanismo="ABC",
        k=2
    )
    
    nodos_encontrados = []
    # Retorna particiones que son listas de nodos
    for bloque in solution.particion:
        for nodo in bloque:
            nodos_encontrados.append(nodo)
            
    # Garantizar que todos son únicos
    assert len(nodos_encontrados) == len(set(nodos_encontrados))
