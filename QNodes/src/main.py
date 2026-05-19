from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.k_q_nodes import KQNodes
from src.models.base.application import aplicacion

K = 3
N = 15
VERSION = "A"

def iniciar():
    """Punto de entrada"""

    # Generación automática basada en N
    estado_inicial = "1" + ("0" * (N - 1))
    condiciones =    "1" * N
    alcance =        "1" * N
    mecanismo =      "1" * N

    # Configurar la versión para que el Manager cargue la TPM correcta
    aplicacion.set_pagina_red_muestra(VERSION)

    gestor_redes = Manager(estado_inicial)
    
    print(f"Abriendo archivo: {gestor_redes.tpm_filename}")
    
    mpt = gestor_redes.cargar_red()

    ### Ejemplo de solución mediante módulo heurístico de k-particiones ###
    analizador_kq = KQNodes(mpt)

    sia_k = analizador_kq.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
        k=K
    )
    print(sia_k)
