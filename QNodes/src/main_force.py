from src.controllers.manager import Manager
from src.strategies.force import BruteForceK
from src.models.base.application import aplicacion

N = 15
VERSION = "A"
K = 2

def iniciar():
    """Punto de entrada"""

    estado_inicial = "1" + ("0" * (N - 1))
    condiciones = "1" * N
    alcance = "1" * N
    mecanismo = "1" * N

    aplicacion.set_pagina_red_muestra(VERSION)
    gestor_redes = Manager(estado_inicial)
    print(f"Archivo TPM: {gestor_redes.tpm_filename}")
    mpt = gestor_redes.cargar_red()

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_bf = BruteForceK(mpt, k=K)

    resultado = analizador_bf.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print("\n===== RESULTADO =====")
    print("Perdida:", resultado.perdida)
    print("Particion:")
    print(resultado.particion)
    print("Tiempo:", resultado.tiempo_ejecucion)
