from src.controllers.manager import Manager
from src.strategies.q_nodes import QNodes
from src.models.base.application import aplicacion

K = 2
N = 8
VERSION = "A"

def iniciar():

    estado_inicial = "1" + ("0" * (N - 1))
    condiciones = "1" * N
    alcance = "1" * N
    mecanismo = "1" * N

    aplicacion.set_pagina_red_muestra(VERSION)

    gestor_redes = Manager(estado_inicial)

    print(f"Archivo TPM: {gestor_redes.tpm_filename}")
    print(f"N = {N}")
    print(f"k = {K}")

    mpt = gestor_redes.cargar_red()

    analizador = QNodes(mpt)

    resultado = analizador.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
        k=K
    )

    print("\n===== RESULTADO =====")
    print("Perdida:", resultado.perdida)
    print("Particion:")
    print(resultado.particion)
    print("Tiempo:", resultado.tiempo_ejecucion)

if __name__ == "__main__":
    iniciar()