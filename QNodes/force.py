from src.models.base.application import aplicacion

from src.main_force import iniciar


def main():
    """Inicialización del aplicativo"""

    # 👇 Investiga en la clase `aplicación` para más configuraciones 👇 #
    aplicacion.activar_profiling()
    aplicacion.set_pagina_red_muestra("A")

    iniciar()


if __name__ == "__main__":
    main()
