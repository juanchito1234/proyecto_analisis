from src.models.base.application import aplicacion
from src.main import ( iniciar, probar_k_geometric )


def main():
    """Inicializar el aplicativo."""

    aplicacion.profiler_habilitado = True
    # aplicacion.pagina_sample_network = "B"

    # pruebitas de las particiones k-geométricas jijijija
    probar_k_geometric()

    #iniciar()


if __name__ == "__main__":
    main()
