import warnings
warnings.filterwarnings("ignore")

from src.models.base.application import aplicacion
from src.main import ( iniciar, iniciar_k_geometric, probar_geometric, probar_k_geometric )


def main():
    """Inicializar el aplicativo."""

    aplicacion.profiler_habilitado = True
    # aplicacion.pagina_sample_network = "B"

    # probar_geometric()
    probar_k_geometric()


if __name__ == "__main__":
    main()
