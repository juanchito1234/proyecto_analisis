import time
import numpy as np

from src.models.base.sia import SIA

from src.controllers.manager import Manager

from src.middlewares.slogger import SafeLogger

from src.middlewares.profile import (
    profile,
    profiler_manager,
)

from src.models.core.solution import Solution

from src.models.geometry.transition_geometry import (
    TransitionGeometry,
)

from src.models.partitions.partition_evaluator import (
    PartitionEvaluator,
)

from src.models.partitions.k_partition_generator import (
    KPartitionGenerator,
)

from src.constants.base import (
    NET_LABEL,
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)

from src.constants.models import (
    GEOMETRIC_STRAREGY_TAG,
    GEOMETRIC_ANALYSIS_TAG,
)


class KGeometric(SIA):
    """
    Estrategia geométrica extendida para
    búsqueda de k-MIP.

    Esta implementación desacopla la lógica
    principal en tres servicios especializados:

    1. TransitionGeometry
        - geometría de transición
        - costos geométricos
        - caminos Hamming

    2. KPartitionGenerator
        - generación heurística de candidatos
        - clusterización geométrica
        - semillas estructurales

    3. PartitionEvaluator
        - evaluación EMD
        - memoización multinivel
        - cálculo de marginales

    Arquitectura:
    --------------

    KGeometric
        coordina

            ↓

        TransitionGeometry
        KPartitionGenerator
        PartitionEvaluator

    Esto permite:

    - reutilización
    - profiling independiente
    - testing modular
    - futuras estrategias híbridas
    """

    def __init__(
        self,
        gestor: Manager
    ):
        """
        Inicializa estrategia K-Geometric.

        Parameters
        ----------
        gestor : Manager
            Gestor principal del sistema.
        """

        super().__init__(gestor)

        profiler_manager.start_session(
            f"{NET_LABEL}"
            f"{len(gestor.estado_inicial)}"
            f"{gestor.pagina}"
        )

        self.logger = SafeLogger(
            GEOMETRIC_STRAREGY_TAG
        )

        """
        Servicios desacoplados.
        """

        self.geometry = None

        self.partition_generator = None

        self.partition_evaluator = None

        """
        Información global del sistema.
        """

        self.vertices = set()

        self.estado_inicial = None

        self.estado_final = None

    @profile(
        context={
            TYPE_TAG:
            GEOMETRIC_ANALYSIS_TAG
        }
    )
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 3,
        MAX_CANDIDATOS = 300
    ):
        """
        Punto de entrada principal de la
        estrategia K-Geometric.

        Flujo:
        -------

        1. Preparar subsistema

        2. Construir geometría

        3. Generar candidatos

        4. Evaluar candidatos

        5. Retornar k-MIP óptima

        Parameters
        ----------
        condicion : str

        alcance : str

        mecanismo : str

        tpm : np.ndarray

        k : int

        Returns
        -------
        Solution
        """

        if k < 2:

            raise ValueError(
                "k debe ser al menos 2."
            )

        inicio_estrategia = time.time()

        """
        Preparar subsistema SIA.
        """

        self.sia_preparar_subsistema(
            condicion,
            alcance,
            mecanismo,
            tpm
        )

        n_futuros = len(
            self.sia_subsistema
            .indices_ncubos
        )

        if k > n_futuros:

            raise ValueError(
                f"k={k} excede la cantidad "
                f"de nodos futuros "
                f"({n_futuros})."
            )

        """
        Construcción de vértices.
        """

        futuro = tuple(
            (
                EFECTO,
                efecto
            )
            for efecto in (
                self.sia_subsistema
                .indices_ncubos
            )
        )

        presente = tuple(
            (
                ACTUAL,
                actual
            )
            for actual in (
                self.sia_subsistema
                .dims_ncubos
            )
        )

        self.vertices = set(
            presente + futuro
        )

        dims = (
            self.sia_subsistema
            .dims_ncubos
        )

        self.estado_inicial = (
            self.sia_subsistema
            .estado_inicial[dims]
        )

        """
        Objetivo geométrico:
        complemento binario.
        """

        self.estado_final = (
            1 - self.estado_inicial
        )

        """
        Inicializar geometría.
        """

        self.geometry = (
            TransitionGeometry(
                sia_subsistema=(
                    self.sia_subsistema
                ),
                estado_inicial=(
                    self.estado_inicial
                ),
                estado_final=(
                    self.estado_final
                )
            )
        )

        inicio = time.time()

        self.geometry.build_geometry()


        """
        Inicializar generador
        de particiones.
        """

        self.partition_generator = (
            KPartitionGenerator(
                sia_subsistema=(
                    self.sia_subsistema
                ),
                tabla_transiciones=(
                    self.geometry
                ),
                caminos=(
                    self.geometry
                    .get_paths()
                ),
                estado_final=(
                    self.estado_final
                ),
                logger=self.logger
            )
        )

        """
        Inicializar evaluador.
        """

        self.partition_evaluator = (
            PartitionEvaluator(
                sia_subsistema=(
                    self.sia_subsistema
                ),
                distribucion_original=(
                    self.sia_dists_marginales
                )
            )
        )

        """
        Encontrar k-MIP.
        """

        mejor_particion = (
            self.find_k_mip(k)
        )

        """
        Obtener evaluación final.
        """

        perdida, dist_particion = (
            self.partition_evaluator
            .evaluate_partition(
                mejor_particion,
                self.split_partition_by_time
            )
        )

        return Solution(
            estrategia=(
                f"K-GEOMETRIC (k={k})"
            ),
            perdida=perdida,
            distribucion_subsistema=(
                self.sia_dists_marginales
            ),
            distribucion_particion=(
                dist_particion
            ),
            tiempo_total=(
                time.time()
                -
                inicio_estrategia
            ),
            particion=mejor_particion,
            raw_particion=mejor_particion,
            dims_ncubos=dims
        )

    def find_k_mip(
        self,
        k: int
    ):
        """
        Busca la k-MIP de menor pérdida.

        Flujo:
        -------

        1. Generar candidatos

        2. Evaluar candidatos

        3. Seleccionar mínimo EMD

        Parameters
        ----------
        k : int

        Returns
        -------
        list
            Mejor partición encontrada.
        """

        self.logger.critic(
            f"Iniciando búsqueda "
            f"de k-MIP con k={k}"
        )

        inicio = time.time()

        candidatos = (
            self.partition_generator
            .identificar_particiones_candidatas(
                k
            )
        )



        if not candidatos:

            raise ValueError(
                "No se generaron "
                "particiones candidatas."
            )

        mejor_particion = None

        menor_perdida = float("inf")

        inicio = time.time()
        for idx, particion in enumerate(
            candidatos
        ):

            try:
                perdida, _ = (
                    self.partition_evaluator
                    .evaluate_partition(
                        particion,
                        self.split_partition_by_time
                    )
                )

                if perdida < menor_perdida:

                    menor_perdida = perdida

                    if perdida == 0:
                        self.logger.critic(
                            "Solución perfecta encontrada."
                        )

                        return particion

                    mejor_particion = (
                        particion
                    )
            except Exception as e:
                print("\n===== ERROR EN PARTICIÓN =====")
                print(particion)
                print(str(e))
                print("==============================\n")

                continue



        if mejor_particion is None:

            raise ValueError(
                "No fue posible encontrar "
                "una k-MIP válida."
            )



        return mejor_particion

    def split_partition_by_time(
        self,
        partition
    ):
        """
        Convierte una partición abstracta
        al formato requerido por:

            k_particionar()

        Parameters
        ----------
        partition : list[set]

        Returns
        -------
        list
            Lista de bloques:

                [
                    (
                        alcance,
                        mecanismo
                    ),
                    ...
                ]
        """

        if not partition:

            raise ValueError(
                "La partición está vacía."
            )

        resultado = []

        for idx, bloque in enumerate(
            partition
        ):

            if not bloque:

                raise ValueError(
                    f"Bloque {idx} vacío."
                )

            alcance = []

            mecanismo = []

            for tipo, nodo in bloque:

                if tipo == EFECTO:

                    alcance.append(
                        nodo
                    )

                elif tipo == ACTUAL:

                    mecanismo.append(
                        nodo
                    )

                else:

                    raise ValueError(
                        f"Tipo inválido: "
                        f"{tipo}"
                    )

            if not alcance and not mecanismo:
                raise ValueError(
                    f"Bloque {idx} vacío."
                )

            resultado.append(
                (
                    np.array(
                        sorted(alcance),
                        dtype=np.int8
                    ),
                    np.array(
                        sorted(mecanismo),
                        dtype=np.int8
                    )
                )
            )

        return resultado

    def nodes_complement(
        self,
        nodes
    ):
        """
        Retorna complemento de nodos
        respecto al conjunto total.

        Parameters
        ----------
        nodes : iterable

        Returns
        -------
        list
        """

        if not self.vertices:

            raise ValueError(
                "self.vertices no "
                "ha sido inicializado."
            )

        return list(
            set(self.vertices)
            -
            set(nodes)
        )
    