import time
import itertools
import numpy as np

from src.models.base.sia import SIA

from src.controllers.manager import Manager

from src.middlewares.slogger import SafeLogger

from src.middlewares.profile import (
    profile,
    profiler_manager,
)

from src.models.core.solution import Solution

from src.models.partitions.partition_evaluator import (
    PartitionEvaluator,
)

from src.constants.base import (
    NET_LABEL,
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)

from src.constants.models import (
    BRUTEFORCE_STRAREGY_TAG,
    BRUTEFORCE_ANALYSIS_TAG,
)


def partition_set(elements, k):
    """
    Genera de forma exacta todas las particiones
    de una lista en k conjuntos no vacíos.

    Implementación recursiva basada en el número
    de Stirling de segunda especie S(n, k).

    Parameters
    ----------
    elements : list
        Lista de elementos a particionar.

    k : int
        Número de bloques (subconjuntos) deseados.

    Yields
    ------
    list[set]
        Cada partición válida como lista de sets.
    """
    if k == 1:
        yield [set(elements)]
        return
    if k == len(elements):
        yield [set([e]) for e in elements]
        return
    if k > len(elements) or k <= 0:
        return

    first = elements[0]
    rest = elements[1:]

    # Caso 1: El primer elemento forma su propio bloque
    for p in partition_set(rest, k - 1):
        yield [set([first])] + p

    # Caso 2: El primer elemento se une a un bloque existente
    for p in partition_set(rest, k):
        for i in range(len(p)):
            new_p = [s.copy() for s in p]
            new_p[i].add(first)
            yield new_p


class KBruteForce(SIA):
    """
    Estrategia de Fuerza Bruta global para encontrar
    la k-MIP exacta en el marco GeoMIP.

    Genera exhaustivamente todas las combinaciones
    posibles de k-particiones válidas mediante el
    emparejamiento de particiones del presente (P)
    y del futuro (F), evalúa sus pérdidas EMD, y
    encuentra la k-MIP óptima global.

    Limitaciones:
    - N ≤ 6 (nodos futuros)
    - k ≤ 4 (número de bloques)

    Sirve como baseline científico para validar
    las heurísticas de KGeometric y KQNodes.
    """

    def __init__(
        self,
        gestor: Manager
    ):
        """
        Inicializa la estrategia K-BruteForce.

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
            BRUTEFORCE_STRAREGY_TAG
        )

        self.vertices = set()

        self.partition_evaluator = None

    @profile(
        context={
            TYPE_TAG:
            BRUTEFORCE_ANALYSIS_TAG
        }
    )
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 3
    ):
        """
        Punto de entrada principal de la
        estrategia K-BruteForce.

        Flujo:
        -------

        1. Preparar subsistema SIA

        2. Generar exhaustivamente todas
           las k-particiones válidas

        3. Evaluar cada partición con EMD

        4. Retornar la k-MIP de menor pérdida

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

        n_presentes = len(
            self.sia_subsistema
            .dims_ncubos
        )

        if n_futuros > 6 or k > 4:
            raise ValueError(
                f"KBruteForce limitado a "
                f"N ≤ 6 y k ≤ 4. "
                f"Subsistema tiene "
                f"N={n_futuros}, k={k}."
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
        Búsqueda exhaustiva.
        """

        presentes_ids = list(
            self.sia_subsistema.dims_ncubos
        )
        futuros_ids = list(
            self.sia_subsistema.indices_ncubos
        )

        mejor_particion = None
        menor_perdida = float("inf")

        self.logger.critic(
            f"Iniciando búsqueda exhaustiva "
            f"k-MIP con k={k}, "
            f"N_futuro={n_futuros}, "
            f"N_presente={n_presentes}"
        )

        # Emparejar particiones de presente y
        # futuro en todas las formas posibles
        for p_part in partition_set(
            presentes_ids, k
        ):
            for f_part in partition_set(
                futuros_ids, k
            ):
                for f_perm in (
                    itertools.permutations(
                        f_part
                    )
                ):

                    # Construir partición abstracta
                    particion_abstracta = []
                    for j in range(k):
                        bloque = set()
                        for p_node in p_part[j]:
                            bloque.add(
                                (ACTUAL, p_node)
                            )
                        for f_node in f_perm[j]:
                            bloque.add(
                                (EFECTO, f_node)
                            )
                        particion_abstracta.append(
                            bloque
                        )

                    try:
                        perdida, _ = (
                            self.partition_evaluator
                            .evaluate_partition(
                                particion_abstracta,
                                self
                                .split_partition_by_time
                            )
                        )

                        if perdida < menor_perdida:
                            menor_perdida = perdida
                            mejor_particion = (
                                particion_abstracta
                            )

                    except Exception:
                        continue

        if mejor_particion is None:
            raise ValueError(
                "No fue posible encontrar "
                "una k-MIP por fuerza bruta."
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

        self.logger.critic(
            f"k-MIP exacta encontrada "
            f"con pérdida mínima="
            f"{menor_perdida}"
        )

        particion_formateada = (
            self.formatear_particion(
                mejor_particion
            )
        )

        return Solution(
            estrategia=(
                f"K-BRUTEFORCE (k={k})"
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
                self.sia_tiempo_inicio
            ),
            particion=particion_formateada
        )

    def split_partition_by_time(
        self,
        partition
    ):
        """
        Convierte una partición abstracta
        al formato requerido por k_particionar().

        Parameters
        ----------
        partition : list[set]

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
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
                    alcance.append(nodo)

                elif tipo == ACTUAL:
                    mecanismo.append(nodo)

                else:
                    raise ValueError(
                        f"Tipo inválido: "
                        f"{tipo}"
                    )

            if not alcance:
                raise ValueError(
                    f"Bloque {idx} sin "
                    f"variables EFECTO."
                )
            if not mecanismo:
                raise ValueError(
                    f"Bloque {idx} sin "
                    f"variables ACTUAL."
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

    def formatear_particion(
        self,
        particion
    ):
        """
        Convierte partición a formato canónico
        hashable para representación de la
        solución.

        Parameters
        ----------
        particion : list[set]

        Returns
        -------
        tuple[tuple]
        """

        resultado = []
        for bloque in particion:
            bloque_ordenado = tuple(
                sorted(list(bloque))
            )
            resultado.append(bloque_ordenado)
        return tuple(resultado)
