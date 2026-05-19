import time
import numpy as np

from src.models.base.sia import SIA
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import profile, gestor_perfilado
from src.models.core.solution import Solution

from src.models.geometry.transition_geometry import TransitionGeometry
from src.models.partitions.partition_evaluator import PartitionEvaluator
from src.models.partitions.k_partition_generator import KPartitionGenerator

from src.constants.base import (
    NET_LABEL,
    ACTUAL,
    EFFECT,
    TYPE_TAG,
)

from src.constants.models import (
    QNODES_STRAREGY_TAG,
    QNODES_ANALYSIS_TAG,
)

from src.models.base.application import aplicacion


class KQNodes(SIA):
    """
    Estrategia de K-particiones para QNodes,
    adaptando la heurística geométrica de GeoMIP.
    """

    def __init__(self, tpm: np.ndarray):
        """
        Inicializa estrategia K-QNodes.
        """
        super().__init__(tpm)

        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[0])}{aplicacion.pagina_red_muestra}"
        )

        self.logger = SafeLogger(QNODES_STRAREGY_TAG)

        self.geometry = None
        self.partition_generator = None
        self.partition_evaluator = None

        self.vertices = set()
        self.estado_inicial = None
        self.estado_final = None

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 3
    ):
        """
        Punto de entrada principal.
        """
        if k < 2:
            raise ValueError("k debe ser al menos 2.")

        # 1. Preparar subsistema
        self.sia_preparar_subsistema(
            estado_inicial,
            condicion,
            alcance,
            mecanismo
        )

        n_futuros = len(self.sia_subsistema.indices_ncubos)

        if k > n_futuros:
            raise ValueError(f"k={k} excede la cantidad de nodos futuros ({n_futuros}).")

        # 2. Construcción de vértices
        futuro = tuple(
            (EFFECT, efecto)
            for efecto in self.sia_subsistema.indices_ncubos
        )

        presente = tuple(
            (ACTUAL, actual)
            for actual in self.sia_subsistema.dims_ncubos
        )

        self.vertices = set(presente + futuro)

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial_bin = self.sia_subsistema.estado_inicial[dims]
        self.estado_final_bin = 1 - self.estado_inicial_bin

        # 3. Inicializar geometría
        self.geometry = TransitionGeometry(
            sia_subsistema=self.sia_subsistema,
            estado_inicial=self.estado_inicial_bin,
            estado_final=self.estado_final_bin
        )
        self.geometry.build_geometry()

        # 4. Inicializar generador
        self.partition_generator = KPartitionGenerator(
            sia_subsistema=self.sia_subsistema,
            tabla_transiciones=self.geometry.get_transition_table(),
            caminos=self.geometry.get_paths(),
            estado_final=self.estado_final_bin,
            logger=self.logger
        )

        # 5. Inicializar evaluador
        self.partition_evaluator = PartitionEvaluator(
            sia_subsistema=self.sia_subsistema,
            distribucion_original=self.sia_dists_marginales
        )

        # 6. Encontrar k-MIP
        mejor_particion = self.find_k_mip(k)

        # 7. Evaluación final
        perdida, dist_particion = self.partition_evaluator.evaluate_partition(
            mejor_particion,
            self.split_partition_by_time
        )

        # Formatear la partición resultante para visualización
        particion_formateada = self.formatear_particion(mejor_particion)

        return Solution(
            estrategia=f"K-QNODES (k={k})",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion_formateada
        )

    def find_k_mip(self, k: int):
        self.logger.critic(f"Iniciando búsqueda de k-MIP con k={k}")

        candidatos = self.partition_generator.identificar_particiones_candidatas(k)

        if not candidatos:
            raise ValueError("No se generaron particiones candidatas.")

        mejor_particion = None
        menor_perdida = float("inf")

        for idx, particion in enumerate(candidatos):
            try:
                perdida, _ = self.partition_evaluator.evaluate_partition(
                    particion,
                    self.split_partition_by_time
                )

                if perdida < menor_perdida:
                    menor_perdida = perdida
                    mejor_particion = particion

            except Exception as e:
                self.logger.warning(f"Error evaluando partición {idx}: {str(e)}")
                continue

        if mejor_particion is None:
            raise ValueError("No fue posible encontrar una k-MIP válida.")

        self.logger.critic(f"k-MIP encontrada con pérdida mínima={menor_perdida}")
        return mejor_particion

    def split_partition_by_time(self, partition):
        if not partition:
            raise ValueError("La partición está vacía.")

        resultado = []
        for idx, bloque in enumerate(partition):
            if not bloque:
                raise ValueError(f"Bloque {idx} vacío.")

            alcance = []
            mecanismo = []

            for tipo, nodo in bloque:
                if tipo == EFFECT:
                    alcance.append(nodo)
                elif tipo == ACTUAL:
                    mecanismo.append(nodo)
                else:
                    raise ValueError(f"Tipo inválido: {tipo}")

            if not alcance:
                raise ValueError(f"Bloque {idx} sin variables EFFECT.")
            if not mecanismo:
                raise ValueError(f"Bloque {idx} sin variables ACTUAL.")

            resultado.append(
                (
                    np.array(sorted(alcance), dtype=np.int8),
                    np.array(sorted(mecanismo), dtype=np.int8)
                )
            )
        return resultado

    def formatear_particion(self, particion):
        """
        Da un formato visual a la k-partición similar al que usa QNodes tradicionalmente.
        """
        resultado = []
        for bloque in particion:
            bloque_ordenado = tuple(sorted(list(bloque)))
            resultado.append(bloque_ordenado)
        return tuple(resultado)

    def nodes_complement(self, nodes):
        if not self.vertices:
            raise ValueError("self.vertices no ha sido inicializado.")
        return list(set(self.vertices) - set(nodes))
