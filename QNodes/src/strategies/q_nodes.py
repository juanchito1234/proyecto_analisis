import time
from typing import Union
import numpy as np
from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.profile import gestor_perfilado, profile
from src.funcs.format import fmt_biparticion_q, fmt_k_particion_q
from src.models.base.sia import SIA

from src.models.core.solution import Solution
from src.constants.models import (
    QNODES_ANALYSIS_TAG,
    QNODES_LABEL,
    QNODES_STRAREGY_TAG,
)
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    LAST_IDX,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion


class QNodes(SIA):
    """
    Clase QNodes para el análisis de redes mediante el algoritmo Q.

    Esta clase implementa un gestor principal para el análisis de redes que utiliza
    el algoritmo Q para encontrar la partición óptima que minimiza la
    pérdida de información en el sistema. Hereda de la clase base SIA (Sistema de
    Información Activo) y proporciona funcionalidades para analizar la estructura
    y dinámica de la red.

    Args:
    ----
        config (Loader):
            Instancia de la clase Loader que contiene la configuración del sistema
            y los parámetros necesarios para el análisis.

    Attributes:
    ----------
        m (int):
            Número de elementos en el conjunto de purview (vista).

        n (int):
            Número de elementos en el conjunto de mecanismos.

        tiempos (tuple[np.ndarray, np.ndarray]):
            Tupla de dos arrays que representan los tiempos para los estados
            actual y efecto del sistema.

        etiquetas (list[tuple]):
            Lista de tuplas conteniendo las etiquetas para los nodos,
            con versiones en minúsculas y mayúsculas del abecedario.

        vertices (set[tuple]):
            Conjunto de vértices que representan los nodos de la red,
            donde cada vértice es una tupla (tiempo, índice).

        memoria (dict):
            Diccionario para almacenar resultados intermedios y finales
            del análisis (memoización).

        logger:
            Instancia del logger configurada para el análisis Q.

    Methods:
    -------
        run(condicion, purview, mechanism):
            Ejecuta el análisis principal de la red con las condiciones,
            purview y mecanismo especificados.

        algorithm(vertices):
            Implementa el algoritmo Q para encontrar la partición
            óptima del sistema.

        funcion_submodular(deltas, omegas):
            Calcula la función submodular para evaluar particiones candidatas.

        view_solution(mip):
            Visualiza la solución encontrada en términos de las particiones
            y sus valores asociados.

        nodes_complement(nodes):
            Obtiene el complemento de un conjunto de nodos respecto a todos
            los vértices del sistema.

    Notes:
    -----
    - La clase implementa una versión secuencial del algoritmo Q para encontrar la partición que minimiza la pérdida de información.
    - Utiliza memoización para evitar recálculos innecesarios durante el proceso.
    - El análisis se realiza considerando dos tiempos: actual (presente) y
      efecto (futuro).
    """

    def __init__(self, tpm: np.ndarray):
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self.m: int
        self.n: int
        self.tiempos: tuple[np.ndarray, np.ndarray]
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.vertices: set[tuple]
        self.clave_submodular = [], []
        self.memoria_delta = {}
        self.memoria_grupo_candidato = {}

        self.indices_alcance: np.ndarray
        self.indices_mecanismo: np.ndarray

        self.logger = SafeLogger(QNODES_STRAREGY_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ):
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # e.g. (1,0)=A (1,1)=B (1,2)=C #
        futuro = tuple(
            (EFFECT, idx_efecto) for idx_efecto in self.sia_subsistema.indices_ncubos
        )

        # e.g. (0,0)=a (0,2)=c (0,4)=e #
        presente = tuple(
            (ACTUAL, idx_actual) for idx_actual in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size

        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        vertices = list(presente + futuro)
        self.vertices = set(presente + futuro)
        
        k = max(2, min(5, k))
        grupos = [vertices]
        
        perdida_final = INFTY_POS
        dist_marginal_final = None
        
        while len(grupos) < k:
            mejor_emd_global = INFTY_POS
            mejor_dist_global = None
            mejor_idx_grupo = -1
            mejor_division = None
            
            for idx_g, grupo in enumerate(grupos):
                if len(grupo) <= 1:
                    continue
                
                self.memoria_grupo_candidato.clear()
                mip = self.algorithm(grupo)
                
                grupo_mip = list(mip)
                grupo_complemento = list(set(grupo) - set(mip))
                
                nueva_particion_nodos = grupos[:idx_g] + [grupo_mip, grupo_complemento] + grupos[idx_g+1:]
                
                particiones = []
                for parte_nodos in nueva_particion_nodos:
                    alc = np.array([idx for t, idx in parte_nodos if t == EFFECT], dtype=np.int8)
                    mec = np.array([idx for t, idx in parte_nodos if t == ACTUAL], dtype=np.int8)
                    particiones.append((alc, mec))
                    
                sistema_particionado = self.sia_subsistema.k_partir(particiones)
                dist_marginal = sistema_particionado.distribucion_marginal()
                emd_candidata = emd_efecto(dist_marginal, self.sia_dists_marginales)
                
                if emd_candidata < mejor_emd_global:
                    mejor_emd_global = emd_candidata
                    mejor_dist_global = dist_marginal
                    mejor_idx_grupo = idx_g
                    mejor_division = (grupo_mip, grupo_complemento)
                    
            if mejor_idx_grupo == -1:
                break
                
            grupos = grupos[:mejor_idx_grupo] + list(mejor_division) + grupos[mejor_idx_grupo+1:]
            perdida_final = mejor_emd_global
            dist_marginal_final = mejor_dist_global

        fmt_mip = fmt_k_particion_q(grupos)

        return Solution(
            estrategia=QNODES_LABEL,
            perdida=perdida_final,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_marginal_final,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    @profile(context={TYPE_TAG: QNODES_ANALYSIS_TAG})
    def algorithm(self, vertices: list[tuple[int, int]]):
        """
        Implementa el algoritmo Q para encontrar la partición óptima de un sistema que minimiza la pérdida de información, basándose en principios de submodularidad dentro de la teoría de lainformación.

        El algoritmo opera sobre un conjunto de vértices que representan nodos en diferentes tiempos del sistema (presente y futuro). La idea fundamental es construir incrementalmente grupos de nodos que, cuando se particionan, producen la menor pérdida posible de información en el sistema.

        Proceso Principal:
        -----------------
        El algoritmo comienza estableciendo dos conjuntos fundamentales: omega (W) y delta.
        Omega siempre inicia con el primer vértice del sistema, mientras que delta contiene todos los vértices restantes. Esta decisión no es arbitraria - al comenzar con un
        solo elemento en omega, podemos construir grupos de manera incremental evaluando cómo cada adición afecta la pérdida de información.

        La ejecución se desarrolla en fases, ciclos e iteraciones, donde cada fase representa un nivel diferente y conlleva a la formación de una partición candidata, cada ciclo representa un incremento de elementos al conjunto W y cada iteración determina al final cuál es el mejor elemento/cambio/delta para añadir en W.
        Fase >> Ciclo >> Iteración.

        1. Formación Incremental de Grupos:
        El algoritmo mantiene un conjunto omega que crece gradualmente en cada j-iteración. En cada paso, evalúa todos los deltas restantes para encontrar cuál, al unirse con omega produce la menor pérdida de información. Este proceso utiliza la función submodular para calcular la diferencia entre la EMD (Earth Mover's Distance) de la combinación y la EMD individual del delta evaluado.

        2. Evaluación de deltas:
        Para cada delta candidato el algoritmo:
        - Calcula su EMD individual si no está en memoria.
        - Calcula la EMD de su combinación con el conjunto omega actual
        - Determina la diferencia entre estas EMDs (el "costo" de la combinación)
        El delta que produce el menor costo se selecciona y se añade a omega.

        3. Formación de Nuevos Grupos:
        Al final de cada fase cuando omega crezca lo suficiente, el algoritmo:
        - Toma los últimos elementos de omega y delta (par candidato).
        - Los combina en un nuevo grupo
        - Actualiza la lista de vértices para la siguiente fase
        Este proceso de agrupamiento permite que el algoritmo construya particiones
        cada vez más complejas y reutilice estos "pares candidatos" para particiones en conjunto.

        Optimización y Memoria:
        ----------------------
        El algoritmo utiliza dos estructuras de memoria clave:
        - individual_memory: Almacena las EMDs y distribuciones de nodos individuales, evitando recálculos muy costosos.
        - partition_memory: Guarda las EMDs y distribuciones de las particiones completas, permitiendo comparar diferentes combinaciones de grupos teniendo en cuenta que su valor real está asociado al valor individual de su formación delta.

        La memoización es relevante puesto muchos cálculos de EMD son computacionalmente costosos y se repiten durante la ejecución del algoritmo.

        Resultado:
        ---------------
        Al terminar todas las fases, el algoritmo selecciona la partición que produjo la menor EMD global, representando la división del sistema que mejor preserva su información causal.

        Args:
            vertices (list[tuple[int, int]]): Lista de vértices donde cada uno es una
                tupla (tiempo, índice). tiempo=0 para presente (t_0), tiempo=1 para futuro (t_1).

        Returns:
            tuple[float, tuple[tuple[int, int], ...]]: El valor de pérdida en la primera posición, asociado con la partición óptima encontrada, identificada por la clave en partition_memory que produce la menor EMD.
        """
        indice_emd = INT_ZERO

        if len(vertices) == 2:
            # Condición Matemática: Cuando hay exactamente 2 vértices o subcomponentes, 
            # len(deltas_ciclo) es 1. El bucle j in range(0) nunca se ejecuta, por lo 
            # que nunca evaluamos deltas. Pero en una red de 2 nodos, solo existe 
            # 1 partición posible. La evaluamos explícitamente:
            emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                vertices[1], [vertices[0]]
            )
            clave_actual = (
                tuple(vertices[1])
                if isinstance(vertices[1], list)
                else (vertices[1],)
            )
            self.memoria_grupo_candidato[clave_actual] = (
                emd_delta,
                dist_marginal_delta,
            )
            return clave_actual

        for i in range(len(vertices) - 1):
            # self.logger.debug(f"total: {len(vertices) - i}")
            omegas_ciclo = [vertices[0]]
            deltas_ciclo = vertices[1:]

            emd_particion_candidata = INFTY_POS
            dist_particion_candidata = None

            for j in range(len(deltas_ciclo) - 1):
                # self.logger.critic(f"   {j=}")
                emd_local = 1e5
                indice_mip: int

                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )

                    emd_iteracion = emd_union - emd_delta

                    # Guardamos TODAS las configuraciones delta evaluadas. Debido a que 
                    # la función de EMD no es estrictamente submodular ni simétrica,
                    # la heurística puede descartar mínimos globales tempranos.
                    clave_actual = (
                        tuple(deltas_ciclo[k])
                        if isinstance(deltas_ciclo[k], list)
                        else (deltas_ciclo[k],)
                    )
                    self.memoria_grupo_candidato[clave_actual] = (
                        emd_delta,
                        dist_marginal_delta,
                    )

                    if emd_iteracion < emd_local:
                        if emd_delta == INT_ZERO:
                            return clave_actual

                        emd_local = emd_iteracion
                        indice_mip = k
                        emd_particion_candidata = emd_delta
                        dist_particion_candidata = dist_marginal_delta
                # self.logger.critic(f"       [k]: {indice_mip}")

                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)
            par_candidato = (
                [omegas_ciclo[LAST_IDX]]
                if isinstance(omegas_ciclo[LAST_IDX], tuple)
                else omegas_ciclo[LAST_IDX]
            ) + (
                deltas_ciclo[LAST_IDX]
                if isinstance(deltas_ciclo[LAST_IDX], list)
                else deltas_ciclo
            )

            omegas_ciclo.pop()
            omegas_ciclo.append(par_candidato)

            vertices = omegas_ciclo

        if len(vertices) == 2:
            # Condición Matemática: Cuando hay exactamente 2 vértices o subcomponentes, 
            # len(deltas_ciclo) es 1. El bucle j in range(0) nunca se ejecuta, por lo 
            # que nunca evaluamos deltas. Pero en una red de 2 nodos, solo existe 
            # 1 partición posible. La evaluamos explícitamente:
            emd_union, emd_delta, dist_marginal_delta = self.funcion_submodular(
                vertices[1], [vertices[0]]
            )
            clave_actual = (
                tuple(vertices[1])
                if isinstance(vertices[1], list)
                else (vertices[1],)
            )
            self.memoria_grupo_candidato[clave_actual] = (
                emd_delta,
                dist_marginal_delta,
            )

        return min(
            self.memoria_grupo_candidato,
            key=lambda k: self.memoria_grupo_candidato[k][indice_emd],
        )

    def funcion_submodular(
        self, deltas: Union[tuple, list[tuple]], omegas: list[Union[tuple, list[tuple]]]
    ):
        """
        Evalúa el impacto de combinar el conjunto de nodos individual delta y su agrupación con el conjunto omega, calculando la diferencia entre EMD (Earth Mover's Distance) de las configuraciones, en conclusión los nodos delta evaluados individualmente y su combinación con el conjunto omega.

        El proceso se realiza en dos fases principales:

        1. Evaluación Individual:
           - Crea una copia del estado temporal del subsistema.
           - Activa los nodos delta en su tiempo correspondiente (presente/futuro).
           - Si el delta ya fue evaluado antes, recupera su EMD y distribución marginal de memoria
           - Si no, ha de:
             * Identificar dimensiones activas en presente y futuro.
             * Realiza bipartición del subsistema con esas dimensiones.
             * Calcular la distribución marginal y EMD respecto al subsistema.
             * Guarda resultados en memoria para seguro un uso futuro.

        2. Evaluación Combinada:
           - Sobre la misma copia temporal, activa también los nodos omega.
           - Calcula dimensiones activas totales (delta + omega).
           - Realiza bipartición del subsistema completo.
           - Obtiene EMD de la combinación.

        Args:
            deltas: Un nodo individual (tupla) o grupo de nodos (lista de tuplas)
                   donde cada tupla está identificada por su (tiempo, índice), sea el tiempo t_0 identificado como 0, t_1 como 1 y, el índice hace referencia a las variables/dimensiones habilitadas para operaciones de substracción/marginalización sobre el subsistema, tal que genere la partición.
            omegas: Lista de nodos ya agrupados, puede contener tuplas individuales
                   o listas de tuplas para grupos formados por los pares candidatos o más uniones entre sí (grupos candidatos).

        Returns:
            tuple: (
                EMD de la combinación omega y delta,
                EMD del delta individual,
                Distribución marginal del delta individual
            )
            Esto lo hice así para hacer almacenamiento externo de la emd individual y su distribución marginal en las particiones candidatas.
        """
        vector_delta_marginal = None
        self.clave_submodular = [], []

        # Delta #

        clave_delta_actual, clave_delta_efecto = self.definir_clave(deltas)
        clave_delta = tuple(clave_delta_actual), tuple(clave_delta_efecto)

        idxs_alcance_delta = self.clave_submodular[EFFECT]
        dims_mecanismo_delta = self.clave_submodular[ACTUAL]

        if clave_delta not in self.memoria_delta:
            particion_delta = self.sia_subsistema.bipartir(
                np.array(idxs_alcance_delta, dtype=np.int8),
                np.array(dims_mecanismo_delta, dtype=np.int8),
            )
            vector_delta_marginal = particion_delta.distribucion_marginal()
            emd_delta = emd_efecto(vector_delta_marginal, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = emd_delta, vector_delta_marginal

        else:
            emd_delta, vector_delta_marginal = self.memoria_delta[clave_delta]

        # Unión #

        for omega in omegas:
            self.definir_clave(omega)

        idxs_alcance_union = self.clave_submodular[EFFECT]
        dims_mecanismo_union = self.clave_submodular[ACTUAL]

        particion_union = self.sia_subsistema.bipartir(
            np.array(idxs_alcance_union, dtype=np.int8),
            np.array(dims_mecanismo_union, dtype=np.int8),
        )
        vector_union_marginal = particion_union.distribucion_marginal()
        emd_union = emd_efecto(vector_union_marginal, self.sia_dists_marginales)

        return emd_union, emd_delta, vector_delta_marginal

    def definir_clave(
        self,
        conjunto: Union[tuple[int, int], list[tuple[int, int]]],
    ):
        if isinstance(conjunto, tuple):
            tiempo, indice = conjunto
            self.clave_submodular[tiempo].append(indice)
        else:
            for tiempo, indice in conjunto:
                self.clave_submodular[tiempo].append(indice)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()
        return self.clave_submodular

    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
