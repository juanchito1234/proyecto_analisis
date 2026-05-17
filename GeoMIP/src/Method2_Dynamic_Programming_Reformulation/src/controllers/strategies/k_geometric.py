import time
import numpy as np

from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.middlewares.slogger import SafeLogger

from src.constants.base import NET_LABEL

from src.constants.models import (
    GEOMETRIC_STRAREGY_TAG,
)

from src.middlewares.profile import (
    profiler_manager,
)

from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)

from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
)

from src.middlewares.profile import (
    profile,
)

from src.models.core.solution import Solution

from src.funcs.base import emd_efecto

class KGeometric(SIA):
    """
    Estrategia GeoMIP extendida para k-particiones.

    Esta clase generaliza la estrategia geométrica actual
    (GeometricSIA) para permitir encontrar la k-MIP
    (Minimum Information Partition de orden k).

    En lugar de evaluar únicamente biparticiones,
    esta estrategia permite evaluar:

        k = 2, 3, 4, 5, ...

    reutilizando:

    - N-Cubos
    - tabla de costos de transición
    - EMD-Effect
    - distribuciones marginales
    - infraestructura base de SIA

    manteniendo compatibilidad total con el caso:

        k = 2
    """

    def __init__(
        self,
        gestor: Manager
    ):
        """
        Inicializa la estrategia K-Geometric.

        Esta clase extiende GeoMIP para permitir
        encontrar k-particiones (k-MIP) sobre
        sistemas matriciales probabilísticos.

        Aquí se preparan las estructuras base
        necesarias para:

        - construir la tabla geométrica de costos
        - almacenar particiones evaluadas
        - reutilizar N-Cubos existentes
        - ejecutar búsquedas por fuerza bruta
        - aplicar heurísticas geométricas

        Se mantiene compatibilidad completa con:

            k = 2

        equivalente al GeoMIP original.

        Parameters
        ----------
        gestor : Manager
            Gestor principal del sistema, encargado
            de suministrar:

            - estado inicial
            - TPM
            - página de análisis
            - configuración general

        Returns
        -------
        None
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
        Tabla principal de costos geométricos.

        Estructura:

            (
                estado_inicial,
                estado_final
            )
            ->
            [
                costo_futuro_0,
                costo_futuro_1,
                ...
            ]
        """
        self.tabla_transiciones = {}

        """
        Conjunto total de vértices del sistema.

        Incluye:

        - presente (ACTUAL)
        - futuro (EFECTO)
        """
        self.vertices = set()

        """
        Estructura auxiliar reutilizada desde
        GeoMIP original.
        """
        self.tabla = {}

        """
        Cache de particiones evaluadas.

        Estructura:

            particion_hash
            ->
            (
                perdida,
                distribucion
            )
        """
        self.memoria_particiones = {}

        """
        Estados agrupados por nivel de distancia
        de Hamming.

        Ejemplo:

            caminos[0] -> estado inicial
            caminos[1] -> vecinos a distancia 1
            caminos[2] -> vecinos a distancia 2
        """
        self.caminos = {}

        """
        Flattened ncubes para acceso rápido
        durante calcular_costo().
        """
        self._flat_data = []

        """
        Índices auxiliares de variables futuras.
        """
        self.idx_ncubos = []

        """
        Estados principales usados por GeoMIP.
        """
        self.estado_inicial = None
        self.estado_final = None

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 3
    ):
        """
        Punto de entrada principal de la estrategia K-Geometric.

        Esta función prepara el subsistema de análisis y ejecuta
        la búsqueda de la k-MIP (Minimum Information Partition
        de orden k).

        Flujo general:

        1. Preparar el subsistema a partir de:
            - condición de fondo
            - alcance
            - mecanismo
            - TPM

        2. Construir el conjunto de vértices:
            - presente (ACTUAL)
            - futuro (EFECTO)

        3. Inicializar:
            - estado inicial
            - estado final objetivo
            - estructuras auxiliares de transición

        4. Construir la tabla geométrica de costos
        reutilizando la estrategia GeoMIP

        5. Buscar la mejor k-partición

        6. Retornar la solución final compatible
        con la arquitectura existente

        Parameters
        ----------
        condicion : str
            Variables condicionadas (background conditions)

        alcance : str
            Variables futuras seleccionadas

        mecanismo : str
            Variables presentes seleccionadas

        tpm : np.ndarray
            Matriz de Probabilidad de Transición

        k : int, default=3
            Número de particiones a evaluar

        Returns
        -------
        Solution
            Objeto final con:

            - estrategia
            - pérdida mínima (phi)
            - distribución del subsistema
            - distribución de la mejor partición
            - tiempo total
            - representación de la k-MIP

        Raises
        ------
        ValueError
            Si k < 2 o si k excede el número de nodos
            disponibles en el subsistema.
        """

        if k < 2:
            raise ValueError(
                "k debe ser al menos 2. "
                "No existe MIP para k < 2."
            )
        
        self.sia_preparar_subsistema(
            condicion,
            alcance,
            mecanismo,
            tpm
        )

        n_futuros = len(self.sia_subsistema.indices_ncubos)

        if k > n_futuros:
            raise ValueError(
                f"k={k} excede la cantidad de nodos futuros "
                f"disponibles ({n_futuros})."
            )

        futuro = tuple(
            (EFECTO, efecto)
            for efecto in self.sia_subsistema.indices_ncubos
        )

        presente = tuple(
            (ACTUAL, actual)
            for actual in self.sia_subsistema.dims_ncubos
        )

        self.vertices = set(presente + futuro)

        dims = self.sia_subsistema.dims_ncubos

        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]

        """
        Igual que GeoMIP:
        el objetivo geométrico es avanzar
        hacia el complemento binario.
        """
        self.estado_final = 1 - self.estado_inicial

        self.idx_ncubos = list(
            range(
                len(self.sia_subsistema.indices_ncubos)
            )
        )

        self.caminos = {
            0: [self.estado_inicial.tolist()]
        }

        self.tabla_transiciones = {}
        self.memoria_particiones = {}

        self.tabla_transiciones[
            (
                tuple(self.caminos[0][0]),
                tuple(self.caminos[0][0])
            )
        ] = [
            0.0
            for _ in range(
                len(self.sia_subsistema.indices_ncubos)
            )
        ]

        self._flat_data = []

        for ncubo in self.sia_subsistema.ncubos:
            self._flat_data.append(
                ncubo.data.ravel()
            )

        for nivel in range(
            1,
            len(self.estado_inicial) + 1
        ):
            self.calcular_costos_nivel(
                self.estado_final,
                nivel
            )

        mejor_particion = self.find_k_mip(k)

        particion_formateada = self.split_partition_by_time(
            mejor_particion
        )

        key = tuple(
            (
                tuple(alcance.tolist()),
                tuple(mecanismo.tolist())
            )
            for alcance, mecanismo in particion_formateada
        )

        perdida, dist_particion = self.memoria_particiones[key]

        return Solution(
            estrategia=f"K-GEOMETRIC (k={k})",
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist_particion,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=str(mejor_particion),
        )

    def find_k_mip(
        self,
        k: int
    ):
        """
        Método principal para encontrar la k-MIP
        (Minimum Information Partition de orden k).

        Este método:

        1. Genera las k-particiones candidatas
        (inicialmente por fuerza bruta controlada)

        2. Evalúa cada partición usando:

            - k_particionar()
            - distribucion_marginal()
            - EMD-Effect

        3. Guarda los resultados en memoria:

            self.memoria_particiones

        4. Selecciona la partición con mínima pérdida:

            argmin(phi)

        Esta implementación sirve como baseline
        experimental y como validación contra:

            k = 2  → GeoMIP original

        Posteriormente este método será optimizado
        usando:

            identificar_particiones_candidatas()

        para reducir el espacio de búsqueda.

        Parameters
        ----------
        k : int
            Número de bloques de la partición.

        Returns
        -------
        list
            Mejor k-partición encontrada.

        Raises
        ------
        ValueError
            Si no se generan particiones válidas.
        """

        self.logger.critic(
            f"Iniciando búsqueda de k-MIP con k={k}"
        )

        candidatos = self.identificar_particiones_candidatas(k)

        if not candidatos:
            raise ValueError(
                f"No se generaron particiones válidas para k={k}"
            )

        mejor_particion = None
        menor_perdida = float("inf")

        for idx, particion in enumerate(candidatos):

            try:
                perdida, distribucion = self.evaluate_partition(
                    particion
                )

                particion_formateada = self.split_partition_by_time(
    particion
)

                key = tuple(
                    (
                        tuple(alcance.tolist()),
                        tuple(mecanismo.tolist())
                    )
                    for alcance, mecanismo in particion_formateada
                )

                self.memoria_particiones[key] = (
                    perdida,
                    distribucion
                )

                if perdida < menor_perdida:
                    menor_perdida = perdida
                    mejor_particion = particion

            except Exception as e:
                self.logger.warning(
                    f"Error evaluando partición {idx}: {str(e)}"
                )
                continue

        if mejor_particion is None:
            raise ValueError(
                "No fue posible encontrar una k-MIP válida."
            )

        self.logger.critic(
            f"k-MIP encontrada con pérdida mínima = {menor_perdida}"
        )

        return mejor_particion

    def generate_k_partitions(
        self,
        k: int
    ):
        """
        Genera todas las k-particiones candidatas
        sobre el conjunto de vértices del sistema.

        Cada vértice representa:

            (tipo, nodo)

        donde:

        - tipo = ACTUAL  → presente
        - tipo = EFECTO  → futuro

        La generación se realiza inicialmente
        mediante fuerza bruta controlada usando:

            k_partitions_recursive()

        Posteriormente esta función será optimizada
        usando la tabla geométrica de costos para
        reducir drásticamente el espacio de búsqueda.

        Importante:
        -----------

        La salida de esta función aún NO está en el
        formato final de:

            System.k_particionar()

        sino como particiones abstractas de vértices.

        Luego:

            split_partition_by_time()

        será el encargado de transformar cada bloque en:

            (alcance, mecanismo)

        Parameters
        ----------
        k : int
            Número de bloques deseados.

        Returns
        -------
        list
            Lista de k-particiones candidatas.

        Raises
        ------
        ValueError
            Si k excede la cantidad de nodos
            disponibles o si no hay vértices.
        """

        if not self.vertices:
            raise ValueError(
                "No existen vértices inicializados "
                "para generar particiones."
            )

        elementos = list(self.vertices)

        if k > len(elementos):
            raise ValueError(
                f"k={k} excede la cantidad total "
                f"de vértices ({len(elementos)})."
            )

        self.logger.critic(
            f"Generando particiones candidatas para k={k}"
        )

        candidatos = list(
            self.k_partitions_recursive(
                elementos,
                k
            )
        )

        if not candidatos:
            raise ValueError(
                f"No fue posible generar "
                f"k-particiones para k={k}"
            )

        self.logger.critic(
            f"Se generaron {len(candidatos)} "
            f"particiones candidatas"
        )

        return candidatos

    def k_partitions_recursive(
        self,
        items,
        k
    ):
        """
        Generador recursivo de k-particiones exactas.

        Produce todas las particiones posibles de:

            items → k bloques no vacíos

        evitando duplicados simétricos.

        Ejemplo:

            items = [A, B, C]
            k = 2

        genera:

            [{A}, {B, C}]
            [{B}, {A, C}]
            [{C}, {A, B}]

        pero NO repite permutaciones equivalentes como:

            [{B, C}, {A}]

        porque representan la misma partición.

        Esta función sirve como baseline exacto
        para validar:

            k = 2

        contra GeoMIP original, y para evaluar
        experimentalmente k > 2.

        Parameters
        ----------
        items : list
            Lista de elementos a particionar.

            Normalmente:

                self.vertices

        k : int
            Número de bloques deseados.

        Yields
        ------
        list[set]
            Una partición válida representada como:

                [
                    {elementos bloque 1},
                    {elementos bloque 2},
                    ...
                ]

        Notes
        -----
        Esta implementación usa backtracking puro.

        Es costosa para n grande, por eso se usa:

        - como baseline
        - para validación
        - para k <= 5 o 6

        tal como exige la entrega.
        """

        if k == 1:
            yield [set(items)]
            return

        if len(items) == k:
            yield [{item} for item in items]
            return

        if not items or k <= 0:
            return

        primer = items[0]
        resto = items[1:]

        """
        Caso 1:
        primer elemento forma su propio bloque
        """

        for particion in self.k_partitions_recursive(
            resto,
            k - 1
        ):
            yield [set([primer])] + particion

        """
        Caso 2:
        primer elemento se inserta en uno de los
        bloques existentes
        """

        for particion in self.k_partitions_recursive(
            resto,
            k
        ):
            for i in range(len(particion)):

                nueva_particion = [
                    bloque.copy()
                    for bloque in particion
                ]

                nueva_particion[i].add(primer)

                yield nueva_particion

    def split_partition_by_time(
        self,
        partition
    ):
        """
        Convierte una partición abstracta de vértices
        en una estructura compatible con:

            System.k_particionar()

        La entrada tiene la forma:

            [
                {
                    (ACTUAL, 0),
                    (ACTUAL, 1),
                    (EFECTO, 0)
                },
                {
                    (ACTUAL, 2),
                    (EFECTO, 1)
                }
            ]

        donde cada bloque contiene vértices mezclados
        entre presente y futuro.

        Esta función transforma cada bloque en:

            [
                (
                    alcance=np.array([...]),
                    mecanismo=np.array([...])
                ),
                ...
            ]

        donde:

        - alcance   → variables futuras (EFECTO)
        - mecanismo → variables presentes (ACTUAL)

        Este formato es el requerido por:

            System.k_particionar()

        Reglas importantes:
        -------------------

        1. Cada bloque debe contener al menos
        un nodo futuro (alcance)

        2. Cada bloque debe contener al menos
        un nodo presente (mecanismo)

        3. No se permiten bloques vacíos

        Parameters
        ----------
        partition : list[set[tuple]]
            Partición abstracta generada por:

                generate_k_partitions()

        Returns
        -------
        list
            Lista de bloques en formato:

                [
                    (alcance, mecanismo),
                    ...
                ]

        Raises
        ------
        ValueError
            Si algún bloque no tiene estructura válida.
        """

        if not partition:
            raise ValueError(
                "La partición recibida está vacía."
            )

        resultado = []

        for idx, bloque in enumerate(partition):

            if not bloque:
                raise ValueError(
                    f"El bloque {idx} está vacío."
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
                        f"Tipo de nodo inválido en bloque {idx}: "
                        f"{tipo}"
                    )

            if not alcance:
                raise ValueError(
                    f"El bloque {idx} no contiene nodos "
                    f"de alcance (EFECTO)."
                )

            if not mecanismo:
                raise ValueError(
                    f"El bloque {idx} no contiene nodos "
                    f"de mecanismo (ACTUAL)."
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

    def evaluate_partition(
        self,
        partition
    ):
        """
        Evalúa una k-partición candidata y calcula
        su pérdida de información:

            phi(partition)

        Flujo de evaluación:

        1. Convertir la partición abstracta mediante:

            split_partition_by_time()

        2. Validar estructura con:

            validar_k_particion()

        3. Construir el sistema k-particionado usando:

            System.k_particionar()

        4. Obtener la distribución marginal de la
        partición resultante:

            distribucion_marginal()

        5. Calcular la distancia EMD-Effect entre:

            - distribución original del subsistema
            - distribución de la k-partición

        El valor obtenido representa la pérdida
        de información asociada a esa partición.

        Parameters
        ----------
        partition : list[set[tuple]]
            Partición abstracta generada por:

                generate_k_partitions()

        Returns
        -------
        tuple
            (
                perdida_emd,
                distribucion_particion
            )

        Raises
        ------
        ValueError
            Si la partición no es válida.
        """

        if not partition:
            raise ValueError(
                "No se puede evaluar una partición vacía."
            )

        particion_formateada = self.split_partition_by_time(
            partition
        )

        self.sia_subsistema.validar_k_particion(
            particion_formateada
        )

        sistema_particionado = self.sia_subsistema.k_particionar(
            particion_formateada
        )

        distribucion_particion = (
            sistema_particionado.distribucion_marginal()
        )

        perdida = emd_efecto(
            distribucion_particion,
            self.sia_dists_marginales
        )

        return (
            perdida,
            distribucion_particion
        )

    def identificar_particiones_candidatas(
        self,
        k: int
    ):
        """
        Extensión k-generalizada de la lógica de
        GeoMIP original para construir múltiples
        particiones candidatas y no una sola.

        En lugar de retornar únicamente una
        partición heurística, esta versión genera
        varias configuraciones prometedoras usando
        distintas ventanas de semillas geométricas.

        Idea:

        1. Obtener costos globales desde:
            estado_inicial -> estado_final

        2. Ordenar variables futuras por menor costo

        3. Tomar múltiples ventanas de tamaño k:

            [0:k]
            [1:k+1]
            [2:k+2]
            ...

        4. Para cada ventana:
            - crear bloques semilla
            - distribuir presentes
            - distribuir futuros restantes

        5. Retornar varias particiones candidatas

        Esto convierte la búsqueda en una verdadera
        selección entre candidatos y no en una única
        solución forzada.

        Parameters
        ----------
        k : int
            Número de bloques deseados.

        Returns
        -------
        list
            Lista de particiones candidatas.
        """

        key = (
            tuple(self.caminos[0][0]),
            tuple(self.estado_final)
        )

        if key not in self.tabla_transiciones:
            raise ValueError(
                "No existe información en la tabla "
                "de transiciones para construir "
                "particiones candidatas."
            )

        costos = self.tabla_transiciones[key]

        if not costos:
            raise ValueError(
                "La tabla de costos está vacía."
            )

        pares = [
            (valor, idx)
            for idx, valor in enumerate(costos)
            if valor is not None
        ]

        if len(pares) < k:
            raise ValueError(
                f"No existen suficientes variables "
                f"para construir k={k} bloques."
            )

        pares.sort(
            key=lambda x: x[0]
        )

        presentes_totales = list(
            self.sia_subsistema.dims_ncubos
        )

        total_futuros = len(
            self.sia_subsistema.indices_ncubos
        )

        candidatos = []

        """
        Número máximo de ventanas a explorar.

        Ejemplo:

        si hay 8 futuros y k=3:

            ventanas posibles:
            [0:3]
            [1:4]
            [2:5]
            [3:6]
            [4:7]
            [5:8]
        """

        max_offset = len(pares) - k + 1

        for offset in range(max_offset):

            """
            Seleccionar semillas geométricas
            usando ventana deslizante.
            """

            semillas = [
                idx
                for _, idx in pares[
                    offset:offset + k
                ]
            ]

            particion = []

            """
            Inicializar bloques:
            un futuro semilla por bloque
            """

            for futuro_seed in semillas:

                bloque = set()

                bloque.add(
                    (EFECTO, futuro_seed)
                )

                particion.append(
                    bloque
                )

            """
            Distribuir presentes usando
            round-robin.
            """

            for idx, presente in enumerate(
                presentes_totales
            ):
                bloque_destino = idx % k

                particion[
                    bloque_destino
                ].add(
                    (ACTUAL, presente)
                )

            """
            Distribuir futuros restantes.
            """

            futuros_restantes = [
                idx
                for idx in range(total_futuros)
                if idx not in semillas
            ]

            for idx, futuro in enumerate(
                futuros_restantes
            ):
                bloque_destino = idx % k

                particion[
                    bloque_destino
                ].add(
                    (EFECTO, futuro)
                )

            """
            Evitar duplicados estructurales.
            """

            firma = tuple(
                sorted(
                    tuple(
                        sorted(bloque)
                    )
                    for bloque in particion
                )
            )

            if firma not in [
                tuple(
                    sorted(
                        tuple(
                            sorted(b)
                        )
                        for b in p
                    )
                )
                for p in candidatos
            ]:
                candidatos.append(
                    particion
                )

        if not candidatos:
            raise ValueError(
                "No fue posible generar "
                "particiones candidatas válidas."
            )

        self.logger.critic(
            f"Se generaron {len(candidatos)} "
            f"particiones candidatas para k={k}"
        )

        return candidatos

    def calcular_costos_nivel(
        self,
        estado_final: np.ndarray,
        nivel: int
    ):
        """
        Construye los estados alcanzables para un nivel
        determinado de distancia de Hamming respecto al
        estado inicial y calcula sus costos de transición.

        Esta función reutiliza directamente la lógica de
        GeoMIP original.

        La idea es:

        nivel 1:
            estados que difieren en 1 bit

        nivel 2:
            estados que difieren en 2 bits

        ...

        hasta llegar al estado final.

        Para cada nuevo estado generado:

            estado_inicial → nuevo_estado

        se calcula:

            tx(i, j)

        usando:

            calcular_costo()

        y se almacena en:

            self.tabla_transiciones

        Además se preserva la estructura:

            self.caminos[nivel]

        que luego será utilizada para construir
        candidatos geométricos.

        Parameters
        ----------
        estado_final : np.ndarray
            Estado objetivo binario.

        nivel : int
            Nivel actual de exploración por
            distancia de Hamming.

        Returns
        -------
        None
        """

        n = len(estado_final)

        visitados = set()

        self.caminos[nivel] = []

        for estado_anterior in self.caminos[nivel - 1]:

            estado_actual = np.array(
                estado_anterior
            )

            for i in range(n):

                if estado_actual[i] != estado_final[i]:

                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]

                    nuevo_estado_tuple = tuple(
                        nuevo_estado
                    )

                    if nuevo_estado_tuple not in visitados:

                        self.caminos[nivel].append(
                            nuevo_estado.tolist()
                        )

                        self.calcular_costo(
                            self.caminos[0][0],
                            nuevo_estado.tolist(),
                            self.idx_ncubos
                        )

                        visitados.add(
                            nuevo_estado_tuple
                        )

    def calcular_costo(
        self,
        estado_inicial,
        estado_final,
        ncubos
    ):
        """
        Calcula el costo geométrico de transición:

            tx(i, j)

        entre:

            estado_inicial -> estado_final

        para las variables futuras definidas en `ncubos`.

        Esta función reutiliza directamente la lógica
        de GeoMIP original y construye la tabla de costos:

            self.tabla_transiciones

        usando la fórmula:

            tx(i,j) = γ * (
                |X[i] - X[j]|
                + sum(tx(k,j))
            )

        donde:

            γ = 1 / 2^(dh(i,j))

        y:

            dh(i,j)

        representa la distancia de Hamming entre ambos
        estados.

        La suma:

            sum(tx(k,j))

        corresponde a los costos de transición desde
        los vecinos intermedios que forman caminos
        óptptimos hacia el estado objetivo.

        Parameters
        ----------
        estado_inicial : list | tuple
            Estado binario de origen.

        estado_final : list | tuple
            Estado binario destino.

        ncubos : list[int]
            Índices de variables futuras a evaluar.

        Returns
        -------
        None

        Notes
        -----
        El resultado se almacena directamente en:

            self.tabla_transiciones

        usando la clave:

            (
                tuple(estado_inicial),
                tuple(estado_final)
            )
        """

        key = (
            tuple(estado_inicial),
            tuple(estado_final)
        )

        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = [
                None
                for _ in range(
                    len(self.sia_subsistema.indices_ncubos)
                )
            ]

        distancia_hamming = self.hamming(
            estado_inicial,
            estado_final
        )

        factor = 1 / (2 ** distancia_hamming)

        """
        Conversión de estados binarios a enteros
        para indexación eficiente sobre flat_data.

        Se mantiene consistencia con Little-Endian.
        """

        estado_ini_int = int(
            "".join(
                map(
                    str,
                    estado_inicial[::-1]
                )
            ),
            2
        )

        estado_fin_int = int(
            "".join(
                map(
                    str,
                    estado_final[::-1]
                )
            ),
            2
        )

        """
        Diferencia absoluta:

            |X[i] - X[j]|

        para cada n-cubo futuro.
        """

        diffs = np.abs(
            np.array([
                flat[estado_ini_int]
                for flat in self._flat_data
            ])
            -
            np.array([
                flat[estado_fin_int]
                for flat in self._flat_data
            ])
        )

        self.tabla_transiciones[key] = diffs.tolist()

        """
        Si la distancia Hamming es mayor que 1,
        agregamos los costos acumulados desde
        vecinos intermedios.
        """

        if distancia_hamming > 1:

            for i in range(len(estado_inicial)):

                if estado_inicial[i] != estado_final[i]:

                    nuevo_estado = list(estado_final)
                    nuevo_estado[i] = estado_inicial[i]

                    temp_key = (
                        tuple(estado_inicial),
                        tuple(nuevo_estado)
                    )

                    if temp_key not in self.tabla_transiciones:
                        continue

                    for n in ncubos:
                        self.tabla_transiciones[key][n] += (
                            self.tabla_transiciones[temp_key][n]
                        )

        """
        Aplicar factor geométrico:

            γ = 1 / 2^(dh)
        """

        resultado_final = []

        for valor in self.tabla_transiciones[key]:

            if valor is not None:
                resultado_final.append(
                    factor * valor
                )
            else:
                resultado_final.append(
                    None
                )

        self.tabla_transiciones[key] = resultado_final

    def hamming(
        self,
        a,
        b
    ):
        """
        Calcula la distancia de Hamming entre
        dos estados binarios.

        La distancia de Hamming representa la
        cantidad de posiciones en las que dos
        vectores binarios difieren.

        Ejemplo:

            a = [0, 1, 0]
            b = [1, 1, 1]

        entonces:

            hamming(a, b) = 2

        porque difieren en:

            posición 0
            posición 2

        Esta métrica es fundamental en GeoMIP
        porque determina:

            γ = 1 / 2^(dh)

        dentro del cálculo del costo geométrico
        de transición:

            tx(i, j)

        Parameters
        ----------
        a : list | tuple | np.ndarray
            Primer estado binario.

        b : list | tuple | np.ndarray
            Segundo estado binario.

        Returns
        -------
        int
            Distancia de Hamming entre ambos estados.

        Raises
        ------
        ValueError
            Si ambos estados no tienen
            la misma longitud.
        """

        if len(a) != len(b):
            raise ValueError(
                "No es posible calcular distancia "
                "de Hamming entre vectores de "
                "distinta longitud."
            )

        return sum(
            x != y
            for x, y in zip(a, b)
        )

    def nodes_complement(
        self,
        nodes
    ):
        """
        Retorna el complemento de un conjunto de nodos
        respecto al conjunto total de vértices del sistema.

        Es decir:

            complemento = self.vertices - nodes

        donde:

            self.vertices

        contiene todos los vértices definidos en:

            - presente  (ACTUAL)
            - futuro    (EFECTO)

        Este método es principalmente útil para:

        - debugging
        - validación estructural
        - visualización de particiones
        - formateo de resultados

        En GeoMIP original se utilizaba para construir:

            fmt_biparte_q()

        mostrando explícitamente ambos lados de la
        bipartición.

        En k-particiones sigue siendo útil para revisar
        subconjuntos específicos y verificar cobertura.

        Parameters
        ----------
        nodes : list | set | tuple
            Conjunto de nodos cuyo complemento se desea
            calcular.

            Ejemplo:

                [
                    (ACTUAL, 0),
                    (EFECTO, 1)
                ]

        Returns
        -------
        list
            Lista de nodos que NO pertenecen a `nodes`
            pero sí a `self.vertices`.

        Raises
        ------
        ValueError
            Si `self.vertices` no ha sido inicializado.
        """

        if not hasattr(self, "vertices") or not self.vertices:
            raise ValueError(
                "self.vertices no ha sido inicializado. "
                "Debe ejecutarse primero aplicar_estrategia()."
            )

        return list(
            set(self.vertices) - set(nodes)
        )