import numpy as np
import copy

from src.constants.base import (
    ACTUAL,
    EFFECT,
)


class KPartitionGenerator:
    """
    Generador geométrico de k-particiones candidatas.

    Esta clase encapsula toda la lógica heurística
    usada para construir particiones prometedoras
    antes de la evaluación EMD.

    Objetivos:
    -----------

    1. Desacoplar generación de candidatos
       de KGeometric

    2. Permitir múltiples estrategias:
        - geométrica
        - greedy
        - clustering
        - beam search

    3. Reducir complejidad del motor principal

    4. Centralizar optimizaciones estructurales

    Estrategia actual:
    ------------------

    Se usa una heurística geométrica basada en:

    - costos de transición
    - semillas geométricas
    - clusterización por cercanía

    evitando el antiguo esquema round-robin.

    Idea principal:
    ----------------

    1. Seleccionar semillas geométricas
       de bajo costo

    2. Construir centroides implícitos

    3. Asignar presentes y futuros restantes
       al bloque geométricamente más cercano

    Esto produce particiones mucho más coherentes
    estructuralmente que round-robin.
    """

    def __init__(
        self,
        sia_subsistema,
        tabla_transiciones,
        caminos,
        estado_final,
        logger=None
    ):
        """
        Inicializa el generador geométrico.

        Parameters
        ----------
        sia_subsistema :
            Subsistema SIA preparado.

        tabla_transiciones : dict
            Tabla geométrica de costos.

        caminos : dict
            Estados agrupados por distancia
            de Hamming.

        estado_final : np.ndarray
            Estado objetivo geométrico.

        logger : optional
            Logger del sistema.
        """

        self.sia_subsistema = sia_subsistema

        self.tabla_transiciones = (
            tabla_transiciones
        )

        self.caminos = caminos

        self.estado_final = estado_final

        self.logger = logger

    def identificar_particiones_candidatas(
        self,
        k: int
    ):
        """
        Construye múltiples k-particiones candidatas
        usando clusterización geométrica.

        Flujo:
        -------

        1. Obtener costos globales

        2. Ordenar futuros por menor costo

        3. Seleccionar ventanas de semillas

        4. Inicializar bloques semilla

        5. Asignar presentes por cercanía

        6. Asignar futuros restantes por cercanía

        7. Eliminar duplicados estructurales

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
                "No existe información geométrica "
                "para construir particiones."
            )

        costos = self.tabla_transiciones[key]

        pares = [
            (valor, idx)
            for idx, valor in enumerate(costos)
            if valor is not None
        ]

        if len(pares) < k:
            raise ValueError(
                f"No existen suficientes futuros "
                f"para construir k={k}."
            )

        """
        Ordenar futuros por menor costo.
        """

        pares.sort(
            key=lambda x: x[0]
        )

        total_futuros = len(
            self.sia_subsistema.indices_ncubos
        )

        presentes_totales = list(
            self.sia_subsistema.dims_ncubos
        )

        candidatos = []
        firmas_vistas = set()
        max_offset = len(pares) - k + 1
        num_trials = min(20, max_offset)
        offsets = np.linspace(0, max_offset - 1, num_trials, dtype=int)

        for offset in offsets:

            """
            Seleccionar semillas geométricas.
            """

            semillas = [
                idx
                for _, idx in pares[
                    offset:offset + k
                ]
            ]

            particion = []

            for futuro_seed in semillas:

                bloque = set()

                bloque.add(
                    (EFFECT, futuro_seed)
                )

                particion.append(
                    bloque
                )

            """
            Asignar presentes usando
            clusterización geométrica.
            """

            for presente in presentes_totales:

                bloque_destino = (
                    self.best_block_for_present(
                        presente,
                        semillas
                    )
                )

                particion[
                    bloque_destino
                ].add(
                    (ACTUAL, presente)
                )

            """
            Asignar futuros restantes
            usando cercanía geométrica.
            """

            futuros_restantes = [
                idx
                for idx in range(total_futuros)
                if idx not in semillas
            ]

            for futuro in futuros_restantes:

                bloque_destino = (
                    self.best_block_for_future(
                        futuro,
                        semillas
                    )
                )

                particion[
                    bloque_destino
                ].add(
                    (EFFECT, futuro)
                )

            """
            Validar bloques mínimos.

            Cada bloque debe contener:
                - al menos un ACTUAL
                - al menos un EFECTO
            """

            if not self.valid_partition(
                particion
            ):
                continue

            """
            Firma estructural canónica.
            """

            firma = tuple(
                sorted(
                    tuple(
                        sorted(bloque)
                    )
                    for bloque in particion
                )
            )

            if firma not in firmas_vistas:

                firmas_vistas.add(
                    firma
                )
                
                particion = copy.deepcopy(particion)

                candidatos.append(
                    particion
                )

        if not candidatos:
            raise ValueError(
                "No fue posible generar "
                "particiones candidatas."
            )

        if self.logger:

            self.logger.critic(
                f"Generadas "
                f"{len(candidatos)} "
                f"particiones candidatas."
            )

        return candidatos

    def best_block_for_present(
        self,
        presente,
        semillas
    ):
        """
        Determina el bloque más cercano
        para una variable presente.

        Heurística:
        ------------

        Se aproxima cercanía usando:

            |presente - semilla|

        Esto mantiene coherencia topológica
        entre variables presentes y futuras.

        Parameters
        ----------
        presente : int

        semillas : list[int]

        Returns
        -------
        int
            Índice del mejor bloque.
        """

        distancias = []

        for semilla in semillas:
            # penaliza distancia + ruido leve para evitar empates
            score = (
                0.7 * abs(presente - semilla)
                + 0.3 * np.random.uniform(0, 0.05)
            )
            distancias.append(score)

        return int(np.argmin(distancias))

    def best_block_for_future(
        self,
        futuro,
        semillas
    ):
        """
        Determina el bloque geométricamente
        más cercano para un futuro restante.

        Heurística:
        ------------

        Se utiliza distancia relativa
        respecto a semillas geométricas.

        Parameters
        ----------
        futuro : int

        semillas : list[int]

        Returns
        -------
        int
            Índice del mejor bloque.
        """

        distancias = []

        for semilla in semillas:
            score = (
                0.7 * abs(futuro - semilla)
                + 0.3 * np.random.uniform(0, 0.05)
            )
            distancias.append(score)

        return int(np.argmin(distancias))

    def valid_partition(
        self,
        particion
    ):
        """
        Verifica que todos los bloques
        tengan estructura válida.

        Reglas:
        --------

        Cada bloque debe contener:

            - mínimo un ACTUAL
            - mínimo un EFECTO

        Parameters
        ----------
        particion : list[set]

        Returns
        -------
        bool
        """

        for bloque in particion:

            tiene_actual = any(
                tipo == ACTUAL
                for tipo, _ in bloque
            )

            tiene_efecto = any(
                tipo == EFFECT
                for tipo, _ in bloque
            )

            if not (
                tiene_actual
                and tiene_efecto
            ):
                return False

        return True