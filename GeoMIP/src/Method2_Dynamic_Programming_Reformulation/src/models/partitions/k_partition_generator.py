import numpy as np
import copy
from itertools import combinations

from src.constants.base import (
    ACTUAL,
    EFECTO,
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

        2. Explorar múltiples combinaciones
        de semillas geométricas

        3. Inicializar bloques semilla

        4. Asignar presentes por cercanía

        5. Asignar futuros restantes por cercanía

        6. Eliminar duplicados estructurales

        Parameters
        ----------
        k : int
            Número de bloques deseados.

        Returns
        -------
        list
            Lista de particiones candidatas.
        """

        from itertools import combinations

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
        Ordenar futuros por menor costo geométrico.
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

        """
        Extraer índices ordenados.
        """

        indices_ordenados = [
            idx
            for _, idx in pares
        ]

        """
        Generar combinaciones reales
        de semillas.

        Esto explora MUCHAS más
        particiones estructurales.
        """

        top_limit = min(
            12,
            len(indices_ordenados)
        )

        indices_reducidos = (
            indices_ordenados[:top_limit]
        )

        combinaciones_semillas = list(
            combinations(
                indices_reducidos,
                k
            )
)

        if self.logger:

            self.logger.critic(
                f"Explorando "
                f"{len(combinaciones_semillas)} "
                f"combinaciones de semillas."
            )

        for semillas in combinaciones_semillas:

            """
            Inicializar bloques.
            """

            particion = []

            for futuro_seed in semillas:

                bloque = set()

                bloque.add(
                    (EFECTO, futuro_seed)
                )

                particion.append(
                    bloque
                )

            """
            Asignar presentes.
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
            Asignar futuros restantes.
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
                    (EFECTO, futuro)
                )

            """
            Validar estructura mínima.
            """

            if not self.valid_partition(
                particion
            ):
                continue

            """
            Construir firma canónica
            para evitar duplicados.
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

                candidatos.append(
                    copy.deepcopy(
                        particion
                    )
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
        Asigna un nodo ACTUAL al bloque cuya
        semilla EFECTO tenga menor costo geométrico.
        """

        key = (
            tuple(self.caminos[0][0]),
            tuple(self.estado_final)
        )

        costos_globales = self.tabla_transiciones[key]

        mejor_bloque = 0
        mejor_score = float("inf")

        for idx_bloque, semilla in enumerate(semillas):

            costo_semilla = costos_globales[semilla]

            """
            Penalización estructural:
            distancia topológica.
            """

            distancia = abs(
                presente - semilla
            )

            score = (
                costo_semilla
                +
                (0.15 * distancia)
            )

            if score < mejor_score:

                mejor_score = score
                mejor_bloque = idx_bloque

        return mejor_bloque

    def best_block_for_future(
        self,
        futuro,
        semillas
    ):
        """
        Asigna futuros restantes usando
        similitud geométrica real.
        """

        key = (
            tuple(self.caminos[0][0]),
            tuple(self.estado_final)
        )

        costos_globales = self.tabla_transiciones[key]

        costo_futuro = costos_globales[futuro]

        mejor_bloque = 0
        mejor_score = float("inf")

        for idx_bloque, semilla in enumerate(semillas):

            costo_semilla = costos_globales[semilla]

            """
            Diferencia geométrica entre nodos.
            """

            diferencia = abs(
                costo_futuro - costo_semilla
            )

            """
            Penalización estructural leve.
            """

            distancia = abs(
                futuro - semilla
            )

            score = (
                diferencia
                +
                (0.10 * distancia)
            )

            if score < mejor_score:

                mejor_score = score
                mejor_bloque = idx_bloque

        return mejor_bloque

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
                tipo == EFECTO
                for tipo, _ in bloque
            )

            if not (
                tiene_actual
                and tiene_efecto
            ):
                return False

        return True