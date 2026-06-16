import numpy as np

from functools import lru_cache
from src.middlewares.tracker import track_time
from src.funcs.base import emd_efecto


class PartitionEvaluator:
    """
    Servicio encargado de evaluar k-particiones
    sobre un subsistema SIA.

    Esta clase desacopla toda la lógica de:

    - validación de particiones
    - construcción de sistemas particionados
    - cálculo de distribuciones marginales
    - cálculo de pérdida EMD
    - memoización multinivel

    del motor principal KGeometric.

    Objetivos:
    -----------

    1. Reducir complejidad de KGeometric

    2. Reutilizar evaluación en futuras estrategias:
        - beam search
        - simulated annealing
        - greedy geometric
        - branch and bound

    3. Centralizar optimizaciones de performance

    4. Evitar recomputaciones costosas

    Arquitectura de memoización:
    ----------------------------

    memoria_marginales:
        key -> distribucion_marginal

    memoria_particiones:
        key -> (
            perdida,
            distribucion
        )
    """

    def __init__(
        self,
        sia_subsistema,
        distribucion_original
    ):
        """
        Inicializa el evaluador de particiones.

        Parameters
        ----------
        sia_subsistema :
            Subsistema SIA preparado.

        distribucion_original : np.ndarray
            Distribución marginal original del
            subsistema completo.
        """

        self.sia_subsistema = sia_subsistema

        self.distribucion_original = (
            distribucion_original
        )

        """
        Cache final:
        pérdida + distribución.
        """
        self.memoria_particiones = {}

        """
        Cache intermedia:
        distribuciones marginales.
        """
        self.memoria_marginales = {}

        self.cache_hits = 0
        self.cache_misses = 0

    @track_time("Evaluar Particiones")
    def evaluate_partition(
        self,
        partition,
        split_function
    ):
        """
        Evalúa una k-partición candidata y calcula
        su pérdida de información.

        Esta función incorpora memoización multinivel
        para minimizar recomputaciones costosas.

        Flujo:
        ------

        1. Convertir partición abstracta

        2. Construir clave hashable canónica

        3. Revisar cache final

        4. Revisar cache intermedia

        5. Construir sistema particionado
           si es necesario

        6. Calcular EMD-Effect

        7. Guardar resultado final

        Parameters
        ----------
        partition : list[set]
            Partición abstracta.

        split_function : callable
            Función encargada de transformar:

                partición abstracta

            en:

                (
                    alcance,
                    mecanismo
                )

        Returns
        -------
        tuple
            (
                perdida,
                distribucion_particion
            )

        Raises
        ------
        ValueError
            Si la partición es inválida.
        """

        if not partition:
            raise ValueError(
                "No se puede evaluar una "
                "partición vacía."
            )

        particion_formateada = split_function(
            partition
        )

        key = self.partition_key(
            particion_formateada
        )

        """
        Cache FINAL:
        pérdida + distribución.
        """

        if key in self.memoria_particiones:
            self.cache_hits += 1
            return self.memoria_particiones[key]
        else:
            self.cache_misses += 1

        """
        Cache INTERMEDIA:
        distribuciones marginales.
        """

        if key in self.memoria_marginales:

            distribucion_particion = (
                self.memoria_marginales[key]
            )

        else:

            self.sia_subsistema.validar_k_particion(
                particion_formateada
            )

            sistema_particionado = (
                self.sia_subsistema.k_particionar(
                    particion_formateada
                )
            )

            distribucion_particion = (
                sistema_particionado
                .distribucion_marginal()
            )

            self.memoria_marginales[key] = (
                distribucion_particion
            )

        """
        Calcular pérdida EMD.
        """

        perdida = emd_efecto(
            distribucion_particion,
            self.distribucion_original
        )

        resultado = (
            perdida,
            distribucion_particion
        )

        """
        Guardar resultado final.
        """

        self.memoria_particiones[key] = (
            resultado
        )

        return resultado

    def partition_key(
        self,
        particion_formateada
    ):
        """
        Construye una representación hashable,
        ordenada y canónica de una k-partición.

        Esto permite:

        - memoización
        - comparación estructural
        - eliminación de duplicados

        Importante:
        ------------

        Dos particiones equivalentes deben generar
        exactamente la misma clave aunque los bloques
        estén permutados.

        Parameters
        ----------
        particion_formateada : list
            Lista de bloques:

                [
                    (
                        alcance,
                        mecanismo
                    ),
                    ...
                ]

        Returns
        -------
        tuple
            Clave hashable canónica.
        """

        bloques = []

        for alcance, mecanismo in particion_formateada:

            bloques.append(
                (
                    tuple(
                        sorted(
                            alcance.tolist()
                        )
                    ),
                    tuple(
                        sorted(
                            mecanismo.tolist()
                        )
                    )
                )
            )

        return tuple(
            sorted(bloques)
        )

    def clear_cache(self):
        """
        Limpia completamente las memorias internas.

        Útil para:

        - benchmarking
        - reinicios experimentales
        - cambio de subsistema
        """

        self.memoria_particiones.clear()

        self.memoria_marginales.clear()
        
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_stats(self):
        """
        Retorna estadísticas simples de cache.

        Returns
        -------
        dict
            Información de utilización de memoria.
        """

        return {
            "particiones": len(
                self.memoria_particiones
            ),
            "marginales": len(
                self.memoria_marginales
            )
        }

    def get_cache_performance(self):
        """
        Retorna las métricas de desempeño de la cache.
        """
        total = self.cache_hits + self.cache_misses
        porcentaje = (self.cache_hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "porcentaje": porcentaje
        }