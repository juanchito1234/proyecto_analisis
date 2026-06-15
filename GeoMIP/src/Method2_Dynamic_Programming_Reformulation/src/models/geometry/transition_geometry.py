import numpy as np
import time


class TransitionGeometry:
    """
    Servicio encargado de construir y administrar
    toda la geometría de transición utilizada
    por KGeometric.

    Esta clase encapsula:

    - construcción de caminos geométricos
    - cálculo de costos de transición
    - distancia de Hamming
    - tabla de costos acumulados
    - acceso eficiente a N-Cubos flatten

    Objetivos:
    -----------

    1. Desacoplar lógica geométrica de KGeometric

    2. Reutilizar geometría en futuras estrategias:
        - beam search
        - branch and bound
        - greedy geometric
        - annealing

    3. Centralizar optimizaciones matemáticas

    4. Facilitar profiling y benchmarking

    Arquitectura:
    --------------

    caminos:
        nivel -> estados alcanzables

    tabla_transiciones:
        (
            estado_inicial,
            estado_final
        )
        ->
        [
            costo_ncubo_0,
            costo_ncubo_1,
            ...
        ]
    """

    def __init__(
        self,
        sia_subsistema,
        estado_inicial,
        estado_final
    ):
        """
        Inicializa la geometría de transición.

        Parameters
        ----------
        sia_subsistema :
            Subsistema SIA preparado.

        estado_inicial : np.ndarray
            Estado binario inicial.

        estado_final : np.ndarray
            Estado binario objetivo.
        """

        self.sia_subsistema = sia_subsistema

        self.estado_inicial = estado_inicial

        self.estado_final = estado_final

        """
        Tabla principal de costos geométricos.
        """

        self.tabla_transiciones = {}

        """
        Estados agrupados por nivel
        de distancia Hamming.
        """

        self.caminos = {}

        """
        Índices de variables futuras.
        """

        self.idx_ncubos = list(
            range(
                len(
                    self.sia_subsistema
                    .indices_ncubos
                )
            )
        )

        """
        Flatten de N-Cubos para acceso rápido.
        """

        self._flat_data = []

        for ncubo in self.sia_subsistema.ncubos:

            self._flat_data.append(
                ncubo.data.ravel()
            )

        """
        Nivel base.
        """

        self.caminos = {
            0: [
                self.estado_inicial.tolist()
            ]
        }

        """
        Costo base:
        estado inicial -> estado inicial.
        """

        key_base = (
            tuple(self.estado_inicial),
            tuple(self.estado_inicial)
        )

        self.tabla_transiciones[
            key_base
        ] = [
            0.0
            for _ in range(
                len(
                    self.sia_subsistema
                    .indices_ncubos
                )
            )
        ]

        """
         Mapeo de estados a enteros para indexación rápida.
        """
        self._state_to_int = {
            tuple(self.estado_inicial): 0
        }

        self._flat_data = np.array(
            [
                ncubo.data.ravel()
                for ncubo in self.sia_subsistema.ncubos
            ]
        )

        self._factores = {
            i: 1/(2**i)
            for i in range(
                len(self.estado_inicial)+1
            )
        }

        self.num_costos_calculados = 0

    def build_geometry(self):
        """
        Construye completamente la geometría
        de transición del sistema.

        Flujo:
        -------

        1. Explorar estados por distancia
           de Hamming

        2. Construir caminos geométricos

        3. Calcular costos acumulados

        4. Poblar tabla_transiciones

        Returns
        -------
        None
        """

        t0 = time.time()
        total_bits = len(
            self.estado_inicial
        )

        for nivel in range(1, total_bits + 1):
            t_nivel = time.time()

            self.calcular_costos_nivel(
                self.estado_final,
                nivel
            )

    def calcular_costos_nivel(
        self,
        estado_final,
        nivel,
    ):
        """
        Construye estados alcanzables para
        un nivel específico de distancia
        Hamming.

        Parameters
        ----------
        estado_final : np.ndarray

        nivel : int

        Returns
        -------
        None
        """

        n = len(estado_final)

        visitados = set()

        self.caminos[nivel] = []

        for estado_anterior in (
            self.caminos[nivel - 1]
        ):

            estado_actual = np.array(
                estado_anterior
            )

            for i in range(n):

                if (
                    estado_actual[i]
                    !=
                    estado_final[i]
                ):

                    nuevo_estado = (
                        estado_actual.copy()
                    )

                    nuevo_estado[i] = (
                        estado_final[i]
                    )

                    nuevo_estado_tuple = tuple(
                        nuevo_estado
                    )

                    if (
                        nuevo_estado_tuple
                        not in visitados
                    ):

                        self.caminos[
                            nivel
                        ].append(
                            nuevo_estado.tolist()
                        )

                        visitados.add(
                            nuevo_estado_tuple
                        )

                        self.state_to_int(nuevo_estado)

    def calcular_costo(
        self,
        estado_inicial,
        estado_final,
        ncubos,
        distancia_hamming
    ):
        """
        Calcula el costo geométrico:

            tx(i,j)

        entre dos estados binarios.

        Fórmula:
        ----------

            tx(i,j) =
                γ * (
                    |X[i] - X[j]|
                    +
                    Σ tx(k,j)
                )

        donde:

            γ = 1 / 2^(dh)

        Parameters
        ----------
        estado_inicial : list

        estado_final : list

        ncubos : list[int]

        Returns
        -------
        None
        """
        self.num_costos_calculados += 1

        key = (
            tuple(estado_inicial),
            tuple(estado_final)
        )

        if key not in (
            self.tabla_transiciones
        ):

            self.tabla_transiciones[
                key
            ] = [
                None
                for _ in range(
                    len(
                        self.sia_subsistema
                        .indices_ncubos
                    )
                )
            ]

        factor = self._factores[
            distancia_hamming
        ]

        """
        Conversión Little-Endian
        para indexación rápida.
        """

        estado_ini_int = self.state_to_int(
            estado_inicial
        )

        estado_fin_int = self.state_to_int(
            estado_final
        )

        """
        Diferencias absolutas.
        """

        diffs = np.abs(
            self._flat_data[:, estado_ini_int]
            -
            self._flat_data[:, estado_fin_int]
        ).astype(np.float64)

        self.tabla_transiciones[key] = diffs

        """
        Costos acumulados desde vecinos
        intermedios.
        """

        if distancia_hamming > 1:

            for i in range(
                len(estado_inicial)
            ):

                if (
                    estado_inicial[i]
                    !=
                    estado_final[i]
                ):

                    nuevo_estado = list(
                        estado_final
                    )

                    nuevo_estado[i] = (
                        estado_inicial[i]
                    )

                    temp_key = (
                        tuple(
                            estado_inicial
                        ),
                        tuple(
                            nuevo_estado
                        )
                    )

                    if temp_key not in (
                        self.tabla_transiciones
                    ):
                        continue

                    self.tabla_transiciones[key] += (
                        self.tabla_transiciones[temp_key]
                    )

        self.tabla_transiciones[key] = (
            factor * self.tabla_transiciones[key]
        )

    def hamming(
        self,
        a,
        b
    ):
        return sum(
            x != y
            for x, y in zip(a, b)
        )

    def get_transition_table(
        self
    ):
        """
        Retorna tabla geométrica
        completa.

        Returns
        -------
        dict
        """

        return self.tabla_transiciones

    def get_paths(
        self
    ):
        """
        Retorna caminos geométricos.

        Returns
        -------
        dict
        """

        return self.caminos
    
    def state_to_int(self, estado):
        key = tuple(estado)

        if key not in self._state_to_int:

            self._state_to_int[key] = int(
                "".join(
                    map(
                        str,
                        estado[::-1]
                    )
                ),
                2
            )

        return self._state_to_int[key]
    
    def get_cost(
        self,
        estado_inicial,
        estado_final
    ):
        key = (
            tuple(estado_inicial),
            tuple(estado_final)
        )

        if key not in self.tabla_transiciones:
            self.num_costos_calculados += 1

            distancia_hamming = self.hamming(
                estado_inicial,
                estado_final
            )

            self.calcular_costo(
                estado_inicial,
                estado_final,
                self.idx_ncubos,
                distancia_hamming
            )

        return self.tabla_transiciones[key]