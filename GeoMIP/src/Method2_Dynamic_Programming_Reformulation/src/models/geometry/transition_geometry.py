import numpy as np


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

        total_bits = len(
            self.estado_inicial
        )

        for nivel in range(
            1,
            total_bits + 1
        ):

            self.calcular_costos_nivel(
                self.estado_final,
                nivel
            )

    def calcular_costos_nivel(
        self,
        estado_final,
        nivel
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

        distancia_hamming = (
            self.hamming(
                estado_inicial,
                estado_final
            )
        )

        factor = (
            1 /
            (2 ** distancia_hamming)
        )

        """
        Conversión Little-Endian
        para indexación rápida.
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
        Diferencias absolutas.
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

        self.tabla_transiciones[key] = (
            diffs.tolist()
        )

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

                    for n in ncubos:

                        self.tabla_transiciones[
                            key
                        ][n] += (
                            self.tabla_transiciones[
                                temp_key
                            ][n]
                        )

        """
        Aplicar factor geométrico.
        """

        resultado_final = []

        for valor in (
            self.tabla_transiciones[key]
        ):

            if valor is not None:

                resultado_final.append(
                    factor * valor
                )

            else:

                resultado_final.append(
                    None
                )

        self.tabla_transiciones[
            key
        ] = resultado_final

    def hamming(
        self,
        a,
        b
    ):
        """
        Calcula distancia de Hamming
        entre dos secuencias binarias.
        """

        if isinstance(a, str):
            a = list(a)

        if isinstance(b, str):
            b = list(b)

        max_len = max(
            len(a),
            len(b)
        )

        a = list(a)
        b = list(b)

        while len(a) < max_len:
            a.insert(0, "0")

        while len(b) < max_len:
            b.insert(0, "0")

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