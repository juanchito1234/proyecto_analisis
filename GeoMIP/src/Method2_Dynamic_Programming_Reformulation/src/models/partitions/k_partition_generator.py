import numpy as np
import copy
from copy import deepcopy
from itertools import combinations
from src.middlewares.tracker import track_time

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

    @track_time("Inicializar Generador")
    def __init__(
        self,
        sia_subsistema,
        tabla_transiciones,
        caminos,
        estado_final,
        logger=None
    ):
        self.sia_subsistema = sia_subsistema
        self.tabla_transiciones = tabla_transiciones
        self.caminos = caminos
        self.estado_final = estado_final
        self.logger = logger
        self.MAX_PARTICIONES_POR_SEMILLA = 50
        self.MAX_CANDIDATOS_TOTALES = 2000
        self.num_consultas = 0
        self.estados_usados = set()
        
        # Pre-calcular matriz de conectividad (Influencia Mecanismo -> Efecto)
        # C[idx_mecanismo][idx_efecto] = Cambio promedio en P(efecto) al mutar mecanismo
        self.conectividad = self._calcular_conectividad()

    def _calcular_conectividad(self):
        abs_p = self.sia_subsistema.dims_ncubos
        abs_f = self.sia_subsistema.indices_ncubos
        matrix = {}
        
        for f_idx, cube in enumerate(self.sia_subsistema.ncubos):
            abs_f_id = abs_f[f_idx]
            for p_idx, p_id in enumerate(cube.dims):
                # La influencia es el promedio de la diferencia absoluta
                # entre las caras del n-cubo al variar la dimensión p
                data = cube.data
                # Crear rodajas para comparar p=0 con p=1
                slice0 = [slice(None)] * len(cube.dims)
                slice1 = [slice(None)] * len(cube.dims)
                slice0[p_idx] = 0
                slice1[p_idx] = 1
                
                diff = np.mean(np.abs(data[tuple(slice0)] - data[tuple(slice1)]))
                matrix[(p_id, abs_f_id)] = diff
        return matrix

    @track_time("Generar Candidatos")
    def identificar_particiones_candidatas(self, k: int):
        candidatos = []
        firmas = set()
        estado_base = tuple(self.caminos[0][0])
        
        abs_p = self.sia_subsistema.dims_ncubos
        abs_f = self.sia_subsistema.indices_ncubos
        
        def registrar_particion(bloques):
            if not self.valid_partition(bloques): return
            firma = tuple(sorted(tuple(sorted(b)) for b in bloques))
            if firma not in firmas:
                firmas.add(firma)
                candidatos.append(copy.deepcopy(bloques))

        node_to_local_idx = {node_abs: idx for idx, node_abs in enumerate(abs_p)}

        def asignar_mecanismos_a_efectos(particion_efectos):
            """
            Dada una partición de efectos {E1, ..., Ek}, asigna cada mecanismo
            m_p al bloque Ej que tiene la mayor influencia sobre él.
            """
            bloques = [set() for _ in range(len(particion_efectos))]
            for i, e_set in enumerate(particion_efectos):
                for eid in e_set:
                    bloques[i].add((EFECTO, eid))
            
            for p_id in abs_p:
                mejor_bloque = 0
                max_infl = -1.0
                for i, e_set in enumerate(particion_efectos):
                    infl_total = sum(self.conectividad.get((p_id, eid), 0.0) for eid in e_set)
                    if infl_total > max_infl:
                        max_infl = infl_total
                        mejor_bloque = i
                bloques[mejor_bloque].add((ACTUAL, p_id))
            
            registrar_particion(bloques)

        def asignar_mecanismos_geometrica(e_seed, p_rest, estado):
            """
            Asigna presentes basándose en el estado de Hamming de la semilla:
            - Si el presente NO cambió (coincide con estado_base), va al bloque de e_seed (bloque 0).
            - Si cambió, se asigna al bloque de p_rest con mayor conectividad.
            """
            bloques = [set() for _ in range(k)]
            # Inicializar efectos
            for eid in e_seed:
                bloques[0].add((EFECTO, eid))
            for i, e_set in enumerate(p_rest):
                for eid in e_set:
                    bloques[i + 1].add((EFECTO, eid))
                    
            # Asignar presentes
            for p_id in abs_p:
                local_idx = node_to_local_idx[p_id]
                if estado[local_idx] == estado_base[local_idx]:
                    bloques[0].add((ACTUAL, p_id))
                else:
                    if k > 2:
                        mejor_bloque_restante = 0
                        max_infl = -1.0
                        for i, e_set in enumerate(p_rest):
                            infl_total = sum(self.conectividad.get((p_id, eid), 0.0) for eid in e_set)
                            if infl_total > max_infl:
                                max_infl = infl_total
                                mejor_bloque_restante = i
                        bloques[mejor_bloque_restante + 1].add((ACTUAL, p_id))
                    else:
                        bloques[1].add((ACTUAL, p_id))
            registrar_particion(bloques)

        def generar_particiones_efectos(elementos, sub_k):
            if sub_k == 1:
                yield [set(elementos)]
                return
            if sub_k == len(elementos):
                yield [set([e]) for e in elementos]
                return
            if sub_k > len(elementos): return
            
            first = elementos[0]
            rest = elementos[1:]
            for p in generar_particiones_efectos(rest, sub_k - 1):
                yield [set([first])] + p
            for p in generar_particiones_efectos(rest, sub_k):
                for i in range(len(p)):
                    new_p = [s.copy() for s in p]
                    new_p[i].add(first)
                    yield new_p

        # 1. Semillas de Un solo Efecto (MIPs clásicas)
        if k == 2:
            for f_id in abs_f:
                e1 = {f_id}
                e2 = set(abs_f) - e1
                asignar_mecanismos_a_efectos([e1, e2])
        
        # 2. Semillas Geométricas (Caminos Hamming)
        niveles = sorted(self.caminos.keys())
        for nivel in niveles[1:]:
            if (
                len(candidatos)
                >=
                self.MAX_CANDIDATOS_TOTALES
            ):
                break
            for estado in self.caminos[nivel]:
                estado = np.array(estado)
                estado_comp = 1 - estado
                key_act = (estado_base, tuple(estado))
                key_cmp = (estado_base, tuple(estado_comp))
                
                self.num_consultas += 1
                self.estados_usados.add(key_act)
                act = self.tabla_transiciones.get_cost(
                    estado_base,
                    estado.tolist()
                )

                self.num_consultas += 1
                self.estados_usados.add(key_cmp)
                cmp = self.tabla_transiciones.get_cost(
                    estado_base,
                    estado_comp.tolist()
                )
                if act is None or cmp is None: continue

                # Identificar bloque de efectos sugerido por la geometría
                e_seed = set()
                for idx_f, f_id in enumerate(abs_f):
                    if act[idx_f] <= cmp[idx_f]:
                        e_seed.add(f_id)
                
                if not e_seed or len(e_seed) == len(abs_f): continue

                # Dividir el resto de efectos en k-1 bloques
                rest_e = sorted(list(set(abs_f) - e_seed))
                if len(rest_e) >= k - 1:
                    contador_semilla = 0
                    for p_rest in generar_particiones_efectos(
                        rest_e,
                        k - 1
                    ):
                        # Generar candidatos con ambas heurísticas
                        asignar_mecanismos_a_efectos(
                            [e_seed] + p_rest
                        )
                        asignar_mecanismos_geometrica(
                            e_seed,
                            p_rest,
                            estado
                        )

                        contador_semilla += 1

                        if (
                            contador_semilla
                            >=
                            self.MAX_PARTICIONES_POR_SEMILLA
                        ):
                            break

                        if (
                            len(candidatos)
                            >=
                            self.MAX_CANDIDATOS_TOTALES
                        ):
                            break

        # 3. Si k es pequeño, añadir partición de efectos puramente exhaustiva
        if len(abs_f) <= 8 and k <= 3:
            for p_efectos in generar_particiones_efectos(list(abs_f), k):
                asignar_mecanismos_a_efectos(p_efectos)
        
        candidatos.sort(
            key=self.score_particion,
            reverse=True
        )

        # Truncar dinámicamente según el tamaño del sistema para mantener el tiempo acotado
        if len(abs_f) > 20:
            limite_candidatos = 10
        elif len(abs_f) > 15:
            limite_candidatos = 25
        elif len(abs_f) > 10:
            limite_candidatos = 100
        else:
            limite_candidatos = 300

        candidatos = candidatos[:limite_candidatos]
        return candidatos

    def valid_partition(self, particion):
        if not particion: return False
        efectos_cubiertos = set()
        for bloque in particion:
            if not bloque: return False
            tiene_efecto = False
            for tipo, nodo in bloque:
                if tipo == EFECTO:
                    efectos_cubiertos.add(nodo)
                    tiene_efecto = True
            if not tiene_efecto: return False
        return efectos_cubiertos == set(self.sia_subsistema.indices_ncubos.tolist())
    
    def score_particion(self, particion):
        score = 0.0
        for bloque in particion:
            mecanismos = [n for t, n in bloque if t == ACTUAL]
            efectos = [n for t, n in bloque if t == EFECTO]
            for p in mecanismos:
                for e in efectos:
                    score += self.conectividad.get((p, e), 0.0)
        return score
