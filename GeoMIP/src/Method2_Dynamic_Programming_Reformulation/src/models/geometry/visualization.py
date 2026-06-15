import os
import numpy as np
from src.constants.base import ACTUAL, EFECTO

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def obtener_particion_mecanismos(partition, dims_ncubos):
    """
    Agrupa las dimensiones del mecanismo (ACTUAL) según los bloques de la partición.
    """
    # Mapear cada variable del mecanismo a su índice de bloque en la partición
    node_to_block = {}
    for block_idx, block in enumerate(partition):
        for tipo, nodo in block:
            if tipo == ACTUAL:
                node_to_block[nodo] = block_idx
                
    # Organizar por bloques
    bloques_mecanismo = {}
    for idx_local, nodo_abs in enumerate(dims_ncubos):
        block_idx = node_to_block.get(nodo_abs, 0) # Por defecto al bloque 0 si no está
        if block_idx not in bloques_mecanismo:
            bloques_mecanismo[block_idx] = []
        bloques_mecanismo[block_idx].append(idx_local)
        
    return bloques_mecanismo, node_to_block


def proyectar_vertice(coords, n_dims):
    """
    Proyecta un vértice n-dimensional a coordenadas 3D para visualización.
    """
    if n_dims == 0:
        return np.array([0.0, 0.0, 0.0])
    elif n_dims == 1:
        return np.array([coords[0] * 2.0 - 1.0, 0.0, 0.0])
    elif n_dims == 2:
        return np.array([coords[0] * 2.0 - 1.0, coords[1] * 2.0 - 1.0, 0.0])
    elif n_dims == 3:
        return np.array([coords[0] * 2.0 - 1.0, coords[1] * 2.0 - 1.0, coords[2] * 2.0 - 1.0])
    else:
        # Tesseract / Proyección para n >= 4 (Cubos anidados)
        base = np.array([coords[0] * 2.0 - 1.0, coords[1] * 2.0 - 1.0, coords[2] * 2.0 - 1.0])
        for i in range(3, n_dims):
            if coords[i] == 1:
                factor = 0.4 * (1.3 ** (i - 3))
                base = base * (1.0 + factor)
        return base

def construir_centroides_voronoi(k):
    """
    Genera centroides distribuidos uniformemente
    sobre un círculo para representar los bloques.
    """
    centroides = []

    for i in range(k):
        angulo = 2 * np.pi * i / max(k, 1)

        centroides.append(
            np.array([
                np.cos(angulo),
                np.sin(angulo),
                0.0
            ])
        )

    return centroides


def asignar_region_voronoi(punto, centroides):
    """
    Retorna el índice del centroide más cercano.
    """

    distancias = [
        np.linalg.norm(punto - centro)
        for centro in centroides
    ]

    return int(np.argmin(distancias))

def generar_visualizacion_hipercubo(partition, dims_ncubos, ruta_salida="results/k_partition_hypercube.png"):
    """
    Genera y guarda una visualización en 3D/2D del hipercubo de estados usando
    la Proyección de Subespacio Representativo, coloreando las aristas por bloque
    y dibujando los hiperplanos divisores correspondientes de manera limpia y premium.
    """
    if not MATPLOTLIB_AVAILABLE:
        return "[Aviso] Matplotlib no está instalado. No se pudo generar la imagen 3D."

    try:
        # Asegurar directorio de resultados
        os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)

        n_dims = len(dims_ncubos)
        if n_dims == 0:
            return "[Aviso] No hay dimensiones de mecanismo que visualizar."

        # Agrupar variables físicas por bloque de partición
        bloques_mec, node_to_block = obtener_particion_mecanismos(partition, dims_ncubos)
        k = len(partition)

        # SELECCIONAR EJES DE PROYECCIÓN (Máximo 3)
        # Tomamos el primer elemento de cada bloque con variables para asegurar representación de bloques
        ejes_seleccionados = []
        bloques_con_vars = [b for b, v in bloques_mec.items() if len(v) > 0]
        for b in bloques_con_vars[:3]:
            ejes_seleccionados.append(bloques_mec[b][0])

        # Rellenar hasta tener al menos min(n_dims, 3) variables en total
        n_ejes_target = min(n_dims, 3)
        todas_vars = list(range(n_dims))
        for v in todas_vars:
            if len(ejes_seleccionados) >= n_ejes_target:
                break
            if v not in ejes_seleccionados:
                ejes_seleccionados.append(v)

        # Ordenar ejes para mantener consistencia
        ejes_seleccionados = sorted(ejes_seleccionados[:n_ejes_target])
        n_plot = len(ejes_seleccionados)

        # Paleta de colores premium
        colores_premium = [
            "#FF6B6B",  # Coral/Rojo suave
            "#4ECDC4",  # Teal/Cian moderno
            "#845EF7",  # Indigo/Lavanda
            "#FCC419",  # Amarillo oro
            "#51CF66",  # Verde menta
            "#339AF0",  # Azul eléctrico
            "#E64980",  # Rosa
        ]
        
        # Generar todos los vértices del hipercubo del subespacio ({0, 1}^n_plot)
        vertices_coords = []
        for i in range(2**n_plot):
            bin_str = bin(i)[2:].zfill(n_plot)
            coords = tuple(int(bit) for bit in bin_str)
            vertices_coords.append(coords)

        fig = plt.figure(figsize=(10, 8))
        if n_plot >= 3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # Proyectar vértices a [-1, 1]
        posiciones = {}
        for c in vertices_coords:
            pos = []
            for i in range(n_plot):
                pos.append(c[i] * 2.0 - 1.0)
            while len(pos) < 3:
                pos.append(0.0)
            posiciones[c] = np.array(pos)

        centroides = construir_centroides_voronoi(k)

        if n_plot == 2:
            centroides_voronoi = [
                c[:2]
                for c in centroides
            ]
        elif n_plot >= 3:
            centroides_voronoi = centroides
        else:
            centroides_voronoi = None

        if centroides_voronoi is not None:
            nube = np.random.uniform(
                -1.2,
                1.2,
                size=(5000, len(centroides_voronoi[0]))
            )

            regiones = np.array([
                asignar_region_voronoi(
                    punto,
                    centroides_voronoi
                )
                for punto in nube
            ])

            for region in range(k):

                puntos_region = nube[
                    regiones == region
                ]

                if len(puntos_region) == 0:
                    continue

                color_region = colores_premium[
                    region % len(colores_premium)
                ]

                if n_plot >= 3:

                    ax.scatter(
                        puntos_region[:,0],
                        puntos_region[:,1],
                        puntos_region[:,2],
                        color=color_region,
                        alpha=0.03,
                        s=4,
                        zorder=0
                    )

                else:

                    ax.scatter(
                        puntos_region[:,0],
                        puntos_region[:,1],
                        color=color_region,
                        alpha=0.03,
                        s=4,
                        zorder=0
                    )

        # Dibujar aristas
        aristas_dibujadas = set()
        labels_agregadas = set()
        for c1 in vertices_coords:
            for c2 in vertices_coords:
                if sum(x != y for x, y in zip(c1, c2)) == 1:
                    # Encontrar qué dimensión del gráfico cambia
                    dim_cambio_grafica = next(idx for idx, (x, y) in enumerate(zip(c1, c2)) if x != y)
                    # Mapearla a la dimensión física
                    dim_fisica = ejes_seleccionados[dim_cambio_grafica]
                    abs_node = dims_ncubos[dim_fisica]
                    block_idx = node_to_block.get(abs_node, 0)
                    
                    key = tuple(sorted([c1, c2]))
                    if key not in aristas_dibujadas:
                        aristas_dibujadas.add(key)
                        p1 = posiciones[c1]
                        p2 = posiciones[c2]
                        
                        color_arista = colores_premium[
                            block_idx % len(colores_premium)
                        ]                        
                        label_bloque = f"Bloque {block_idx}"
                        
                        # Evitar duplicar leyendas
                        if label_bloque in labels_agregadas:
                            label_str = ""
                        else:
                            label_str = label_bloque
                            labels_agregadas.add(label_bloque)

                        if n_plot >= 3:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                                    color=color_arista, linewidth=3, alpha=0.9,
                                    label=label_str)
                        else:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                                    color=color_arista, linewidth=3, alpha=0.9,
                                    label=label_str)

        # Dibujar vértices
        for c, pos in posiciones.items():
            lbl = "".join(map(str, c))
            
            # Determinar región de Voronoi para este vértice según su posición proyectada
            if centroides_voronoi is not None:
                pos_proj = pos[:len(centroides_voronoi[0])]
                bloque_vertice = asignar_region_voronoi(pos_proj, centroides_voronoi)
            else:
                bloque_vertice = 0
                
            color_vertice = colores_premium[bloque_vertice % len(colores_premium)]

            if n_plot >= 3:
                ax.scatter(pos[0], pos[1], pos[2], color=color_vertice, s=60, zorder=10)
                ax.text(pos[0], pos[1], pos[2] + 0.1, lbl, fontsize=9, color="#1C7ED6", weight='bold', ha='center', zorder=15)
            else:
                ax.scatter(pos[0], pos[1], color=color_vertice, s=60, zorder=10)
                ax.text(pos[0], pos[1] + 0.1, lbl, fontsize=9, color="#1C7ED6", weight='bold', ha='center', zorder=15)

        # Dibujar las fronteras exactas de Voronoi (hiperplanos divisores)
        if k > 1:
            for i in range(k):
                # Ángulo de la frontera entre el centroide i e i+1
                angulo_frontera = 2 * np.pi * (i + 0.5) / k
                cos_f = np.cos(angulo_frontera)
                sin_f = np.sin(angulo_frontera)
                
                color_linea = "#495057" # Gris oscuro para las fronteras
                
                if n_plot >= 3:
                    # En 3D, las fronteras son planos verticales a lo largo del eje Z
                    r_vals = np.linspace(0, 1.5, 5)
                    z_vals = np.linspace(-1.5, 1.5, 5)
                    R, Z = np.meshgrid(r_vals, z_vals)
                    X = R * cos_f
                    Y = R * sin_f
                    ax.plot_surface(X, Y, Z, color=color_linea, alpha=0.15, shade=False, zorder=2)
                else:
                    # En 2D, las fronteras son líneas (rayos desde el origen)
                    ax.plot([0, 1.5 * cos_f], [0, 1.5 * sin_f], color=color_linea, linestyle="--", linewidth=1.5, alpha=0.6, zorder=2)

        # Nombres de variables para los ejes
        def nombre_eje(dim_local):
            var_abs = dims_ncubos[dim_local]
            lbl = chr(65 + var_abs) if var_abs < 26 else f"X{var_abs}"
            b_idx = node_to_block.get(var_abs, 0)
            return f"Var {lbl} (Bloque {b_idx})"

        # Ajustar límites y etiquetas del gráfico
        titulo = f"Proyección de Subespacio ({n_plot}D de {n_dims}D) y Cortes de Partición"
        if n_dims > 3:
            titulo += f"\n(Proyectado sobre variables representativas)"
        
        ax.set_title(titulo, fontsize=12, fontweight='bold', pad=15)
        
        if n_plot >= 3:
            ax.set_xlabel(nombre_eje(ejes_seleccionados[0]), fontsize=10, labelpad=10)
            ax.set_ylabel(nombre_eje(ejes_seleccionados[1]), fontsize=10, labelpad=10)
            ax.set_zlabel(nombre_eje(ejes_seleccionados[2]), fontsize=10, labelpad=10)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-1.5, 1.5)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, linestyle=":", alpha=0.5)
        else:
            ax.set_xlabel(nombre_eje(ejes_seleccionados[0]), fontsize=10)
            if n_plot >= 2:
                ax.set_ylabel(nombre_eje(ejes_seleccionados[1]), fontsize=10)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.grid(True, linestyle=":", alpha=0.5)

        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=True)
        plt.tight_layout()
        plt.savefig(ruta_salida, dpi=150)
        plt.close()
        return f"[Éxito] Visualización del hipercubo guardada en: {ruta_salida}"

    except Exception as e:
        return f"[Error] No se pudo generar la visualización: {str(e)}"


def generar_ascii_hipercubo(partition, dims_ncubos):
    """
    Genera una interpretación de texto sobre los cortes en el hipercubo y la
    distribución de variables, omitiendo la representación visual del cubo en sí.
    """
    n_dims = len(dims_ncubos)
    bloques_mec, node_to_block = obtener_particion_mecanismos(partition, dims_ncubos)
    
    # SELECCIONAR EJES DE PROYECCIÓN (Igual que en la gráfica para consistencia)
    ejes_seleccionados = []
    bloques_con_vars = [b for b, v in bloques_mec.items() if len(v) > 0]
    for b in bloques_con_vars[:3]:
        ejes_seleccionados.append(bloques_mec[b][0])

    n_ejes_target = min(n_dims, 3)
    todas_vars = list(range(n_dims))
    for v in todas_vars:
        if len(ejes_seleccionados) >= n_ejes_target:
            break
        if v not in ejes_seleccionados:
            ejes_seleccionados.append(v)

    ejes_seleccionados = sorted(ejes_seleccionados[:n_ejes_target])
    n_plot = len(ejes_seleccionados)
    
    ascii_out = []
    ascii_out.append("   Interpretación Geométrica de la Partición:")
    ascii_out.append("   " + "═" * 60)
    
    def var_str(idx_local):
        var_abs = dims_ncubos[idx_local]
        lbl = chr(65 + var_abs) if var_abs < 26 else f"X{var_abs}"
        b_idx = node_to_block.get(var_abs, 0)
        return f"{lbl} (Bloque {b_idx})"
    
    # Explicación de cortes en los ejes del subespacio
    if n_plot >= 2:
        ascii_out.append("   Ejes del Subespacio de Estados Proyectado:")
        if n_plot == 3:
            ascii_out.append(f"     - Eje X: {var_str(ejes_seleccionados[0])}")
            ascii_out.append(f"     - Eje Y: {var_str(ejes_seleccionados[1])}")
            ascii_out.append(f"     - Eje Z: {var_str(ejes_seleccionados[2])}")
            
            corta_a = node_to_block.get(dims_ncubos[ejes_seleccionados[0]], 0) != node_to_block.get(dims_ncubos[ejes_seleccionados[1]], 0)
            corta_b = node_to_block.get(dims_ncubos[ejes_seleccionados[1]], 0) != node_to_block.get(dims_ncubos[ejes_seleccionados[2]], 0)
            
            cortes = []
            if corta_a:
                cortes.append("un plano divisor entre el Eje X y Eje Y")
            if corta_b:
                cortes.append("un plano divisor entre el Eje Y y Eje Z")
            
            if cortes:
                ascii_out.append(f"   Hiperplanos divisores detectados: Hay " + " y ".join(cortes) + ".")
            else:
                ascii_out.append("   Hiperplanos divisores detectados: Las variables seleccionadas pertenecen al mismo bloque, no hay cortes en este subespacio.")
        elif n_plot == 2:
            ascii_out.append(f"     - Eje X: {var_str(ejes_seleccionados[0])}")
            ascii_out.append(f"     - Eje Y: {var_str(ejes_seleccionados[1])}")
            
            corta_a = node_to_block.get(dims_ncubos[ejes_seleccionados[0]], 0) != node_to_block.get(dims_ncubos[ejes_seleccionados[1]], 0)
            if corta_a:
                ascii_out.append("   Hiperplanos divisores detectados: Hay un corte vertical divisor entre el Eje X y Eje Y.")
            else:
                ascii_out.append("   Hiperplanos divisores detectados: No hay cortes en este subespacio.")
    else:
        ascii_out.append(f"   Espacio de estados unidimensional o sin variables de mecanismo.")
        
    ascii_out.append("")
    ascii_out.append("   Distribución completa de variables por bloque de partición:")
    for b_idx in sorted(bloques_mec.keys()):
        variables = bloques_mec[b_idx]
        nombres_vars = [chr(65 + dims_ncubos[v]) if dims_ncubos[v] < 26 else f"X{dims_ncubos[v]}" for v in variables]
        ascii_out.append(f"     - Bloque {b_idx}: {nombres_vars}")
            
    ascii_out.append("   " + "═" * 60)
    return "\n".join(ascii_out)
