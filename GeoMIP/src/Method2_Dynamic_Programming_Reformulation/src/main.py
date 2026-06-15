# from src.controllers.manager import Manager

# from src.controllers.strategies.force import BruteForce
# from src.controllers.strategies.q_nodes import QNodes
# from src.controllers.strategies.geometric import GeometricSIA


# def iniciar():
#     """Punto de entrada principal"""
#                     # ABCD #
#     # estado_inicial = "100"
#     # condiciones =    "111"
#     # alcance =        "111"
#     # mecanismo =      "111"
#     # estado_inicial = "0000"
#     # condiciones =    "1111"
#     # alcance =        "1111"
#     # mecanismo =      "1111"
#     # estado_inicial = "1000"
#     # condiciones =    "1111"
#     # alcance =        "0111"
#     # mecanismo =      "1111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "101011"
#     # mecanismo =      "111111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "111111"
#     # mecanismo =      "111111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "111111"
#     # mecanismo =      "011111"
#     # estado_inicial = "1000000000"
#     # condiciones =    "1111111111"
#     # alcance =        "1111111111"
#     # mecanismo =      "1111111111"
#     estado_inicial = "1000000000"
#     condiciones =    "1111111111"
#     alcance =        "0101010101"
#     mecanismo =      "1111111111"
#     # estado_inicial = "1000000000"
#     # condiciones =    "1111111111"
#     # alcance =        "1111111110"
#     # mecanismo =      "1111111111"
#     # estado_inicial = "10000000000000000000"
#     # condiciones =    "11111111111111111111"
#     # alcance =        "11111111111111111111"
#     # mecanismo =      "11111111111111111111"
#     # estado_inicial = "10000000000000000000"
#     # condiciones =    "11111111111111111111"
#     # alcance =        "11011011011011011011"
#     # mecanismo =      "10101010101010101010"

#     gestor_sistema = Manager(estado_inicial)

#     ### Ejemplo de solución mediante módulo de fuerza bruta ###
#     analizador_fb = GeometricSIA(gestor_sistema)
#     # analizador_fb = BruteForce(gestor_sistema)
#     sia_uno = analizador_fb.aplicar_estrategia(
#         condiciones,
#         alcance,
#         mecanismo,
#     )
#     print(sia_uno)
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.k_geometric import KGeometric
# Optional import: this project often runs only geometric strategy.
try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None
import multiprocessing
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path


METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT = Path(__file__).resolve().parents[3]

K = 2
N = 25
VERSION = "A"
TPM_FILE = f"N{N}{VERSION}.csv"
ESTADO_INICIAL = "1" + ("0" * (N - 1))
CONDICIONES = "1" * N
ALCANCE = "1" * N
MECANISMO = "1" * N

def limpiar_particion(obj):
    # NumPy integer -> int normal
    if isinstance(obj, np.integer):
        return int(obj)
    # Tuplas
    elif isinstance(obj, tuple):
        return tuple(
            limpiar_particion(x)
            for x in obj
        )
    # Listas
    elif isinstance(obj, list):
        return [
            limpiar_particion(x)
            for x in obj
        ]
    # Sets
    elif isinstance(obj, set):
        return {
            limpiar_particion(x)
            for x in obj
        }
    # Otros tipos
    return obj

def convertir_a_binario(texto, n_bits=20):
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)

def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm):
    try:
        analizador_fi = GeometricSIA(config_sistema)
        sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado_queue.put({
            "particion": sia_dos.particion,
            "perdida": str(sia_dos.perdida).replace('.', ','),
            "tiempo": str(sia_dos.tiempo_ejecucion).replace('.', ','),
        })

    except Exception as e:
        resultado_queue.put({
            "particion": None,
            "perdida": None,
            "tiempo": None,
        })

def ejecutar_k_geometric(
    config_sistema,
    condiciones,
    alcance,
    mecanismo,
    resultado_queue,
    tpm,
    k=3
):
    """
    Ejecuta la estrategia K-Geometric para pruebas
    experimentales.

    Esta función replica exactamente el flujo usado
    por GeoMIP clásico para facilitar comparación:

    - pérdida
    - tiempo
    - partición
    - estabilidad

    Parameters
    ----------
    config_sistema : Manager
        Configuración principal del sistema.

    condiciones : str
        Variables condicionadas.

    alcance : str
        Variables futuras.

    mecanismo : str
        Variables presentes.

    resultado_queue : multiprocessing.Queue
        Cola para retornar resultados.

    tpm : np.ndarray
        Matriz TPM del sistema.

    k : int
        Número de bloques deseados.
    """

    try:

        analizador = KGeometric(
            config_sistema
        )

        solucion = analizador.aplicar_estrategia(
            condiciones,
            alcance,
            mecanismo,
            tpm,
            k=k
        )

        resultado_queue.put({
            "particion": solucion.particion,
            "perdida": str(
                solucion.perdida
            ).replace('.', ','),

            "tiempo": str(
                solucion.tiempo_total
            ).replace('.', ','),

            "estrategia": f"K-GEOMETRIC k={k}"
        })

    except Exception as e:

        print(
            f"[ERROR K-GEOMETRIC] {str(e)}"
        )

        resultado_queue.put({
            "particion": None,
            "perdida": None,
            "tiempo": None,
            "estrategia": None
        })

def resolver_tpm_path(
    estado_inicio: str,
    archivo_tpm: str | None = None,
    version="A"
) -> Path:

    # Si el usuario pasa un archivo exacto
    if archivo_tpm is not None:

        possible_paths = (
            METHOD2_ROOT / "src" / ".samples" / archivo_tpm,
            METHOD2_ROOT / ".samples" / archivo_tpm,
            GEOMIP_ROOT / "data" / "samples" / archivo_tpm,
        )

        for path in possible_paths:
            if path.exists():

                print(f"\n===== USANDO TPM: {path} =====\n")

                return path

        raise FileNotFoundError(
            f"No se encontró el archivo TPM: {archivo_tpm}"
        )

    # Si no pasa archivo, usar lógica automática
    sample_name = f"N{len(estado_inicio)}{version}.csv"

    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT / "data" / "samples" / sample_name,
    )

    for candidate in candidates:

        if candidate.exists():

            print(f"\n===== USANDO TPM: {candidate} =====\n")

            return candidate

    raise FileNotFoundError(
        f"No se encontró la TPM '{sample_name}'"
    )


def inferir_estado_inicial() -> str:
    """Infer an initial state from available datasets (prefers largest NxA.csv)."""
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for sample_file in sample_dir.glob("N*.csv"):
            match = pattern.match(sample_file.name)
            if match:
                available_sizes.append(int(match.group(1)))

    if not available_sizes:
        raise FileNotFoundError("No hay archivos de muestras TPM disponibles en data/samples ni .samples.")

    n_bits = max(available_sizes)
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio=0,
    cantidad=50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
    k: int | None = None,
):
    df = pd.read_excel(ruta_excel, sheet_name=8, usecols="B", skiprows=3, names=["Subsistema"]) #! here
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    estado_inicio = estado_inicio or inferir_estado_inicial()
    condiciones = condiciones or ("1" * len(estado_inicio))
    tpm_path = resolver_tpm_path(
        estado_inicio,
        archivo_tpm=TPM_FILE
    )
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(partes[0][:len(partes[0]) - 3], n_bits=len(estado_inicio))
        mecanismus = convertir_a_binario(partes[1][:len(partes[1]) - 1], n_bits=len(estado_inicio))
        print(f"Iteración {i} - Alcance: {alcance}, Mecanismo: {mecanismus}")

        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        if k is not None:
            proceso = multiprocessing.Process(
                target=ejecutar_k_geometric,
                args=(
                    config_sistema,
                    condiciones,
                    alcance,
                    mecanismus,
                    resultado_queue,
                    tpm,
                    k
                )
            )
        else:
            proceso = multiprocessing.Process(
                target=ejecutar_con_tiempo,
                args=(
                    config_sistema,
                    condiciones,
                    alcance,
                    mecanismus,
                    resultado_queue,
                    tpm
                )
            )
        
        proceso.start()
        proceso.join(timeout=3600)  

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración": i,
            "Alcance": alcance,
            "Mecanismo": mecanismus,
            "Partición": resultado["particion"],
            "Pérdida": resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")

def iniciar():
    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "resultados_Geometric.xlsx"),
        )
    )
    ejecutar_desde_excel(ruta_entrada, ruta_salida)

def iniciar_k_geometric():
    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / f"resultados_K_Geometric_k{K}.xlsx"),
        )
    )
    ejecutar_desde_excel(ruta_entrada, ruta_salida, k=K)

def probar_geometric():
    print("\n===== GEOMETRIC =====\n")

    estado_inicial = ESTADO_INICIAL

    condiciones = CONDICIONES

    alcance = ALCANCE

    mecanismo = MECANISMO

    tpm_path = resolver_tpm_path(
        estado_inicial,
        archivo_tpm=TPM_FILE
    )

    print("TPM usada:")
    print(tpm_path)

    print("\nLeyendo TPM...")

    tpm = np.genfromtxt(
        tpm_path,
        delimiter=","
    )

    print("Creando gestor...")

    gestor = Manager(
        estado_inicial
    )

    print("Creando estrategia...")

    estrategia = GeometricSIA(
        gestor
    )

    print("Aplicando estrategia...")

    solucion = estrategia.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
        tpm
    )

    print("\n===== RESULTADO =====")

    print("Pérdida:")
    print(solucion.perdida)

    print("\nPartición:")
    print(solucion.particion)

    solucion.particion = limpiar_particion(
        solucion.particion
    )

    print("\nSolución completa:")
    print(solucion)

def probar_k_geometric():

    print("\n===== K-GEOMETRIC =====\n")

    estado_inicial = ESTADO_INICIAL

    condiciones = CONDICIONES

    alcance = ALCANCE

    mecanismo = MECANISMO

    tpm_path = resolver_tpm_path(
        estado_inicial,
        archivo_tpm=TPM_FILE
    )

    print("TPM usada:")
    print(tpm_path)

    print("\nLeyendo TPM...")

    tpm = np.genfromtxt(
        tpm_path,
        delimiter=","
    )

    print("Creando gestor...")

    gestor = Manager(
        estado_inicial
    )

    print("Creando estrategia...")

    estrategia = KGeometric(
        gestor
    )

    print("Aplicando estrategia...")

    solucion = estrategia.aplicar_estrategia(
        condicion=condiciones,
        alcance=alcance,
        mecanismo=mecanismo,
        tpm=tpm,
        k=K
    )

    print("\n===== RESULTADO =====")

    print("Pérdida:")
    print(solucion.perdida)

    print("\nPartición:")
    print(solucion.particion)

    solucion.particion = limpiar_particion(
        solucion.particion
    )

    print("\nSolución completa:")
    print(solucion)
