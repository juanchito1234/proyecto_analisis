# Proyecto-20261

Este repositorio contiene tres implementaciones principales para el analisis de MIP/IIT:

1. `QNodes` (base clasica, antes referida como Proyecto-2025A)
2. `GeoMIP/src/Method2_Dynamic_Programming_Reformulation`

## Requisitos

- Linux (probado en Ubuntu) o Windows
- Python 3.11+ (hay entornos locales con 3.12 y 3.14)
- `uv` instalado

Instalacion de `uv` (si no lo tienes):

```bash
pip install uv
```

## Estructura Rapida

- `QNodes/`: ejecucion directa de un caso de prueba (`exec.py`).
- `GeoMIP/src/Method1_GPU_Accelerated/`: procesamiento por lotes desde Excel.
- `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`: procesamiento por lotes desde Excel.
- `GeoMIP/data/samples/`: datasets TPM `N*.csv` usados por Method1/Method2.
- `GeoMIP/results/`: archivos Excel de entrada/salida para Method1/Method2.

## Distribución de Carpetas y Archivos

A continuación se muestra la estructura jerárquica completa del proyecto, con anotaciones sobre el propósito de cada archivo y carpeta relevante.

```
proyecto_analisis/
│
├── README.md                          # Este archivo – guía general del proyecto
├── Proyecto_KQMIP.docx                # Documento del proyecto académico
│
├── QNodes/                            # ── Implementación del algoritmo QNodes ──
│   ├── pyproject.toml                 # Configuración del proyecto y dependencias (uv)
│   ├── q_nodes.py                     # Punto de entrada: ejecuta la estrategia QNodes
│   ├── force.py                       # Punto de entrada: ejecuta la estrategia BruteForce
│   │
│   ├── src/                           # Código fuente principal
│   │   ├── main.py                    # Orquestador principal (QNodes)
│   │   ├── main_force.py              # Orquestador principal (BruteForce)
│   │   │
│   │   ├── constants/                 # Constantes globales
│   │   │   ├── base.py                # Índices, etiquetas y valores por defecto
│   │   │   ├── error.py               # Mensajes de error
│   │   │   └── models.py              # Tags y etiquetas de modelos
│   │   │
│   │   ├── funcs/                     # Funciones utilitarias puras
│   │   │   ├── iit.py                 # Cálculo de EMD y lógica IIT
│   │   │   ├── format.py              # Formateo visual de particiones
│   │   │   └── force.py               # Generación exhaustiva de biparticiones
│   │   │
│   │   ├── middlewares/               # Servicios transversales
│   │   │   ├── profile.py             # Perfilado de rendimiento (pyinstrument)
│   │   │   └── slogger.py             # Logger seguro con colores y niveles
│   │   │
│   │   ├── models/                    # Modelos del dominio
│   │   │   ├── base/
│   │   │   │   ├── sia.py             # Clase base SIA (Sistema de Información Activo)
│   │   │   │   └── application.py     # Configuración global (notación, distancia)
│   │   │   ├── core/
│   │   │   │   ├── system.py          # System: motor de estados, bipartir(), k_partir()
│   │   │   │   ├── ncube.py           # NCube: tensor probabilístico N-dimensional
│   │   │   │   └── solution.py        # Solution: encapsula resultado (φ, partición)
│   │   │   └── enums/
│   │   │       ├── distance.py        # Enum de métricas de distancia
│   │   │       ├── notation.py        # Enum de notación (little/big-endian)
│   │   │       └── temporal_emd.py    # Enum de modos temporales EMD
│   │   │
│   │   ├── strategies/                # Estrategias de búsqueda de particiones
│   │   │   ├── q_nodes.py             # QNodes: algoritmo Greedy submodular (k-particiones)
│   │   │   ├── force.py               # BruteForce: búsqueda exhaustiva
│   │   │   └── phi.py                 # Phi: cálculo directo de φ
│   │   │
│   │   └── .samples/                  # Matrices TPM de prueba (CSV)
│   │
│   └── tests/                         # Pruebas unitarias (pytest)
│       └── test_qnodes.py             # Tests de QNodes para k=2, k=3, k=4
│
├── GeoMIP/                            # ── Implementación del algoritmo GeoMIP ──
│   ├── Dataset_Description.md         # Descripción de los datasets utilizados
│   │
│   ├── data/
│   │   ├── creation.py                # Script de generación de TPMs sintéticas
│   │   └── samples/                   # Datasets TPM en formato CSV
│   │       ├── N3A.csv ... N3B.csv    # Redes de 3 nodos
│   │       ├── N4A.csv ... N4C.csv    # Redes de 4 nodos
│   │       ├── N5A.csv, N5B.csv       # Redes de 5 nodos
│   │       ├── N6A.csv                # Red de 6 nodos
│   │       ├── N8A.csv                # Red de 8 nodos
│   │       ├── N10A.csv               # Red de 10 nodos
│   │       └── N15A.csv, N15B.csv     # Redes de 15 nodos
│   │
│   ├── results/                       # Archivos de entrada/salida Excel
│   │   ├── Pruebas_Metodo2.xlsx       # Entrada: configuraciones de prueba
│   │   ├── pruebas_Metodo1.xlsx       # Entrada: configuraciones Método 1
│   │   └── resultados_Geometric.xlsx  # Salida: resultados de ejecuciones
│   │
│   └── src/
│       └── Method2_Dynamic_Programming_Reformulation/
│           ├── pyproject.toml         # Configuración del proyecto y dependencias (uv)
│           ├── exec.py                # Punto de entrada: procesamiento por lotes
│           │
│           └── src/                   # Código fuente principal
│               ├── main.py            # Orquestador principal (carga Excel, ejecuta)
│               │
│               ├── constants/         # Constantes globales
│               │   ├── base.py        # Índices, etiquetas y valores por defecto
│               │   ├── error.py       # Mensajes de error
│               │   └── models.py      # Tags y etiquetas de modelos
│               │
│               ├── funcs/             # Funciones utilitarias puras
│               │   ├── base.py        # Funciones base (reindexar, etc.)
│               │   ├── format.py      # Formateo visual de particiones
│               │   └── system.py      # Funciones auxiliares del sistema
│               │
│               ├── middlewares/       # Servicios transversales
│               │   ├── profile.py     # Perfilado de rendimiento
│               │   └── slogger.py     # Logger seguro con colores
│               │
│               ├── controllers/       # Controladores y estrategias
│               │   ├── manager.py     # Manager: carga de datos desde Excel/CSV
│               │   └── strategies/
│               │       ├── geometric.py      # GeoMIP original (biparticiones)
│               │       ├── k_geometric.py    # KGeoMIP extendido (k-particiones)
│               │       ├── k_brute_force.py  # KBruteForce (fuerza bruta k-particiones)
│               │       ├── force.py          # BruteForce original (biparticiones)
│               │       ├── q_nodes.py        # QNodes adaptado para GeoMIP
│               │       └── phi.py            # Cálculo directo de φ
│               │
│               ├── models/            # Modelos del dominio
│               │   ├── base/
│               │   │   ├── sia.py             # Clase base SIA
│               │   │   └── application.py     # Configuración global
│               │   ├── core/
│               │   │   ├── system.py          # System: motor de estados y k-particiones
│               │   │   ├── ncube.py           # NCube: tensor probabilístico
│               │   │   └── solution.py        # Solution: resultado (φ, partición)
│               │   ├── enums/
│               │   │   ├── distance.py        # Enum de métricas de distancia
│               │   │   └── notation.py        # Enum de notación
│               │   ├── geometry/
│               │   │   └── transition_geometry.py  # Geometría de transiciones (Hamming)
│               │   └── partitions/
│               │       ├── k_partition_generator.py  # Generador de k-particiones (Stirling)
│               │       └── partition_evaluator.py    # Evaluador EMD con caché
│               │
│               └── tests/             # Pruebas unitarias (pytest)
│                   └── geometry/
│                       ├── test_k_geometric.py          # Test de integración KGeoMIP
│                       ├── test_partition_generator.py   # Tests del generador de particiones
│                       ├── test_partition_evaluator.py   # Tests del evaluador EMD
│                       └── test_transition_geometry.py   # Tests de la geometría de transición
│
└── docs/                              # Documentación adicional del proyecto
```

## 1) Ejecutar QNodes

### Dependencias

Desde `QNodes/`:

```bash
cd QNodes
uv sync
```

### Ejecucion

```bash
uv run q_nodes.py
uv run force.py
```

### Que hace

- Carga una red desde `QNodes/src/.samples/` (segun el estado inicial y pagina configurada).
- Ejecuta estrategia `BruteForce` desde `QNodes/src/main.py`.
- Imprime la solucion en consola.

### Ajustes comunes

Edita `QNodes/src/main.py`:

- `estado_inicial`
- `condiciones`
- `alcance`
- `mecanismo`

Si termina muy rapido, no necesariamente es error: puede ser un caso pequeno o corte temprano cuando `phi = 0`.

## 3) Ejecutar Method2_Dynamic_Programming_Reformulation

### Dependencias

Desde `GeoMIP/src/Method2_Dynamic_Programming_Reformulation/`:

```bash
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv sync
```

### Ejecucion

```bash
uv run exec.py
```

### Entrada por defecto

- Excel entrada: `GeoMIP/results/Pruebas_Metodo2.xlsx`
- Hoja usada actualmente: indice `8`
- Columna subsistema: `B`

### Salida por defecto

- Excel salida: `GeoMIP/results/resultados_Geometric.xlsx`

---

## Generación de Matrices TPM $2^N \times N$ para Escalabilidad

El repositorio incluye un script centralizado (`generador_escalabilidad.py`) en la raíz del proyecto para crear matrices de probabilidad de transición (TPM) de tamaño $2^N \times N$ para cualquier tamaño de red $N$. 

Este generador guarda los archivos `.csv` automáticamente en las carpetas de muestras de ambos subproyectos (`QNodes/src/.samples/` y `GeoMIP/data/samples/`), utilizando nombres estructurados que los cargadores automáticos del sistema reconocen de manera nativa (ej: `N16A.csv`).

### Ejecución del Generador

Para ejecutar el generador desde la raíz del proyecto, puedes utilizar `uv` directamente indicando el número de nodos y una versión identificadora:

```bash
# Generar una matriz de 16 nodos (2^16 x 16 estados) con versión 'A' (por defecto)
uv run generador_escalabilidad.py --nodos 16 --version A

# Generar una matriz de 20 nodos (2^20 x 20 estados) con versión 'B'
uv run generador_escalabilidad.py --nodos 20 --version B
```

El script imprimirá las dimensiones, la memoria RAM estimada y confirmará la creación exitosa del archivo delimitado por comas (`,`) en ambas rutas correspondientes.

### Cómo Configurar el Proyecto para Usar la Nueva Matriz

Una vez generada la matriz:

#### En QNodes (`QNodes/src/main.py` o `QNodes/exec.py`):
Modifica las variables de configuración en la parte superior del archivo:
```python
N = 16  # El número de nodos generado
VERSION = "A"  # El identificador de versión que usaste
TPM_FILE = f"N{N}{VERSION}.csv"
```

#### En GeoMIP (`GeoMIP/src/Method2_Dynamic_Programming_Reformulation/src/main.py`):
Modifica las variables en la sección de configuración global:
```python
N = 16  # El número de nodos generado
VERSION = "A"  # El identificador de versión que usaste
TPM_FILE = f"N{N}{VERSION}.csv"
```

---

# Pruebas Unitarias y Cobertura (Testing Suite)

Esta sección está destinada a estudiantes y evaluadores académicos que requieran validar matemáticamente el correcto funcionamiento de las extensiones de `K-Particiones` para `GeoMIP` y `QNodes`.

## Requisitos Previos

Asegúrate de haber instalado y activado las dependencias usando `uv` (el cual incluye `pytest` y `pytest-cov` instalados como dependencias de desarrollo).

## Ejecución Rápida para Evaluación Académica

Si eres evaluador y deseas ejecutar **todas** las pruebas rápidamente y verificar que el software es robusto:

### En Windows (PowerShell)
Para GeoMIP:
```powershell
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/ -v
```

Para QNodes:
```powershell
cd QNodes
$env:PYTHONPATH="."; uv run pytest tests/ -v
```

### En Linux/macOS (Bash)
Para GeoMIP:
```bash
cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
PYTHONPATH="." uv run pytest src/tests/geometry/ -v
```

Si el proyecto cumple con los requisitos, observarás que todas las líneas resultan en un texto verde que dice **`PASSED`**.

## Interpretación de Resultados

- **`PASSED` (Verde)**: La funcionalidad cumple los criterios matemáticos y de software.
- **`FAILED` (Rojo)**: La lógica falló (por ejemplo, los resultados no fueron los esperados matemáticamente).
- **`ERROR` (Amarillo/Rojo)**: Hubo un fallo en la ejecución o configuración (ej. un archivo falta, error de sintaxis).

## Ejecución Individual de Pruebas

Puedes probar componentes aislados apuntando directamente al archivo:

**Generador de Particiones (GeoMIP)**

cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation

```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/test_partition_generator.py -v
```

**Evaluador EMD y Caché (GeoMIP)**
```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/test_partition_evaluator.py -v
```

**Topología de Transición (GeoMIP)**
```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/test_transition_geometry.py -v
```

**Integración de la Estrategia (GeoMIP)**
```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/test_k_geometric.py -v
```

**Estrategia Greedy (QNodes)**

cd QNodes

```bash
$env:PYTHONPATH="."; uv run pytest tests/test_qnodes.py -v
```

## Generación de Cobertura de Código

Para visualizar exactamente qué porcentaje y qué líneas de código de `src/` están cubiertas por las pruebas automatizadas, se utiliza `pytest-cov`.

cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation

**1. Reporte Rápido en Consola:**
```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/ --cov=src
```
Te mostrará una tabla con el porcentaje `Cover` en la última columna.

**2. Reporte Interactivo en HTML:**
```bash
$env:PYTHONPATH="."; uv run pytest src/tests/geometry/ --cov=src --cov-report=html
```
Esto creará una carpeta llamada `htmlcov/`. Puedes abrir el archivo `htmlcov/index.html` en cualquier navegador web para revisar línea por línea el código analizado.

## Verificación de Requisitos Académicos

| Requisito del Proyecto | Prueba que lo valida |
| ---------------------- | -------------------- |
| **Extender GeoMIP k-particiones (2 ≤ k ≤ 5)** | `test_k_geometric.py`, `test_partition_generator.py` |
| **Extender QNodes k-particiones (2 ≤ k ≤ 5)** | `test_qnodes.py::test_qnodes_k3` y `k4` |
| **Reutilizar infraestructura y N-Cubos** | `test_transition_geometry.py`, `test_partition_evaluator.py` |
| **Evaluar y generar candidatos** | `test_partition_evaluator.py`, `test_partition_generator.py` |
| **Consistencia con comportamiento para k=2** | `test_generador_k2`, `test_qnodes_k2` |
| **Pruebas unitarias para validar componentes** | _(Todos los scripts anteriores)_ |
| **Productos tensoriales para k-particiones** | Cubierto implícitamente en `System.distribucion_marginal()`, validado matemáticamente en `test_evaluacion_particion_valida` dentro del `PartitionEvaluator` ya que procesa y une los resultados tensoriales internamente para el cálculo de distancias (EMD). |
