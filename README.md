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
