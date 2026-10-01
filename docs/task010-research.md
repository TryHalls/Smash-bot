# Task 010 — learned candidate scorer: Gate A research

Estado: **RESEARCH ONLY**. Este documento no implementa entrenamiento, no modifica el detector/tracker de producción y no usa HOLDOUT.

Fecha de auditoría: 2026-10-01. Rama de investigación: `task/010-learned-candidate-scorer`. Base: `a64655eb7d1a4c1fb9cd21aca6c206e304ed8551`.

## Decisión resumida

**Recomendación:** evaluar primero un scorer aprendido de **candidate nodes** (`recommended_learned_target = candidate`) con un patch CNN pequeño como opción principal de aprendizaje. Mantener el scorer clásico de OpenCV como control de bajo riesgo y dejar un detector full-frame como fallback, no como siguiente implementación.

La razón es específica del cuello de botella medido en Task 009:

- las propuestas `yellow_only` ya alcanzan 61/63 frames visibles DEV dentro de 20 px (96.83%);
- cuando la adquisición inicial es correcta, B_01 obtuvo 19/21 frames a 20 px con errores de pocos píxeles;
- la adquisición/ranking heurístico eligió objetos persistentes en A_01 y C_01;
- el tracker existente no necesita ser sustituido para probar esta hipótesis.

Un scorer de nodo aprende exactamente la decisión que falta: puntuar o rechazar cada propuesta amarilla y producir `NO OBSERVATION` cuando ninguna es suficientemente confiable. Puede reutilizarse en adquisición y reacquisition, mientras el tracker temporal existente sigue siendo responsable del seguimiento confirmado. Un scorer de pares o tracklets duplicaría inicialmente la función temporal que ya existe y aumentaría el riesgo de leakage entre frames adyacentes.

Esta recomendación es una decisión de investigación, no una autorización para implementar V2 ni para descargar un modelo.

## Evidencia de Task 009

### Dataset y protocolo

El snapshot humano contiene 136 records, 68 DEV y 68 HOLDOUT, con bursts completos y sin solapamiento entre splits. El HOLDOUT no se usó para selección de features, representación, reglas, debugging dirigido ni tuning; permanece sellado para una evaluación posterior.

El DEV activo consta de `A_01`, `B_01` y `C_01` (63 frames visibles), además de cinco negativos de estado. Los negativos son muestras aisladas: sirven para medir falsos positivos, pero no demuestran por sí solos un gate de estado temporal robusto.

### Resultados que motivan el scorer

| Medición DEV | Resultado | Lectura |
|---|---:|---|
| Yellow raw oracle @20 | 61/63 = 96.83% | La representación de propuestas es suficiente en la mayoría de frames |
| V1 single-hypothesis global @20 | 19/63 | El ranking/adquisición es el cuello de botella |
| V1 `A_01` @20 | 0/21 | Adquirió un distractor persistente |
| V1 `B_01` @20 | 19/21 | Control positivo: tracker y propuesta funcionan con adquisición correcta |
| V1 `C_01` @20 | 0/21 | Adquirió otro distractor persistente |
| Negativos FP | 0/5 | No es evidencia suficiente de un gate temporal de estado |
| Pair oracle raw | 58/60 ventanas | Hay hipótesis temporales correctas disponibles |
| Tracklet oracle raw, 3 frames | 55/57 ventanas | La temporalidad ayuda a diagnosticar, pero el beam heurístico falló |

La escalada heurística quedó cerrada por dos gates aceptados como FAIL. En la primera ventana correcta, las reglas de appearance geometry G1–G4 obtuvieron:

| Regla | A_01 | B_01 | C_01 |
|---|---:|---:|---:|
| G1 | 6 | 653 | 126 |
| G2 | 14 | 544 | 11 |
| G3 | 10 | 653 | 284 |
| G4 | 40 | 647 | 294 |

Ninguna mantuvo una hipótesis correcta en los tres bursts dentro de un beam de 32. No se propone otra regla manual.

### Coste conocido

La optimización ROI-local del hot path de Task 009 fue semánticamente equivalente en 68 frames DEV. El par aceptado por el equivalence gate fue:

| Timer | Antes mean / p50 / p95 / max (ms) | Después mean / p50 / p95 / max (ms) |
|---|---:|---:|
| `component_scoring_ms` | 766.147 / 741.156 / 1155.177 / 1371.271 | 8.166 / 6.610 / 16.128 / 20.901 |
| `total_algorithm_ms` | 892.194 / 884.761 / 1287.206 / — | 114.649 / 116.063 / 163.245 / — |

La familia diagnóstica de geometría no representa el runtime de producción: reconstruía muchas máscaras y tracklets deliberadamente. El scorer aprendido debe medirse en el pipeline normal, no extrapolar ese coste.

## Alternativas

### A1 — scorer clásico aprendido con OpenCV

**Diseño:** patch de propuesta a resolución nativa, descriptor compacto de gradientes/color/HOG y `cv2.ml.SVM` u otro modelo clásico disponible en OpenCV.

**Ventajas:** no añade dependencia de runtime; el entorno OpenCV ya está probado; entrenamiento CPU sencillo; modelo pequeño (normalmente KB–pocos MB); exportación y reproducibilidad simples.

**Riesgos:** las features manuales de Task 009 ya fueron insuficientes para separar objetos persistentes; HOG/color puede capturar más textura local, pero puede volver a fallar ante el mismo fondo/UI. Requiere validar que el descriptor preserve cabeza amarilla, cuerpo blanco, estela cyan y contexto sin esconder el objeto diminuto.

**Conclusión:** control obligatorio de bajo coste y buena línea base. No es la hipótesis principal de mayor capacidad discriminativa.

### A2 — patch CNN aprendido

**Diseño:** patch centrado en una propuesta amarilla, scorer binario o de ranking, entrenamiento separado y exportación a un formato de inferencia CPU.

**Ventajas:** aprende combinaciones de textura, color, forma, estela y contexto que las reglas manuales no separaron; trabaja sobre la representación de alta recuperación ya validada; el mismo nodo puede usarse en adquisición y reacquisition.

**Riesgos:** pocos scenes independientes, fuerte correlación temporal, desbalance candidato positivo/distractor y riesgo de memorizar A/B/C. Pretrained weights introducen licencia y descarga adicionales; entrenamiento scratch puede quedar corto sin ampliar TRAIN.

**Conclusión:** opción recomendada, empezando por una red pequeña y una comparación honesta contra A1. No se descarga PyTorch ni weights en Gate A.

Backbones a investigar en Gate C, sin elegir sólo por popularidad:

- tiny custom CNN: opción inicial más reproducible, pocos parámetros, entrenamiento scratch;
- MobileNet-style pequeño: buena eficiencia si la exportación y las depthwise convolutions funcionan en OpenCV DNN;
- ShuffleNet-style: alternativa de bajo coste, con la misma verificación de operadores;
- EfficientNet-lite/small: sólo si la mejora justifica su mayor complejidad y coste.

La primera prueba no necesita pretrained weights. Si se considera transferencia más adelante, se auditarán por separado el código y los weights.

### A3 — detector full-frame

**Ventajas:** aprende propuesta y clasificación conjuntamente y puede servir como fallback si la representación amarilla deja de ser suficiente fuera de DEV.

**Costes y riesgos:** requiere cajas o anotaciones equivalentes, no sólo centros; el shuttle es pequeño frente a 864×1920; aumenta datos, entrenamiento, runtime CPU y superficie de integración; reproduce trabajo que Task 009 ya resolvió parcialmente con propuestas yellow de 96.83% @20.

YOLOX tiene código Apache-2.0, pero el modelo/weights y cualquier dataset asociado requieren auditoría separada. Las alternativas de torchvision tienen código BSD-3-Clause, pero no eliminan el coste de datos, runtime ni licencia de weights. Se evita Ultralytics AGPL como dependencia por defecto cuando el problema no exige un detector full-frame.

### Comparación

| Alternativa | Poder discriminativo esperado | Datos | Runtime/dependencias | Coste de integración | Encaje con Task 009 |
|---|---|---|---|---|---|
| A1 OpenCV classical | Bajo–medio; interpretable, puede superar reglas manuales con textura local | Centros/candidatos; poco coste adicional | OpenCV existente; CPU muy bajo; modelo pequeño | Bajo | Control y fallback inmediato |
| A2 patch CNN | Medio–alto para la distinción local que falta | Candidatos etiquetados y scenes independientes; expansión TRAIN probable | PyTorch sólo para training; ONNX/OpenCV DNN para runtime; CPU a medir | Medio | Mejor encaje con el cuello de botella |
| A3 full detector | Alto potencial, pero aprende un problema más grande | Cajas, más variedad y mayor coste de anotación | Framework/modelo pesado; CPU y weights con riesgo | Alto | Fallback si A2 no generaliza |

No se usa una puntuación arbitraria 1–10: la selección se basa en el techo de propuestas medido, la separación entre adquisición y tracking y el presupuesto de CPU.

## Target aprendido recomendado

```yaml
recommended_learned_target: candidate
use_in: [acquisition, reacquisition]
pair_or_tracklet_model: deferred
tracker: existing
```

### Por qué no pair/tracklet primero

El target de nodo tiene labels directos a partir de la distancia al centro humano y puede evaluarse por burst. Un modelo de pair/tracklet requeriría duplicar la asociación temporal, manejar combinaciones que ya llegaron a cientos de miles de hipótesis y protegerse de forma más estricta contra leakage entre frames vecinos. La evidencia demuestra que existen propuestas temporales correctas, no que el modelo deba reemplazar el tracker.

El scorer de nodo puede entregar `NO OBSERVATION`; una hipótesis tentativa no cuenta como detection hasta que la semántica de adquisición existente la confirme. En tracking confirmado, el gate geométrico reduce el número de candidatos y el scorer puede actuar sólo como apariencia/tie-breaker o durante reacquisition.

## Crop y preprocesado propuestos

Configuración inicial, a congelar antes de una evaluación LOBO; no es una búsqueda de tamaños:

```yaml
source_frame: 864x1920 full-resolution
crop_physical_size: 96x96 pixels
model_input: 64x64 pixels
center: yellow candidate centroid, not ground-truth center
color: RGB
normalization: float32, [0, 1], then fixed mean/std recorded with model
border: reflect padding, clipped only after padding is defined
```

El patch físico 96×96 conserva la cabeza amarilla, un posible cuerpo blanco, la estela cyan próxima y suficiente cancha/UI local para distinguir un objeto persistente. 32×32 arriesga perder la señal del cuerpo; un patch mucho mayor hace que el shuttle sea demasiado pequeño y aumenta coste. 64×64 es el input inicial razonable, no una afirmación de que sea óptimo.

El crop se centra en el candidato amarillo producido por el detector. En producción nunca usa el centro GT. Cerca de los bordes, se aplica padding reflect de forma determinista antes de recortar; el modelo recibe siempre la misma geometría. El canal y normalización se guardan en la configuración del modelo exportado. No se introducen aún canales extra ni múltiples crops.

## Etiquetas de candidatos

Las etiquetas se generan offline después de producir candidatos yellow; el GT no entra en el generador ni en producción.

### Banda propuesta

Task 009 midió para yellow raw 57/63 frames a 5 px, 60/63 a 10 px y 61/63 a 20 px. Por ello la primera propuesta separa precisión de ambigüedad así:

```yaml
positive: candidate distance <= 10 px from visible GT center
ignore: 10 px < distance <= 30 px
negative: distance > 30 px, sólo en un frame con estado/label válido
```

El radio positivo de 10 px conserva la mayoría de propuestas claramente correctas sin convertir la banda 10–20, donde ya hay incertidumbre de candidate identity, en negativos duros. El límite de 30 px es una banda de exclusión explícita, no un threshold de producción ni un resultado de tuning.

Reglas adicionales:

- Si hay varios candidatos a ≤10 px, el más cercano es positivo; los demás hasta 30 px son `ignore`, no negativos. Si dos empatan, se conserva el orden determinista del candidato y se registra la ambigüedad.
- Si sólo hay candidato a 10–20 px, no se inventa un positivo: queda `ignore` y el frame cuenta como limitación de propuesta para el análisis.
- Un candidato >30 px puede ser negativo sólo si el frame tiene una anotación válida que permite esa interpretación. Los candidatos dudosos no se convierten silenciosamente en negativos.
- `visible=false`: no hay positivo; las propuestas >30 px pueden formar negativos si el estado es válido, y las cercanas quedan `ignore` para no castigar una etiqueta de oclusión dudosa.
- `visible=true, occluded=true`: se conserva el centro si existe y se etiqueta por distancia; el flag de oclusión se conserva como metadata y puede servir para análisis estratificado.
- `visible=false, occluded=true` o `ambiguous=true`: no se crea positivo; candidatos no inequívocos quedan `ignore` salvo negatives claramente separados y válidos.
- Frames sin propuesta ≤20 px se registran como `proposal_miss`; no se sintetiza un crop positivo.

El scorer debe entrenarse con pesos o muestreo documentados para que los miles de distractores no dominen la pérdida. No se seleccionará el balance mirando HOLDOUT.

## Leakage y validación

El protocolo primario es leave-one-burst-out sobre DEV:

```text
fold A: train B_01 + C_01, validate A_01
fold B: train A_01 + C_01, validate B_01
fold C: train A_01 + B_01, validate C_01
```

Los bursts son unidades indivisibles. Nunca se separan frames adyacentes del mismo burst entre train y validation. Los candidatos negativos del mismo frame permanecen en el mismo fold que ese frame; no se mezclan copias ni crops derivados entre folds. Los negativos de estado se asignan por su record y se mantienen en el protocolo predeclarado.

La selección de arquitectura, crop, label band, normalización, augmentations, ranking y modelo debe basarse en el resumen LOBO y no en un frame individual. Sólo después de congelar todo se autoriza una única evaluación HOLDOUT de A_02, B_02, C_02 y negativos pares. Hasta ese punto el HOLDOUT no se inspecciona.

## Suficiencia de datos y expansión TRAIN

El DEV actual es suficiente para una prueba de factibilidad y un smoke test de leakage, pero no para afirmar generalización: sólo hay tres scenes activas independientes. Hay 63 frames visibles y el diagnóstico yellow-only produjo 61/63 propuestas a ≤20; el conjunto de candidatos contiene muchos distractores correlacionados temporalmente. La cantidad de frames no equivale a la cantidad de escenas.

Propuesta de expansión TRAIN, usando sólo captures Task 008 existentes y excluyendo todas las ventanas congeladas DEV/HOLDOUT:

```yaml
source: A/B/C accepted captures plus permitted negative-state material
sampling: deterministic every 8th decoded frame, with fixed run/frame exclusions
target: roughly 240–360 additional frames
scenes: multiple temporal regions per clip, not one contiguous burst
labels: human confirmation required for shuttle center/state
role: TRAIN only; never changes frozen DEV/HOLDOUT records
```

La etiqueta automática por proximidad a un candidato puede proponer ejemplos, pero no debe convertir frames sin candidato, oclusiones ni UI en negativos sin revisión humana. El tamaño es un plan inicial, no una etiqueta ya creada ni una autorización de captura nueva.

## Auditoría de dependencias y licencias

No se instaló ni descargó ningún paquete o weight en este gate. El entorno local existente sólo tiene:

```text
Python 3.11.2
numpy 2.4.6
opencv-python-headless 4.14.0.94 / cv2 4.14.0
```

No están instalados `torch`, `torchvision`, `onnx`, `onnxruntime` ni `scikit-learn`. La siguiente tabla da candidatos y estimaciones de planificación; los tamaños no son artefacts descargados ni sustituyen una medición antes de instalar.

| Componente | Candidato de investigación | Código/licencia a auditar | Compatibilidad CPU/Python | Estimación de disco antes de instalar | Decisión Gate A |
|---|---|---|---|---:|---|
| OpenCV | existente 4.14.0.94 | Apache-2.0; NumPy separado | probado en Python 3.11; CPU | ya instalado | mantener |
| PyTorch | versión CPU compatible a fijar en Gate C; pareja candidata 2.7.x | BSD-3-Clause para proyecto; dependencias bundled con licencias propias | CPU y Python 3.11 disponibles según wheel elegido | aproximadamente 250–500 MiB wheel/paquetes; 0.8–1.5 GiB instalado, sujeto a plataforma | no instalar ahora |
| torchvision | pareja compatible con torch, candidato 0.22.x | BSD-3-Clause; revisar cada weight/model card | CPU/Python 3.11 si la pareja exacta existe | aproximadamente 10–100 MiB, según wheel | no instalar ahora |
| ONNX | versión compatible con exporter a fijar | Apache-2.0 | Python 3.11; CPU | aproximadamente 20–60 MiB | sólo si Gate C lo necesita |
| ONNX Runtime CPU | versión a fijar tras export test | MIT para runtime; revisar dependencias | CPU/Python 3.11 | aproximadamente 15–40 MiB wheel, mayor instalado | preferir OpenCV DNN si basta |
| scikit-learn | 1.7.x como referencia, no requisito | BSD-3-Clause; requiere NumPy/SciPy y revisar sus licencias | Python 3.11/CPU | aproximadamente 100–300 MiB con dependencias | no necesario para A1 |
| Full detector | YOLOX u opción BSD/permisiva equivalente | código y weights se auditan por separado; no asumir licencia de weights | CPU incierto y costoso | cientos de MiB a GiB según stack/weights | fallback solamente |

La estimación PyTorch es incompatible con el margen actual si se instala sin una limpieza o autorización adicional: el preflight dejó aproximadamente 1.56 GB libres y el hard stop del proyecto es 1.10 GiB. No se debe ejecutar una instalación especulativa. PyTorch documenta requisitos de Python y builds CPU; Torchvision publica su compatibilidad y licencia; ONNX Runtime documenta el paquete CPU y la API de inferencia; scikit-learn documenta su árbol de dependencias. Referencias oficiales consultadas:

- [PyTorch install/local](https://pytorch.org/get-started/locally/) y [PyTorch source/license](https://github.com/pytorch/pytorch/blob/main/LICENSE);
- [Torchvision](https://github.com/pytorch/vision);
- [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html);
- [scikit-learn install](https://scikit-learn.org/stable/install);
- [OpenCV license](https://opencv.org/license/);
- [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX).

Debe distinguirse siempre:

1. licencia del código de entrenamiento;
2. licencia de los weights/pretrained backbone;
3. licencia/provenance del dataset y labels internos;
4. licencia del runtime de producción.

Para este proyecto, los labels Task 009 son datos internos con provenance del snapshot; no se añade una licencia externa. Ningún pretrained weight queda elegido en Gate A.

## Separación training/runtime

La arquitectura preferida es:

```text
training-only environment (PyTorch or smaller approved trainer)
    -> frozen export (ONNX)
    -> operator compatibility check
    -> production runtime: NumPy + OpenCV DNN CPU
```

El entrenamiento no entra en el runtime de Smash-bot. OpenCV DNN debe probar el grafo exportado concreto, no sólo el nombre del backbone. Si una depthwise convolution, resize, normalización u operador no es compatible, se debe cambiar a un tiny CNN con operadores básicos o evaluar ONNX Runtime CPU como dependencia explícita. No se acepta una conversión silenciosa ni una discrepancia de preprocesado.

Un modelo pretrained no es automáticamente preferible: con pocos scenes puede ayudar, pero introduce un archivo grande, una licencia de weights y posibles sesgos de dominio. La primera comparación debe incluir un tiny CNN scratch y A1; la transferencia se autoriza sólo si la expansión TRAIN y la auditoría de weights lo justifican.

## Presupuesto de CPU

El gate de Task 009 permanece en algorithm p95 ≤33 ms/frame y ≥30 FPS. No todo ese presupuesto pertenece al scorer.

Propuesta inicial de presupuesto, a verificar con medición, no garantía:

```yaml
normal_confirmed_tracking: scorer amortizado <=2–3 ms/frame p95
acquisition/reacquisition scorer: <=8 ms p95 por frame de búsqueda
remaining budget: registration, masks, proposals, association, tracker and I/O
```

Con hasta ~60 propuestas por frame no es aceptable ejecutar una CNN pesada de forma serial sin medir. La implementación futura debe evaluar inferencia NCHW batched para los crops, y limitar el scorer a acquisition/reacquisition cuando el gate geométrico del tracking confirmado ya reduce candidatos. Una preselección barata sólo es válida si se demuestra que no reduce el techo de propuestas; no se fija ahora ningún cap nuevo.

El informe de Gate C debe incluir p50/p95/max del scorer solo, número de propuestas, coste de batch y coste total del algoritmo en el mismo host. Si el scorer no puede respetar el presupuesto con un modelo pequeño, el resultado será FAIL de runtime aunque la métrica semántica sea buena.

## Plan por gates

### Gate B — candidate dataset tooling

**Objetivo:** extraer patches reproducibles de candidates yellow y construir labels con la banda positiva/ignore/negative sin tocar producción ni HOLDOUT.

**PASS:** identidades, PTS, crop geometry y labels son deterministas; bursts y folds no tienen leakage. **FAIL:** no se puede reconstruir una muestra o la banda produce ambigüedad no registrada. **Artifacts:** manifest compacto, estadísticas de clase y tests; no vídeo ni PNG masivos.

### Gate C — feasibility de scorer

**Objetivo:** comparar A1 contra un tiny CNN (y sólo las variantes predeclaradas) en CPU, usando LOBO DEV.

**PASS:** el modelo supera de forma reproducible el control en ranking/first-acquisition y no rompe el contrato `NO OBSERVATION`; además exporta y ejecuta con el preprocesado idéntico. **FAIL:** no separa distractores, no exporta, o excede claramente el presupuesto. **INCONCLUSIVE:** datos insuficientes o folds con ausencia de positivos. No se consulta HOLDOUT.

### Gate D — freeze LOBO

**Objetivo:** fijar arquitectura, crop, label band, normalización, entrenamiento, export, ranking y estado de adquisición a partir de los folds DEV.

**PASS:** configuración reproducible congelada y model artifact pequeño con provenance. **FAIL:** selección depende de un burst o de ajustes post-hoc.

### Gate E — evaluación única HOLDOUT

**Objetivo:** ejecutar una sola medición sobre A_02, B_02, C_02 y negativos pares después del freeze.

**PASS/FAIL:** se aplican los gates de Task 009 sin reajuste. Si la evidencia no permite una conclusión, es INCONCLUSIVE y se conserva el modelo; no se reabre tuning sobre HOLDOUT.

### Gate F — integración de adquisición

**Objetivo:** conectar el scorer sólo en acquisition/reacquisition, conservar tracker y distinguir observación de predicción.

**PASS:** contrato de PTS, `NO OBSERVATION`, confirmación temporal, latencia y errores regresan correctamente. **FAIL:** cualquier predicción se cuenta como detection o se altera el tracker sin evidencia.

### Gate G — optimización/fallback

**Objetivo:** medir runtime de producción y decidir si hace falta optimización o un detector full-frame fallback.

**HARD STOP:** no iniciar A3 por defecto; sólo investigar full-frame si A2 no alcanza representación/generalización o si los candidatos dejan de cubrir el dominio.

## Hechos, resultados y riesgos

### FACT

- Task 009 tiene 136 labels, con DEV/HOLDOUT congelados.
- Yellow proposals alcanzan 96.83% @20 en DEV.
- El tracker funcionó cuando B_01 adquirió correctamente.
- No hay PyTorch, Torchvision, ONNX, ONNX Runtime ni scikit-learn instalados en el entorno auditado.
- Gate A no instaló paquetes, descargó weights, accedió al teléfono ni inspeccionó HOLDOUT.

### MEASURED RESULT

- El ranking single-hypothesis corrigió B_01 parcialmente, pero falló A_01/C_01.
- Los gates temporal y appearance-geometry no encontraron una regla ≤32 en las primeras ventanas A/B/C.
- La propuesta yellow es suficientemente rica para justificar aprender el ranking antes de ampliar a detector full-frame.

### ARCHITECTURE DECISION

- `candidate` es el primer target aprendido.
- A1 es control; A2 patch CNN tiny es la ruta principal; A3 es fallback.
- Training y runtime se separan; el runtime preferido es OpenCV DNN CPU si el grafo exportado es compatible.
- El tracker actual se conserva.

### OPEN RISK

- Tres scenes activas DEV no bastan para afirmar generalización.
- Los candidatos positivos están fuertemente correlacionados temporalmente y desbalanceados frente a distractores.
- El presupuesto de CPU puede impedir puntuar ~60 crops por frame; batching y uso limitado a acquisition deben medirse.
- Pretrained weights pueden tener licencia o coste de disco incompatibles.
- El scorer puede aprender el estilo de los tres bursts y fallar HOLDOUT; por eso LOBO y el HOLDOUT sellado son obligatorios.

## Resultado del Gate A

```yaml
status: PASS_RESEARCH_ONLY
production_code_changed: false
packages_installed: false
weights_downloaded: false
holdout_used: false
phone_accessed: false
recommended_learned_target: candidate
recommended_path: A2_patch_cnn_with_A1_opencv_control
full_detector: fallback_only
next_action: Gate B dataset/candidate extraction tooling
```

Gate A termina aquí. No se implementa aún Task 010, no se modifica `pyproject.toml`, no se instala PyTorch y no se descarga ningún modelo.
