# Task 011 — Stateful sparse shuttle perception feasibility

## Conclusión

`LOCAL_TRACK_RUNTIME_FEASIBLE__SYNC_CANDIDATE_PROPOSAL_ACQUISITION_FAIL`

Task 011 cierra la investigación de una cascada síncrona basada en propuestas
amarillas y un scorer por candidato. El estado y el tracking local siguen
siendo reutilizables; la adquisición/re-adquisición síncrona no satisface a la
vez cobertura y coste.

Todos los resultados son DEV-only. `HOLDOUT used=false` en todos los gates.
No existe un modelo de percepción de producción ni se hizo integración en
producción.

## Protocolo y gates congelados

La evaluación respetó los gates heredados:

- recall@20 >= 0.90;
- recall@10 >= 0.80;
- localization p50 <= 10 px;
- localization p95 <= 20 px;
- longest miss <= 2 frames;
- reacquisition <= 2 frames y <= 70 ms;
- algorithm p95 <= 33 ms/frame;
- >= 30 FPS.

El dataset fue A_01/B_01/C_01, con PTS del dispositivo. Ground truth sólo se
usó después de generar propuestas, como evaluator. No se cargó ni inspeccionó
HOLDOUT para tomar decisiones.

## Gate A — inventario y decisión inicial

El punto de partida combinaba un V1 full-frame costoso con un tracker temporal.
El V1 tenía algorithm p95 de `218.500 ms` y aproximadamente `6.75 FPS`.
La población raw yellow era de `4064` candidatos en 63 frames; la distribución
full-frame era media `64.51`, p50 `60`, p95 `94`, máximo `94` candidatos/frame.

La evaluación centrada en la predicción mostró que una ROI de 120 px podía
mantener cardinalidad baja (raw: p50 `5`, p95 `9.05`) y supervivencia de
`58/60` positivos en la medición correspondiente. El resultado fue recomendar
una cascada explícita `ACQUIRE/TENTATIVE/TRACK/COAST/REACQUIRE`, manteniendo
observación y predicción separadas y usando PTS.

Task 010 aportaba un scorer semánticamente prometedor, pero su coste full
candidate no justificaba por sí solo la adquisición síncrona.

## Gate B — cascada síncrona

La primera ejecución pasó la equivalencia full-frame (`63` frames, `4064`
candidatos), pero falló la equivalencia local exacta en A_01 frame 80: la
cardinalidad coincidía, aunque un centroide difería aproximadamente
`2.3e-13`. El resultado fue `STOP_LOCAL_PROPOSAL_EQUIVALENCE`.

Gate B-R corrigió el protocolo de comparación, la selección por logit y la
contabilidad de llamadas local/full. El replay de los tres folds mantuvo los
hashes aceptados de parámetros y la paridad OpenCV DNN. La semántica integrada
final no pasó:

- recall@20 global: `49/63 = 77.78%`;
- A_01: `19/21`, B_01: `18/21`, C_01: `12/21`;
- localization p50 `2.126 px`, p95 `6.437 px`;
- longest miss burst `4` frames;
- reacquisition máxima `2` frames y `33.775 ms`;
- confirmed negative FP: `0/5`;
- stale accepted: `0`.

El runtime integrado tuvo p95 total `116.949 ms` y `26.83 FPS`, con 17 llamadas
full y 49 locales. La propuesta p95 fue `50.238 ms` y DNN p95 `57.516 ms`.
Por tanto falló tanto la semántica global como el presupuesto integrado.

Evidence: `data/task011/gate_b_failure.json` y
`data/task011/gate_b_r2_summary.json`.

## Gate C — atribución y suelo yellow-only

Se implementó un path diagnóstico yellow-only real: BGR→HSV, thresholds
congelados, exclusión HUD, `MORPH_OPEN 3x3`, connected components y ordering
determinista. La equivalencia full-frame fue PASS: `63` frames y `4064`
candidatos, sin ground truth en la generación.

La atribución DEV global de los fallos fue:

| Causa primaria | Count |
|---|---:|
| HIGHER_LOGIT_WRONG_SELECTION | 1 |
| INITIAL_TENTATIVE_NO_OBSERVATION | 5 |
| NO_FULL_RAW_POSITIVE | 2 |
| POSITIVE_LOCAL_LOGIT_NONPOSITIVE | 3 |
| POSITIVE_OUTSIDE_LOCAL_RADIUS | 1 |
| WRONG_OBSERVATION_GT_ERROR_GT20 | 1 |
| OTHER | 1 |

El scorer tenía `56/60` positivos en top-1 y `59/60` en top-8 como diagnóstico
de ranking, pero eso no se tradujo en una adquisición/confirmación robusta.

El suelo computacional yellow-only, sin scorer ni tracker, fue:

- full proposal directo, OpenCV threads=1: p95 `28.453 ms`;
- local proposal directo: p95 `5.688 ms`;
- candidato/frame del benchmark: media `17.32`, p50 `5`, p95 `61.8`, máximo `89`.

La decisión fue `NEED_BOUNDED_ACQUISITION_SHORTLIST`, no un cambio de
thresholds, radios ni política.

Evidence: `data/task011/gate_c_summary.json`.

## Gate D — shortlist de estadísticas

Se probó únicamente el MLP congelado `6→8→1` bajo LOBO, para K=4 y K=8.
No hubo búsqueda de parámetros.

| Shortlist | Retención global | A_01 | B_01 | C_01 | pares iniciales |
|---|---:|---:|---:|---:|---|
| K4 | 7/60 | 2/20 | 5/21 | 0/19 | A/C fail, B pass |
| K8 | 20/60 | 4/20 | 7/21 | 9/19 | A/C fail, B pass |

La determinación de los dos entrenamientos por fold fue PASS, pero ninguna
shortlist conservó la semántica requerida. La línea de shortlist estadística
queda cerrada; no se convirtió en política de producción.

Evidence: `data/task011/gate_d_summary.json`.

## Gate E y E-R — coarse-to-fine half-resolution

El primer Gate E se detuvo proceduralmente antes del replay porque no había
pesos semánticos persistidos. Gate E-R corrigió el orden: primero evaluó la
propuesta, sin cargar ni reconstruir CNN folds.

Contrato exacto:

- slice `frame_bgr[260:1920, 0:864]`;
- resize único a `432x830` con `INTER_AREA`;
- HSV amarillo congelado;
- `MORPH_OPEN 3x3`, connected components 8-connectivity;
- área coarse `[1,125]`;
- `x_full=2*cx+0.5`, `y_full=260+2*cy+0.5`;
- sin refinamiento full-resolution.

La propuesta falló antes de cualquier CNN replay:

- frozen positives @20: `34/60 = 56.67%`;
- A_01: `5/20`;
- B_01: `11/21`;
- C_01: `18/19`;
- cobertura @20 en los 63 frames activos: `35/63 = 55.56%`;
- primeros pares: `2/3`, endpoints `5/6`;
- cardinalidad coarse: p50 `16`, p95 `23`, máximo `26`;
- distancia nearest positive: p50 `5.167 px`, p95 `405.856 px`, máximo
  `408.486 px`.

El par A_01 `78→79` falló el segundo endpoint; B_01 `78→79` y C_01 `351→352`
pasaron. Al fallar la propuesta no se reconstruyeron modelos, no se ejecutó
transfer semantics y no se ejecutó runtime de tres repeticiones.

Evidence: `data/task011/gate_e_summary.json`.

## Approaches rechazados

Quedan explícitamente fuera de producción dentro de Task 011:

- adquisición full-frame con todos los candidatos y CNN síncrona;
- shortlist estadística K4/K8, por pérdida semántica;
- el contrato half-resolution yellow-only, por pérdida de propuestas antes del
  scorer;
- reinterpretar un logit positivo como garantía de adquisición;
- usar predicciones como detecciones;
- repetir tuning de HSV, morphology, radio, thresholds o arquitectura;
- usar los modelos LOBO DEV como modelo final all-data;
- evaluación o calibración con HOLDOUT.

El fallo no demuestra que el tracking local sea inútil. Demuestra que la
familia de propuestas síncronas probada no satisface el conjunto de gates.

## Evidencia reutilizable y handoff

Se puede reutilizar:

- estados explícitos ACQUIRE/TENTATIVE/TRACK/COAST/REACQUIRE;
- `TemporalTracker` guiado por PTS;
- separación estricta observación/predicción;
- evidencia de ROI local de 120 px;
- tooling de propuesta yellow-only y sus equivalencias;
- mediciones del suelo de runtime TRACK/local;
- evidencia semántica del scorer learned de Task 010 sólo como contexto;
- manifests, snapshots y controles DEV/HOLDOUT.

No se deben tratar como supuestos de producción:

- política cascade congelada basada en `logit > 0`;
- adquisición full-res all-candidate;
- shortlist free-stat K4/K8;
- propuesta yellow half-res;
- modelos LOBO DEV como modelo de producción;
- cualquier resultado seleccionado en DEV como evidencia de generalización.

Task 012 debe investigar un detector dedicado para ACQUIRE/REACQUIRE,
preferiblemente heatmap/direct localization u otra familia full-frame cuya
salida sea naturalmente pequeña. Debe comparar alternativas, dependencias y
coste del host antes de entrenar. Este informe no preselecciona una
arquitectura.

## Snapshots y provenance

Los snapshots compactos usados son:

- `data/task011/gate_a_diagnostics.json`
- `data/task011/gate_b_failure.json`
- `data/task011/gate_b_r2_summary.json`
- `data/task011/gate_c_summary.json`
- `data/task011/gate_d_summary.json`
- `data/task011/gate_e_summary.json`

No se añaden PNG, H264, CSV diagnósticos grandes, checkpoints, caches ni
artifacts ignorados. `HOLDOUT used=false` y `production_model=false`.
