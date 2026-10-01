# Task 009 — Offline shuttle perception benchmark and heuristic feasibility

Estado: cerrado tras el gate `Appearance-geometry gate FAIL accepted — heuristic escalation ends here` de Issue #17.

Provenance principal: rama `task/009-shuttle-perception`, baseline final del diagnóstico `a630c0c3628bf1b3672465a195c3fe814666924e`; el informe de este documento se construye únicamente con artifacts DEV ya existentes y sin acceso al teléfono.

## Dataset y reglas de evaluación

**FACT.** El ground truth congelado contiene 136 records: 68 DEV y 68 HOLDOUT, en bursts completos para evitar leakage temporal. El snapshot trackeado es `data/task009/ground_truth.json`; sus labels provienen de 136 anotaciones humanas validadas.

**FACT.** HOLDOUT nunca se utilizó para tuning, selección de representación, selección de reglas, debugging dirigido ni decisiones de arquitectura. Queda reservado para una evaluación posterior del learned scorer de Task 010.

**FACT.** La identidad del snapshot y la provenance de las anotaciones están incluidas en el propio archivo. Los snapshots compactos trackeados son:

- `data/task009/baselines/dev_untuned.json`
- `data/task009/baselines/v1_single_hypothesis_fixed.json`
- `data/task009/baselines/v2_temporal_beam_fail.json`
- `data/task009/ground_truth.json`

No se incluyen PNG, H.264, CSV diagnósticos grandes, `.venv` ni paths absolutos locales.

## Baseline inicial

**MEASURED RESULT.** El baseline `BASELINE_UNTUNED` tuvo recall seleccionado 0/63 en @5, @10 y @20 px, y FP en 5/5 frames negativos. El detector retenía 32 candidatos por frame; el diagnóstico previo al cap observó 6.702 candidatos en 68 frames.

**MEASURED RESULT.** Runtime inicial: algorithm-only p50 847.289 ms, p95 1,273.847 ms, media 873.560 ms y 1.145 FPS efectivos. El registro tuvo 49/60 transformaciones válidas y residual p95 3.054 px.

## Optimización semánticamente neutra

**FACT.** El scoring de componentes fue cambiado de máscaras full-frame por componente a operaciones ROI-locales, manteniendo scores, orden, coordenadas y tie-breaking.

**MEASURED RESULT.** Equivalence gate PASS en los 68 frames DEV: candidatos raw, candidatos retenidos, scores y asociación permanecieron equivalentes.

| `component_scoring_ms` | Antes | Después |
|---|---:|---:|
| Media | 776.003 | 12.540 |
| p50 | 735.945 | 10.358 |
| p95 | 1,152.207 | 23.710 |
| Máximo | no congelado en snapshot | 64.921 |

La reducción observada fue aproximadamente 61.9× en media, 71.0× en p50 y 48.6× en p95. El runtime global todavía no cumplía el gate de producción; esta optimización sólo eliminó un hot path sin alterar la semántica.

## Representación yellow

**MEASURED RESULT.** La representación `yellow_only` conserva un techo alto:

- raw yellow oracle: 57/63 @5, 60/63 @10 y 61/63 @20 (96.83% @20);
- el área dura `[16.0, 424.2]` reduce el techo @20 a 57/63 (90.48%);
- el diagnóstico candidate-centric procesó 4.585 candidatos en 68 frames, 41.327 pares y 497.730 tracklets de tres frames sin cap top-32 ni gate de área duro.

**ARCHITECTURE DECISION.** Yellow queda como etapa de propuestas de alta recuperación. No se justifica seguir modificando HSV, morphology o la banda de área para resolver el ranking.

## V1: adquisición de una sola hipótesis

**MEASURED RESULT.** El rerun corregido, con configuración congelada `yellow_only`, R5, banda `[16.0, 424.2]`, dos confirmaciones y gate geométrico de 120 px, obtuvo:

| Burst | Recall @20 | Interpretación |
|---|---:|---|
| A_01 | 0/21 | adquirió un distractor persistente |
| B_01 | 19/21 | adquisición correcta; tracking posterior correcto |
| C_01 | 0/21 | adquirió un distractor persistente |

Globalmente fue 19/63 @20; el error de localización de las detecciones emparejadas tuvo p50 1.542 px y p95 3.797 px. FP rate negativo: 0/5. El peor miss burst fue 5 frames y la reacquisition observada fue 2 frames / 66.291 ms.

**FACT.** B_01 demuestra que el tracker y la representación pueden seguir correctamente al shuttle cuando la adquisición inicial es correcta. El fallo principal es adquisición/ranking, no el tracker.

## Multi-hypothesis temporal

**MEASURED RESULT.** Sin gate duro de área:

- oracle de pares: 58/60 ventanas, 96.667%; 37.100 hipótesis;
- oracle de tracklets de tres frames: 55/57 ventanas, 96.491%; 457.928 hipótesis;
- A_01 y B_01 tuvieron cobertura completa;
- C_01 perdió dos ventanas consecutivas: pares en frames 369–370 y tracklets en 368–369.

El temporal beam FAIL no invalidó las propuestas yellow: ninguna regla predeclarada P1/T1–T6 mantuvo una hipótesis correcta en beam ≤32 en la primera ventana de adquisición de A/B/C.

## Appearance geometry y cierre heurístico

El diagnóstico candidate-centric añadió, sin cambiar producción:

- geometría yellow: área, aspect ratio, perímetro, circularidad, convex hull, solidity y radio equivalente;
- companions white separados, sin hacer `yellow OR white`;
- geometría cyan: componentes cercanos, PCA, linearidad, ejes, extensión y endpointness;
- comparación entre vector head→trail y velocidad temporal;
- reglas de nodo N1–N3 y tracklet G1–G4.

**MEASURED RESULT.** Rangos de la primera ventana correcta:

| Regla | A_01 | B_01 | C_01 |
|---|---:|---:|---:|
| G1 | 6 | 653 | 126 |
| G2 | 14 | 544 | 11 |
| G3 | 10 | 653 | 284 |
| G4 | 40 | 647 | 294 |

Ninguna regla conserva la hipótesis correcta en las tres ventanas dentro de beam 32. El gate `appearance_geometry_gate` es `FAIL`; no se seleccionó regla ni beam recomendado. El reporte completo, con distribuciones p05/p10/p25/p50/p75/p90/p95 y controles A/B/C, está en `artifacts/task009/appearance_geometry/report.json` (gitignored, reproducible desde los captures existentes).

**ARCHITECTURE DECISION.** La escalada de heurísticas termina aquí:

```yaml
hybrid_heuristic_representation: useful_high_recall_proposals
heuristic_acquisition_ranking: inadequate
next_architecture: learned_candidate_scorer
```

## Runtime y gates congelados

El diagnóstico final de geometría tuvo p50 aproximado de 670.419 ms/frame para preprocesado+geometría y no representa un runtime de producción; procesó muchas features y tracklets para medir factibilidad.

Los gates de Task 009 permanecen sin cambios:

- recall @20 ≥ 0.90;
- recall @10 ≥ 0.80;
- localization p50 ≤10 px y p95 ≤20 px;
- FP negativo = 0/5;
- longest miss burst ≤2 frames;
- reacquisition ≤2 frames y ≤70 ms;
- registration success ≥0.80 y residual p95 ≤4 px;
- algorithm p95 ≤33 ms/frame y ≥30 FPS.

No se relajan gates por resultados DEV y no se ha ejecutado evaluación HOLDOUT.

## Recomendación para Task 010

**ARCHITECTURE DECISION.** No añadir más reglas manuales. Preservar las propuestas yellow como etapa de alta recuperación y entrenar un scorer/ranker candidate-centric sobre patches de resolución nativa, usando centros humanos y validación por bursts. El scorer debe decidir también `NO OBSERVATION` y alimentar la adquisición/beam existente; el tracker temporal se conserva.

Un detector full-frame queda como fallback si el scorer de propuestas no generaliza. Esta ruta evita empezar por una arquitectura más grande y ataca directamente el cuello de botella medido.

Issue #18 define primero una auditoría de alternativas y dependencias; Task 010 aún no se implementa.

## Riesgos abiertos

- Sólo hay tres bursts activos en DEV; la selección de modelo debe usar validación por burst y mantener HOLDOUT intacto.
- Los frames negativos son muestras de estado aisladas; no prueban por sí solos un gate temporal de estado activo.
- El scorer aprendido puede no generalizar fuera de A/B/C; la evaluación HOLDOUT será la primera comprobación reservada.
- La latencia del pipeline completo sigue por encima del gate aunque el hot path de scoring se haya optimizado.

## Cierre

**FACT.** Task 009 no accedió al teléfono, no envió input/control, no instaló nuevas dependencias durante esta fase de cierre y no utilizó HOLDOUT.

**MEASURED RESULT.** La representación yellow ofrece propuestas suficientes; todas las reglas de ranking temporal y candidate-centric probadas fallan el gate de primera adquisición.

**ARCHITECTURE DECISION.** Task 009 queda cerrada. La siguiente investigación justificada es el learned candidate scorer de Issue #18.

**OPEN RISK.** No debe interpretarse este documento como evidencia de generalización a HOLDOUT ni como implementación de Task 010.
