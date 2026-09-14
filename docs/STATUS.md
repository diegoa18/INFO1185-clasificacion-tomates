# Estado experimental

Última actualización: 2026-09-14

## Autoridad y milestone actual

`docs/IA_2026_P1.pdf` es el enunciado oficial y prevalece sobre este estado.
Se corrigió el protocolo experimental existente conforme a sus secciones
2, 3.1–3.5 y 4. SFS y PCA siguen pendientes de implementación.

## Partición corregida y congelada

Unidad experimental: una imagen, incluso si contiene varios tomates.
Dataset validado: 62 imágenes con sus máscaras, 31 ripe y 31 unripe.

| Split | ripe | unripe | Total | Porcentaje |
| --- | ---: | ---: | ---: | ---: |
| training | 18 | 19 | 37 | 59.7 % |
| validation | 6 | 6 | 12 | 19.4 % |
| test | 7 | 6 | 13 | 21.0 % |

Archivo: `artifacts/split.csv`. Las proporciones aproximan 60/20/20 con
redondeo entero y preservan las 13 imágenes de test ya utilizadas.
Ninguna imagen de test volvió al desarrollo.

Las 49 imágenes antiguamente no-test se ordenaron por `label,image` y se
subdividieron mediante `train_test_split(test_size=12, stratify=label,
random_state=42)`. La semilla no se eligió comparando resultados.
`src/protocol.py` verifica la distribución, la subdivisión determinista y
la huella SHA256 de la pertenencia original a test. Repetir
`scripts/create_split.py` comprueba el split sin sobrescribirlo.

### Función de cada partición

- **Training:** EDA, justificación manual y ajuste de parámetros GaussianNB.
- **Validation:** selección de canales por Jaccard y umbral mediante Youden.
- **Test:** evaluación final del pipeline congelado.

Development significa training + validation; ya no es un valor de `split`.
Se utiliza holdout, sin CV anidada. No se reajusta el clasificador con
training + validation después de elegir el umbral: se conserva el modelo
de training para que el score de test use exactamente la misma escala.

Validation es un conjunto de selección, no una estimación independiente
del desempeño final. Sus imágenes fueron usadas previamente en el EDA del
antiguo development: la repartición no elimina esa exposición histórica.

## EDA y selección manual: recalculados en training

Se regeneraron tablas e histogramas/boxplots RGB y HSV sobre 37 imágenes.
Las máscaras de referencia se usan aquí para describir fruto y fondo;
no se usan para construir entradas del clasificador.

| Canal | Media ripe | Media unripe | Diferencia absoluta |
| --- | ---: | ---: | ---: |
| R | 0.7789 | 0.5633 | 0.2156 |
| G | 0.2978 | 0.6688 | 0.3710 |
| B | 0.1868 | 0.2886 | 0.1018 |
| H | 0.1939 | 0.2224 | 0.0286 |
| S | 0.7840 | 0.5986 | 0.1854 |
| V | 0.7815 | 0.6704 | 0.1110 |

Se reconfirma **R,G,S**: G presenta la mayor separación, R recoge el mayor
componente rojo de ripe y S aporta separación en saturación. Las tres
diferencias superan las restantes. H es circular y su media aritmética
puede ocultar separación alrededor del origen del tono.

La evidencia está en `results/eda/tables/ripe_vs_unripe_summary.csv` y
`results/eda/figures/`. Es una selección basada en análisis, no una búsqueda
de subconjuntos optimizada por exactitud de validation/test.

## Segmentación: selección recalculada en validation

Se ejecutaron las siete combinaciones sobre training y validation por
separado (343 ejecuciones). Se eligió el máximo Jaccard medio de validation;
los empates siguen el orden R,G,B,RG,RB,GB,RGB.

| Canales | Jaccard medio training | Jaccard medio validation |
| --- | ---: | ---: |
| R | 0.4796 | 0.3624 |
| G | 0.4821 | 0.3077 |
| B | 0.4488 | 0.2974 |
| **RG** | **0.5476** | **0.3808** |
| RB | 0.5347 | 0.3725 |
| GB | 0.4992 | 0.3139 |
| RGB | 0.5315 | 0.3651 |

Configuración congelada: **RG**. Su selección está respaldada por esta
comparación; las diferencias pequeñas no demuestran superioridad estadística.

Parámetros: K=2 (fruto/fondo), init=k-means++, n_init=10, max_iter=300,
tol=1e-4, random_state=42, border_fraction=0.05. Son parámetros fijos, no
optimizados con test. Se fija un hilo numérico con threadpoolctl para
evitar variabilidad por reducciones paralelas y sobreasignación de hilos.
K-Means se ajusta a los píxeles de cada imagen como operación de inferencia;
no se aprende una transformación entre imágenes usando test.

Se identifica el fruto como el cluster con menor ocupación del borde,
bajo el supuesto de que el fondo suele alcanzar los límites de la imagen.
En empate numérico (`np.isclose`), se usa el cluster del píxel central.
La referencia nunca decide el cluster en inferencia.

El diagnóstico oracle se recalculó solo en training: para RG, Jaccard
automático 0.5476, oracle 0.5544, pérdida de selección 0.0068 y tasa de
elección del mejor cluster 0.9189. Sugiere que dos clusters cromáticos
siguen siendo una limitación en escenas complejas, más que la regla de borde.

`results/segmentation/selected_configuration.json` conserva selección,
parámetros y huella del split; los scripts posteriores lo consultan.
Las figuras representativas corresponden a validation: mejor 0.8039,
ejemplo más cercano a la mediana 0.2460, peor 0.1341. La mediana del
conjunto de 12 valores es 0.2813; el ejemplo es una imagen real cercana.

## Características del clasificador

`results/features/segmented_features.csv` fue regenerado desde las máscaras
predichas con la configuración seleccionada. El artefacto activo de
desarrollo contiene **49 filas**: 37 training y 12 validation, sin valores
no finitos en las seis características. Las 13 filas de test se excluyen
del artefacto activo mientras SFS y PCA sigan pendientes.

Únicos candidatos: **R,G,B,H,S,V**, medias aritméticas sobre el foreground
predicho. RGB se divide por 255 (escala fija, sin ajuste con datos); HSV
se obtiene por transformación determinista. HSV permite analizar tono,
saturación e intensidad complementariamente a RGB; H conserva la limitación
de circularidad indicada arriba.

Jaccard, tamaño del foreground, etiqueta, split y rutas son metadatos, no
entradas. La máscara de referencia solo participa en Jaccard, calculado
separadamente de las características.

La extracción habitual excluye test. `--include-test` permite ejecutar
la segmentación congelada para la evaluación final.

## Bayes + selección manual: corregido y reevaluado

Gaussian Naive Bayes con `var_smoothing=1e-9`, R,G,S y ajuste exclusivo
sobre 37 imágenes de training. Es un modelo sencillo para variables
continuas y un dataset pequeño; asume gaussianidad e independencia
condicional, aproximaciones que no están garantizadas para colores.

El score ahora calcula explícitamente las log-densidades gaussianas usando
`theta_` y las varianzas suavizadas `var_` de training:

`log p(x | ripe) - log p(x | unripe)`.

No incluye priors. Positivo: ripe; negativo: unripe. ROC especifica
`pos_label="ripe"` y AUC codifica explícitamente ripe=1. Se verificaron
orientación y ausencia de priors con ejemplos sintéticos desbalanceados.

Se maximiza Youden en validation porque pondera por igual sensibilidad y
especificidad sin introducir costes no establecidos en el enunciado.
Empates: mayor umbral finito, con tolerancia absoluta 1e-12 para que el
redondeo flotante no rompa igualdades matemáticas. Regla: ripe si score >= umbral.

- Umbral congelado: **-0.14287354715030742**.
- Youden en validation: **0.6666666666666667**.
- Modelo (medias/varianzas), subconjunto, criterio y umbral se guardan en
  `results/classification/bayes_analysis/decision.json` antes de evaluar test.
- `--evaluate-test` reconstruye el modelo solo con training y verifica
  igualdad con la decisión guardada antes de calcular resultados de test.

| Evaluación | Exactitud | Sensibilidad | Especificidad | AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (selección) | 0.8333 | 0.8333 | 0.8333 | 0.8333 | 5 | 1 | 1 | 5 |

Hubo una evaluación de test previa a la implementación de SFS y PCA.
Esa exposición histórica no puede deshacerse, pero sus métricas numéricas
se omiten deliberadamente de este estado operativo para no condicionar las
etapas pendientes. Durante el desarrollo de SFS/PCA no se deben inspeccionar
ni regenerar predicciones, métricas, ROC, figuras o características de test.

Los artefactos activos de Bayes conservan la decisión congelada y la
evaluación de validation. La evaluación final sobre test se regenerará
únicamente cuando los tres pipelines requeridos estén congelados.

## Resultados anteriores sustituidos

Los artefactos anteriores se conservaron sin editar su contenido en
`results/legacy_development_test/`. `create_split.py` también detecta los
esquemas/resultados antiguos si otra copia recibe el split ya corregido por
Git, antes de regenerar las salidas. No son resultados del protocolo vigente:

- EDA y justificación calculados sobre las 49 imágenes mezcladas;
- agregados de segmentación y diagnóstico sobre development completo;
- antigua tabla de características con etiquetas `development`;
- ROC, umbral, métricas y predicciones OOF del protocolo anterior;
- evaluación de test del modelo ajustado con 49 imágenes y score posterior.

R,G,S y RG coinciden con las elecciones anteriores, pero se reconfirmaron
con los conjuntos y criterios corregidos. Los parámetros de GaussianNB,
el score y el umbral se recalcularon; las cifras históricas no se trasladan
a la comparación final. Los resultados de test históricos no fueron
consultados para diseñar esta corrección.

## Reproducción y verificación

Entorno ejecutado: Python 3.13.5. Versiones fijadas en `requirements.txt`:
NumPy 2.4.3, pandas 3.0.3, Pillow 11.1.0, scikit-learn 1.9.0,
matplotlib 3.10.8 y threadpoolctl 3.6.0.

Desde la raíz, con las dependencias instaladas:

```bash
python3 scripts/create_split.py
python3 scripts/extract_eda_features.py
python3 scripts/plot_eda.py
python3 scripts/run_segmentation.py
python3 scripts/diagnose_segmentation.py
python3 scripts/plot_segmentation_examples.py
python3 scripts/extract_segmented_features.py
python3 scripts/run_bayes_analysis.py
python3 scripts/verify_protocol.py
```

Solo para la compuerta final de test, después de congelar **los tres**
pipelines (manual, SFS y PCA):

```bash
python3 scripts/extract_segmented_features.py --include-test
python3 scripts/run_bayes_analysis.py --evaluate-test
```

Mientras SFS o PCA sigan pendientes, no ejecutar esos comandos.

Comprobaciones requeridas:

```bash
python3 -m compileall src scripts
python3 scripts/verify_protocol.py
git diff --check
git status --short
git diff
```

`verify_protocol.py` comprueba la partición determinista, rechazo de cambios
de test, EDA de training, elección de segmentación en validation, parámetros
del modelo de training, umbral de validation, scores guardados y orientación
de ROC/AUC. Los resultados están ignorados por Git y se regeneran con los
comandos anteriores. No se ha realizado staging, commit ni push.

### Revisiones de este milestone

- **methodology-reviewer:** dictamen favorable; no detectó contradicciones
  con el PDF ni leakage que bloquee este milestone. Reconoció las
  limitaciones de exposición histórica y del tamaño de validation/test.
- **reviewer:** detectó dos errores corregidos: archivado omitido cuando
  el split corregido llega por Git y desempate de Youden sensible al
  redondeo. Su segunda revisión confirmó ambos arreglos, sin hallazgos
  pendientes ni regresiones detectadas.
- Se añadieron regresiones sintéticas para ambos casos. Después de los
  arreglos se ejecutaron de nuevo `create_split.py` (dos veces), Bayes en
  validation, `compileall`, `verify_protocol.py` y `git diff --check`.
  Todas las comprobaciones de desarrollo pasaron y el umbral de validation
  permaneció estable. Los reviewers hicieron revisión estática; las
  ejecuciones anteriores se realizaron en la sesión principal.

## Siguiente milestone

**SFS + GaussianNB**, solo cuando se solicite. Usar candidatos R,G,B,H,S,V.
Antes de evaluar candidatos, documentar métrica, parada y desempates.
Ajustar cada candidato en training, seleccionar en validation y elegir
Youden en validation para el modelo seleccionado. Congelar subconjunto,
modelo de training y umbral. Registrar la trayectoria y comparar con R,G,S.
No inspeccionar ni regenerar test durante este milestone. No se exige CV
anidada en este protocolo holdout.

Posteriormente: StandardScaler y PCA ajustados solo en training, número
de componentes justificado por varianza explicada de training (PDF 3.6),
GaussianNB en training y Youden en validation. Cuando manual, SFS y PCA
estén congelados, abrir una única compuerta final de test y comparar los
tres sin revisar decisiones a partir de ese resultado. SFS y PCA no están
implementados en este milestone.
