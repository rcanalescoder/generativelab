# Seguimiento manual v3

Estado: publicado (2026-06-12).

## Objetivo

Rehacer el PDF como versión 3 contando el ciclo de calidad: qué cambios de **algoritmos** y de
**parámetros** se hicieron en cada modelo, y el **antes/después visual y medido** (petición
explícita del usuario). Fuente de datos: `Lista de Mejoras.md` + `experiments/ledger.jsonl`.

## Qué añade la v3 sobre la v2

- Portada y cierre actualizados a «Versión 3» (ribbon, subtítulo con las cifras, pie de edición).
- Puntero en la introducción (el «hilo conductor» agéntico ahora remite al capítulo nuevo).
- **Cápsulas de actualización** al final de los capítulos VAE, GAN y Diffusion (el capítulo v2 se
  conserva como «antes» histórico; la cápsula dice qué cambió y remite al capítulo v3).
- **Capítulo nuevo «Versión 3 · La historia»** (5 secciones, ~13 páginas):
  1. La historia: diagnóstico ×4, el ciclo en 4 pasos, las métricas (KID/diversidad/rejilla) y el protocolo.
  2. VAE: el bug de escala — antes/después (rejillas reales 8×8), tabla de cambios con chips
     algoritmo/parámetro, evidencia (608→241→121), anécdota logσ² y límite honesto.
  3. GAN: 4 estabilizadores — antes/después, tabla de cambios, evidencia (199→71→52→37,5) y la
     lección «los screens cortos mintieron» (TTUR descartado con datos).
  4. Diffusion: presupuesto real — antes/después, tabla de cambios (incl. «lo que no se tocó»),
     evidencia (175→19,4).
  5. Veredicto: tabla final v2/v3, timeline V3.0–V3.4, póster de 4 lecciones, ruta reproducible
     (comandos del runner) y autocomprobación.
- Anexo A · paso 4 actualizado a las recetas v3 (runner + train_diffusion_v3) con nota del ledger.

## Assets creados

- `docs/assets/v3_{vae,gan,diff}_{antes,despues}.jpg` — rejillas 8×8 reales de
  `experiments/grids/` (semilla fija, 768px, JPG q92).
- `docs/assets/pdf_cover.jpg` y `pdf_preview.jpg` regenerados desde el PDF v3
  (preview: páginas 35 · 37 · 40 · 45).
- CSS v3: `.ba-pair` (par antes/después), `.ba-tag`, `.delta-strip`, chips `.tipo`, y reglas
  `break-inside:avoid` para metric-guide/concept-flow/chapter-frame/two-col (arregló el corte
  de tarjetas detectado en QA).

## QA ejecutado

- Rutas de imágenes del HTML: 17 referenciadas, 0 faltantes.
- PDF (Chrome headless): **55 páginas, 5,8 MB** (v2: 42 páginas).
- QA textual (pypdf): «Versión 3» en portada/cierre, cifras 608/37,5/19,4 presentes, 4 anexos
  presentes (los bands extraen como «A N E X O» por el letter-spacing — igual que en v2),
  sin restos de «Versión 2 ampliada».
- QA visual (pypdfium2, renders en `qa_visual/v3/`): portada, capítulo v3 completo (35–46),
  glifos σ y superíndices ⁻⁴ verificados en el texto extraído. Incidencia corregida: las
  tarjetas de métricas se partían entre páginas (cajas vacías) → reglas break-inside.

## Publicación

- `docs/Laboratorio-de-Modelos-Generativos-v3.pdf` + `docs/report_v3.html` (fuente).
- v2 movida a `Back/` (PDF + report_v2.html); v1 ya la había movido el usuario a mano.
- README: sección del cuaderno → v3 (55 páginas · ~5,8 MB), portada y preview regenerados.
- `build_manual_v3.py` lee la fuente v2 desde `Back/` (reproducible tras el archivado).
