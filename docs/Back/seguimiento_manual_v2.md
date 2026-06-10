# Seguimiento manual v2

Estado: validado en PDF.

## Objetivo

Crear una versión 2 del manual PDF, más didáctica, pedagógica y detallada, conservando la versión anterior.

## Criterios de aceptación

- Existe `docs/Laboratorio-de-Modelos-Generativos-v2.pdf`.
- Existe `docs/report_v2.html` como fuente.
- La versión anterior `docs/Laboratorio-de-Modelos-Generativos.pdf` sigue intacta.
- El PDF incluye anexos A, B, C y D.
- Las imágenes referenciadas existen.
- Se hizo QA textual y visual del PDF.

## Log

- 2026-06-10: leídas instrucciones del proyecto, protocolo de generación de libros, PDF/HTML actual y estructura del código.
- 2026-06-10: planificada v2 como capa pedagógica sobre `report.html`, con nuevos bloques de lectura, visuales SVG y anexos formales.
- 2026-06-10: generado `report_v2.html` desde `docs/Back/build_manual_v2.py`.
- 2026-06-10: generado `docs/Laboratorio-de-Modelos-Generativos-v2.pdf` con Chrome headless sin cabeceras/pies del navegador.
- 2026-06-10: QA textual con `pypdf`: 42 páginas, 3,96 MB, anexos por título presentes, 4 bloques de error/límite y sin marcadores pendientes ni marcas antiguas.
- 2026-06-10: QA visual con Ghostscript: 42 páginas renderizadas a PNG, hojas `contact_all.png` y `contact_key.png` revisadas sin cortes ni solapes visibles.

## Assets creados

- Visuales nuevos embebidos como SVG/HTML en `docs/report_v2.html`.
- QA visual en `docs/Back/qa_visual/v2/` tras generar el PDF.

## Páginas PDF revisadas

- Portada.
- Guía de lectura.
- Datos y espacio latente.
- Inicio y cierre pedagógico de AE, VAE, GAN y Diffusion.
- Comparador.
- Anexos A, B, C y D.
- Cierre editorial/licencia.

## Decisiones de publicación

- No se actualiza `README.md` ni se mueve la versión anterior hasta revisión humana.
- No se hace commit porque el usuario pidió crear la v2, no publicar/cerrar una fase de producto.
