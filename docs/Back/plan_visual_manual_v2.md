# Plan visual y editorial del manual v2

## Decisiones

- Mantener el PDF anterior en `docs/` sin moverlo ni borrarlo, por petición explícita del usuario.
- Crear fuente nueva `docs/report_v2.html` y PDF nuevo `docs/Laboratorio-de-Modelos-Generativos-v2.pdf`.
- Reusar capturas reales y métricas existentes para preservar evidencia del sistema real.
- Añadir visuales HTML/SVG embebidos para pedagogía: mapas, microscopios, flujos y pósters.
- Formalizar anexos: instalación, mapa del código, código esencial, glosario/lecturas.

## Plan por capítulo

| Capítulo | Visual nuevo | Función pedagógica | Arquetipo |
|---|---|---|---|
| Guía de lectura | Ruta en 4 pasos + timeline | Orientar antes de conceptos | Mapa de sistema |
| Datos y latente | Píxeles→tensor→modelo→z | Definir vocabulario común | Flujo secuencial |
| Autoencoder | Microscopio 12.288→128 | Hacer visible la compresión | Radiografía |
| VAE | Puntos vs nubes | Explicar muestreo y KL | Mapa conceptual |
| GAN | Bucle adversarial | Mostrar feedback G/D | Flujo causal |
| Diffusion | Forward/reverse | Separar proceso fijo y aprendido | Storyboard |
| Comparador | Poster de compromisos | Evitar ranking simplista | Postal editorial |
| Anexo D | Glosario operativo | Relectura y auditoría | Referencia |

## QA previsto

- Comprobar rutas de imágenes referenciadas por `report_v2.html`.
- Generar PDF con Chrome headless.
- Contar páginas, tamaño y presencia de Anexo A/B/C/D con `pypdf`.
- Renderizar páginas clave a PNG y revisar contact sheets.
- Buscar restos de marcadores pendientes y referencias internas antiguas.
