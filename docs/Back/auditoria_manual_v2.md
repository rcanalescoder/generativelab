# Auditoría pedagógica del manual v2

Fecha: 2026-06-10

Documento base: `docs/Laboratorio-de-Modelos-Generativos.pdf` y `docs/report.html`.
Documento nuevo: `docs/report_v2.html` → `docs/Laboratorio-de-Modelos-Generativos-v2.pdf`.

| Sección | Linealidad para junior | Qué había que ampliar | Visuales que ayudan |
|---|---|---|---|
| Introducción | Buena, con tono accesible. | Faltaba mapa de lectura explícito y contrato de evidencia. | Ruta de lectura en 4 pasos y timeline de modelos. |
| Datos y latente | Estaba distribuido entre capítulos. | Convenía explicar tensor, dataset y tipos de latente antes de los modelos. | Figura píxeles→tensor→modelo→z y tabla AE/VAE/GAN/Diffusion. |
| Autoencoder | Sólido y con resultados reales. | Añadir pregunta inicial, microscopio del cuello, guía de métricas, límites y autocomprobación. | Flujo de compresión 12.288→128, guía MSE/PSNR/SSIM. |
| VAE | Explicaba bien la nube latente. | Reforzar el trade-off β y diferenciar prior/posterior. | Comparación de puntos AE vs nubes VAE. |
| GAN | Correcto para el duelo G/D. | Aclarar que no reconstruye, que la pérdida no se lee como supervisado y que nitidez no es diversidad. | Bucle adversarial generador-discriminador. |
| Diffusion | Buena intuición de ruido. | Hacer explícito forward conocido vs reverse aprendido y coste de pasos. | Filmstrip conceptual x0→xT y flecha reverse. |
| Comparador | Útil, pero podía leerse como ranking. | Enfatizar que compara compromisos, no un ganador universal. | Poster de síntesis por tarea. |
| Anexos | Existían como secciones, no como anexos formales. | Nombrar A/B/C/D y añadir glosario/lecturas. | Índice de anexos y glosario de pares confundibles. |
