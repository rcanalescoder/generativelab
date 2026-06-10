# Instrucciones para generar libros técnicos-pedagógicos

Este documento define un protocolo reutilizable para convertir la documentación de un proyecto en un manual técnico visual, pedagógico y publicable. Está escrito para poder aplicarlo a otra disciplina o repositorio sin repetir todas las decisiones tomadas durante la creación de `Arkanoid-DRL-Learning-Lab-v4.pdf`.

La idea operativa es:

> Lee `docs/Back/instrucciones generación de libros.md`, audita el proyecto y construye un manual nuevo siguiendo este protocolo.

## Principio central

El libro no debe ser una colección de páginas bonitas. Debe ser un sistema de explicación:

- Un lector junior debe poder seguir el razonamiento sin dominar previamente la disciplina.
- Cada imagen debe enseñar algo que el texto por sí solo no descarga de la memoria de trabajo.
- El diseño debe ser homogéneo, pero las infografías no deben parecer repetidas.
- Las cifras, gráficas y capturas deben indicar qué demuestran y qué no.
- El resultado final debe poder verificarse: HTML, assets, PDF, README y enlaces.
- El documento original se conserva; las versiones antiguas se archivan en `Back/`.

## Entregables esperados

Adaptar nombres al proyecto, pero mantener esta estructura:

```text
docs/
├── <Nombre-del-manual>-vN.pdf        # PDF final publicado
├── report_vN.html                    # fuente HTML final
├── assets/                           # imágenes usadas por el HTML final
└── Back/                             # versiones previas y documentos de trabajo
    ├── <manuales-antiguos>.pdf
    ├── <reports-antiguos>.html
    ├── auditoria_*.md
    ├── plan_*.md
    ├── seguimiento_*.md
    ├── assets_intermedios/           # regenerable; normalmente ignorado por Git
    └── qa_visual/                    # renders QA; normalmente ignorado por Git
```

Nombres recomendados:

- Fuente nueva: `docs/report_v4.html`, `docs/report_v5.html`, etc.
- PDF final: `docs/<Proyecto>-v4.pdf`, `docs/<Proyecto>-v5.pdf`, etc.
- Evitar nombres con marcas internas del agente en el PDF final, salvo que el usuario lo pida.
- Las versiones previas deben moverse a `docs/Back/` para que no confundan a quien abre el repositorio.

## Flujo completo

1. Auditar el documento original y el proyecto.
2. Crear una copia de trabajo del HTML o documento fuente.
3. Revisar la linealidad pedagógica de todos los capítulos.
4. Definir interfaz editorial común: colores, cajas, captions, jerarquía, anexos.
5. Crear un plan visual por capítulo y por función pedagógica.
6. Diseñar visuales por lotes, con variedad de arquetipos.
7. Renderizar visuales a PNG y revisar contact sheets.
8. Integrar visuales en el HTML y comprobar rutas.
9. Añadir anexos finales: uso del proyecto, mapa de código, snippets, glosario y lecturas.
10. Generar el PDF solo cuando el lote esté completo o el usuario lo autorice.
11. Renderizar páginas clave del PDF y revisar visualmente.
12. Actualizar README, portada/preview y enlaces.
13. Mover versiones antiguas a `Back/`.
14. Ejecutar build/test del proyecto si aplica.
15. Commit y push solo cuando el usuario lo pida o sea parte explícita de la publicación.

Si el usuario pide no generar PDF hasta el final, trabajar con HTML + QA de imágenes y hacer una única generación final.

## Auditoría pedagógica

Antes de crear imágenes, leer todos los apartados de texto y responder:

- ¿El capítulo define los términos antes de usarlos?
- ¿La progresión va de intuición a mecanismo, de mecanismo a evidencia y de evidencia a límites?
- ¿Un ingeniero junior puede entenderlo sin conocimiento experto previo?
- ¿Hay fórmulas sin intuición o ejemplo antes?
- ¿Hay resultados sin protocolo, muestra, semillas, split o contexto?
- ¿Hay capturas sin decir qué mirar?
- ¿Hay gráficas sin explicar ejes, unidades, dispersión o lectura correcta?
- ¿Se confunde recompensa con éxito, demo con evidencia o validación con test?
- ¿Hay conceptos repetidos con nombres o colores inconsistentes?
- ¿El cierre fija la idea o simplemente termina el texto?

Registrar la auditoría en una tabla:

```md
| Sección | Linealidad para junior | Qué habría que ampliar | Visuales que ayudarían |
|---|---|---|---|
```

## Interfaz pedagógica común

Cada capítulo o sección debe parecer parte del mismo manual. La estructura base es:

1. Pregunta inicial: qué problema resuelve.
2. Mapa o escena: dónde encaja.
3. Intuición: explicación llana.
4. Mecanismo: fórmula, arquitectura, flujo o algoritmo.
5. Traducción al proyecto: dónde aparece en el sistema real.
6. Mini-ejemplo: caso concreto, paso, transición o evento.
7. Evidencia: tabla, captura, gráfica o resultado.
8. Límite: qué no demuestra.
9. Error común: confusión probable.
10. Síntesis: qué debe recordar el lector.
11. Autocomprobación: preguntas breves.
12. Bloque experto opcional: formalismo para lectores avanzados.

No todos los capítulos necesitan todas las piezas, pero la experiencia debe ser estable.

## Tipos de capítulos

### Capítulo conceptual

Usar:

- Mapa de la idea.
- Analogía o escena.
- Definición formal.
- Traducción al proyecto.
- Mini-ejemplo.
- Trampa conceptual.
- Infografía de cierre.

Visuales recomendados: 2-4 visuales más una infografía final.

### Capítulo de algoritmo, método o arquitectura

Debe incluir, siempre que sea posible:

1. Ficha de identidad: familia, qué aprende, entrada, salida, coste y resultado principal.
2. Diagrama mental: intuición del método.
3. Pipeline de entrenamiento o ejecución.
4. Arquitectura: entradas, ramas, fusión, salidas.
5. Función de pérdida u objetivo visualizado.
6. Captura o inspector real anotado.
7. Curva o resultado anotado.
8. Infografía de cierre: cuándo funciona, cuándo falla y qué enseña.

Visuales recomendados: 4-7 si es un algoritmo central.

### Capítulo de datos, resultados o evaluación

Antes de cualquier gráfico, explicar:

- Métrica.
- Unidad.
- Eje X.
- Eje Y.
- Split o conjunto de evaluación.
- Número de semillas, muestras o repeticiones.
- Qué significa la banda o dispersión.
- Qué lectura sería equivocada.

Visuales recomendados:

- Guía de lectura de la métrica.
- Gráfico o tabla principal.
- Visual de incertidumbre, varianza o colapso.
- Matriz comparativa o dashboard anotado.
- Resumen del veredicto.

### Capítulo de producto, interfaz o demo

Conservar capturas reales cuando prueban que el sistema existe o enseñan una interfaz importante.

Cada captura debe tener:

- Callouts o numeración.
- Caption con “qué mirar”.
- Separación entre lo que la demo enseña y lo que no demuestra.
- Relación con el protocolo real de evaluación.

### Capítulo de cierre

Debe ser editorial y memorable:

- Poster final del viaje.
- Lecciones transferibles.
- Glosario visual o mapa mental.
- Ruta reproducible: código -> ejecución -> datos -> figuras -> libro.
- Checklist para seguir aprendiendo o auditar otro proyecto.

## Anexos obligatorios en la edición final

Una versión publicable no debe terminar solo con conclusiones. Debe ayudar a reproducir, usar y auditar el proyecto.

Añadir al final, salvo que no aplique:

### Anexo A: instalación y ejecución

Debe incluir:

- Requisitos mínimos.
- Comandos de instalación.
- Cómo abrir la app o demo.
- Cómo ejecutar tests, verificaciones o build.
- Cómo lanzar una ejecución pequeña desde terminal.
- Qué mirar al arrancar.
- Qué no se debe inferir de una ejecución rápida.

Ejemplo de estructura:

```md
Requisitos
Instalación limpia
Abrir la interfaz
Verificar el proyecto
Ejecutar un experimento pequeño
Errores comunes
```

### Anexo B: mapa del código

Debe mostrar un árbol de archivos con responsabilidad pedagógica:

```text
src/
├── core/          # constantes, configuración, eventos
├── domain/        # lógica central del proyecto
├── models/        # modelos, algoritmos o componentes principales
├── data/          # datasets, buffers, loaders
├── training/      # bucles, métricas, evaluación
├── ui/            # interfaz, paneles, visualización
└── scripts/       # CLI, verificación, generación de artefactos
```

No basta con listar archivos. Cada rama debe decir por qué importa.

### Anexo C: código esencial

Debe contener fragmentos cortos y legibles:

- Configuración central.
- Modelo o arquitectura principal.
- Bucle de entrenamiento o ejecución.
- Función de pérdida, métrica o regla de decisión.
- CLI o script de reproducción.
- Verificación o evaluación.

Reglas:

- Snippets breves, no volcados enteros de archivo.
- Cada snippet debe tener una frase que explique qué idea del libro implementa.
- Escapar correctamente `<`, `>`, `&` si el documento es HTML.
- No meter líneas tan largas que salgan del bloque.

### Anexo D: glosario y lecturas

Debe incluir:

- Términos operativos usados en el manual.
- Diferencias que suelen confundirse.
- Lecturas recomendadas con una frase de por qué importan.
- Preguntas finales para no perder rigor.

El glosario debe servir a un junior para volver al libro semanas después.

### Licencia y autoría

Cerrar con:

- Licencia del proyecto.
- Autoría.
- Herramientas o colaboradores si se deben acreditar.
- Repositorio y web.

## Cantidad de imágenes

Regla práctica:

- Cada capítulo que introduce conceptos necesita al menos 2 visuales.
- Cada capítulo denso necesita 4-7 visuales.
- Cada 1-2 páginas debe aparecer una ayuda visual, salvo anexos muy textuales.
- Cada sección o capítulo debe cerrar con una infografía de síntesis si aporta comprensión.
- Cada parte grande debe cerrar con una infografía más ambiciosa.
- No añadir imágenes decorativas.

Funciones válidas:

- Orientar: mapa, ruta, índice visual.
- Explicar: mecanismo, flujo, arquitectura.
- Comparar: antes/después, A/B, familias.
- Medir: gráfica, tabla, heatmap, curva.
- Advertir: error común, límite, trampa conceptual.
- Resumir: ficha final, poster, checklist.
- Reproducir: ruta desde código a resultado.

## Infografías de cierre

Al final de una sección importante debe haber una infografía de síntesis con fondo blanco y estilo de panel docente. No debe ser una tarjeta grande ni una lista de bullets adornada. Debe poder leerse como una página visual autónoma.

Estructura recomendada:

1. Título grande: conclusión pedagógica.
2. Subtítulo: por qué importa en el proyecto.
3. Secciones numeradas: `1`, `2`, `3`, opcionalmente `4`.
4. Pieza central fuerte: arquitectura, flujo, red, mapa, gráfica, tablero, dashboard o ejemplo.
5. Tarjetas laterales: conceptos, pasos, métricas, errores o variantes.
6. Franja final: conclusión fuerte en una o dos frases.

Reglas visuales:

- Fondo blanco real.
- Azul editorial como columna vertebral.
- Colores secundarios por función, no por decoración.
- Bordes finos y consistentes.
- Iconos o mini-diagramas que expliquen.
- Mucho aire blanco.
- Texto corto dentro de cajas.
- Conclusión visible sin leer todo el capítulo.

Evitar:

- Fondos pastel extensos.
- Verde claro dominante, beige dominante, morado pálido dominante.
- Paneles que parecen una colección de tarjetas iguales.
- Títulos que pisan iconos.
- Fórmulas pegadas a bordes.
- Textos saliendo de cajas.
- Miniaturas decorativas que no enseñan nada.

## Variedad visual sin perder coherencia

La coherencia viene de:

- Fondo blanco.
- Tipografía y escala consistentes.
- Paleta funcional.
- Bordes y espaciado comunes.
- Captions homogéneos.
- Misma forma de nombrar variables y métricas.
- QA visual repetible.

La variedad viene de elegir el arquetipo correcto:

| Arquetipo | Sirve para | Ejemplos |
|---|---|---|
| Mapa de sistema | Relacionar piezas | Arquitectura, protocolo, dependencias |
| Radiografía | Abrir una caja negra | Red, modelo, módulo, pipeline |
| Flujo secuencial | Explicar pasos | Entrenamiento, inferencia, datos |
| Storyboard | Mostrar una historia | Episodio, error acumulado, decisión |
| Antes/después | Enseñar desbloqueo | Formulación mala vs buena |
| Matriz comparativa | Comparar opciones | Algoritmos, métricas, configuraciones |
| Dashboard anotado | Guiar una interfaz | App, panel, monitor, inspector |
| Gráfica explicada | Leer evidencia | Curvas, barras, heatmaps |
| Microscopio de fórmula | Hacer legible una ecuación | Pérdida, actualización, score |
| Diagnóstico | Pasar de síntoma a causa | Colapso, sesgo, fallo, cuello de botella |
| Ruta reproducible | Mostrar trazabilidad | Código -> run -> datos -> figura -> libro |
| Postal editorial | Fijar una idea | Final de parte, lecciones, glosario |

Reglas anti-monotonía:

- No repetir el mismo arquetipo en más de dos cierres consecutivos.
- Si una figura solo enumera, añadir relación visual: eje, flecha, matriz, comparación o mini-ejemplo.
- Si explica mecanismo interno, priorizar radiografía, flujo o microscopio.
- Si explica resultados, priorizar gráfica, matriz o dashboard.
- Si explica una historia causal, usar antes/después, diagnóstico o storyboard.
- Si cierra una parte, usar postal editorial o mapa de sistema.
- Antes de crear un lote, hacer inventario de arquetipos.

El objetivo es que el lector note valor nuevo en cada figura.

## Estilo visual recomendado

Paleta funcional:

- Azul: ruta, idea base, estructura.
- Cian: sistema, interfaz, flujo real.
- Violeta: mecanismo técnico, fórmula, modelo.
- Verde: evidencia, validación, reproducibilidad.
- Ámbar/naranja: qué mirar, advertencia suave.
- Rojo/rosa: error, colapso, trampa conceptual.
- Gris/pizarra: límite, experto, formalización.

Reglas:

- No usar colores libres.
- No usar gradientes como relleno decorativo de páginas técnicas.
- Reservar sombras para separar niveles, no para embellecer.
- Las tarjetas deben tener radio moderado.
- La figura debe ser legible en PDF a tamaño de página.
- Si el contenido es denso, preferir PNG renderizado desde HTML/SVG/canvas con control tipográfico.

## Texto dentro de imágenes

Reglas estrictas:

- No meter párrafos largos en tarjetas.
- Frases de 3-8 palabras cuando sea posible.
- Partir manualmente títulos y etiquetas largas.
- Comprobar que ninguna palabra larga rompe caja.
- No dejar fórmulas o etiquetas pegadas a bordes.
- No confiar en auto-wrap si se va a exportar a PNG/PDF.
- El caption puede explicar más; la figura debe llevar lo visual.

Mal:

```text
La formulación puso el techo; el algoritmo solo pudo subir cuando el techo dejó de estar cerrado.
```

Mejor:

```text
La formulación puso el techo.
El algoritmo subió cuando el techo se abrió.
```

## Tipos de assets aceptados

Elegir formato por calidad, no por costumbre:

- SVG para diagramas controlados.
- HTML/CSS/canvas renderizado a PNG para infografías densas.
- PNG para composiciones editoriales.
- Capturas reales anotadas.
- Gráficas replotteadas desde datos.
- Tablas visuales.
- Imágenes generadas si aportan más claridad que un vector manual.

Siempre conservar, si es razonable:

- Fuente editable del asset.
- PNG final usado por el HTML.
- Registro de cómo se generó.

## Capturas reales

Conservar capturas cuando:

- Enseñan el producto real.
- Muestran un inspector, dashboard, modelo en acción o resultado observable.
- Ayudan al lector a ubicarse.

Pero deben ir acompañadas de:

- Callouts.
- Caption con “qué mirar”.
- Límite explícito: una captura no es evidencia estadística.
- Relación con el resultado protocolizado.

## Gráficos de datos

Siempre que sea posible, recrear gráficos desde datos fuente.

Un gráfico aceptable debe incluir:

- Título interpretativo.
- Unidad.
- Contexto experimental.
- Split o conjunto.
- Semillas o tamaño de muestra.
- Dispersión si existe.
- Etiquetas legibles.
- Caption con lectura y límite.

Preguntas obligatorias:

- ¿Qué demuestra?
- ¿Qué no demuestra?
- ¿Cuál es la métrica?
- ¿Cuál es el protocolo?
- ¿Hay varianza, colapsos o incertidumbre?
- ¿Qué lectura equivocada evita?

## QA visual obligatorio

Una imagen no está terminada hasta pasar QA.

Rutina por lote:

1. Renderizar SVG/HTML/figura a PNG.
2. Crear hoja de contacto.
3. Revisar a tamaño completo.
4. Corregir overflow, solapes, cortes, jerarquía, escalas y captions.
5. Re-renderizar.
6. Integrar en HTML.
7. Comprobar rutas de imágenes.
8. Generar PDF cuando toque.
9. Renderizar páginas PDF afectadas.
10. Revisar visualmente esas páginas.
11. Registrar incidencias y páginas revisadas.

Comprobaciones mínimas:

```bash
# Imágenes referenciadas por el HTML
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.src = []
    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.src.append(dict(attrs).get("src", ""))

html = Path("docs/report_vN.html")
p = P()
p.feed(html.read_text())
missing = [s for s in p.src if s and not Path("docs", s).exists()]
print({"images": len(p.src), "missing": len(missing), "missing_paths": missing})
PY
```

```bash
# Restos de versiones antiguas
rg -n 'version-codex|v3|figura_antigua|imagen_obsoleta|TODO' docs/report_vN.html
```

Para PDF:

```bash
node scripts/generarPDF.mjs docs/report_vN.html docs/<Proyecto>-vN.pdf
```

Renderizar páginas clave con `pypdfium2`, Ghostscript, Playwright o herramienta equivalente. Revisar:

- Portada.
- Primeras páginas de cada parte.
- Páginas con infografías nuevas.
- Páginas con código.
- Anexos.
- Última página.

## QA textual del PDF

Después de generar PDF:

- Contar páginas.
- Verificar tamaño.
- Extraer texto y buscar nombres/versiones incorrectas.
- Confirmar que aparecen anexos.
- Confirmar que no aparece una marca antigua.

Ejemplo:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from pypdf import PdfReader

pdf = Path("docs/<Proyecto>-vN.pdf")
r = PdfReader(str(pdf))
text = "\n".join((p.extract_text() or "") for p in r.pages)
print("pages", len(r.pages))
print("size_mb", round(pdf.stat().st_size / 1024 / 1024, 2))
for needle in ["Edición N", "ANEXO A", "ANEXO B", "ANEXO C", "ANEXO D", "versión antigua"]:
    print(needle, text.count(needle))
PY
```

## README y publicación

Cuando el PDF final sea la versión buena:

1. Actualizar README para apuntar al PDF nuevo.
2. Regenerar portada y preview del PDF si el README las muestra.
3. Cambiar número de páginas y tamaño aproximado.
4. Eliminar referencias a versiones antiguas.
5. Mover PDFs, HTML y planes antiguos a `docs/Back/`.
6. Mantener en `docs/assets/` solo assets usados por la versión publicada o por el README.
7. Ignorar o no subir QA pesada/regenerable.
8. Ejecutar build/test del proyecto si aplica.
9. Commit y push si el usuario lo pide.

Regla de claridad: al abrir `docs/`, una persona nueva debe ver la versión buena sin dudar.

## Registro persistente

Mantener un seguimiento, por ejemplo `docs/Back/seguimiento_manual.md`.

Debe incluir:

- Objetivo.
- Documento original.
- Documento de trabajo.
- Criterios de aceptación.
- Plan por capítulos.
- Assets creados.
- Incidencias corregidas.
- Páginas PDF revisadas.
- Decisiones de publicación.
- Log cronológico.

Estados recomendados:

- Pendiente.
- En curso.
- QA PNG completado.
- Integrado.
- Validado en PDF.
- Publicado.

No marcar “validado” si no se ha visto en PDF, salvo que el usuario haya pedido explícitamente no generar PDF todavía.

## Criterios de aceptación final

El libro está listo cuando:

- Existe una versión final separada del original.
- El original o versiones previas están en `Back/`.
- README apunta al PDF final.
- La portada y preview del README corresponden al PDF final.
- Todas las imágenes referenciadas existen.
- No hay textos fuera de cajas.
- No hay solapes o cortes.
- Las capturas tienen captions útiles.
- Las gráficas tienen contexto y límites.
- Los anexos permiten usar y auditar el proyecto.
- El PDF fue generado y revisado visualmente.
- El build/test del proyecto pasa, si aplica.
- El estado Git no incluye cambios inesperados.

## Checklist final

Antes de entregar:

```bash
# 1. Rutas del HTML
python3 <script_de_rutas>

# 2. Referencias antiguas
rg -n 'v3|version-codex|imagen_obsoleta|TODO' README.md docs/report_vN.html

# 3. PDF
ls -lh docs/*vN.pdf

# 4. Páginas y texto PDF
.venv/bin/python <script_pdf_qa>

# 5. Build/test
npm run build   # o el comando equivalente del proyecto

# 6. Estado Git
git status --short --branch
```

En la respuesta final incluir:

- PDF final.
- HTML fuente.
- Qué anexos se añadieron.
- Qué se movió a `Back/`.
- Qué QA se ejecutó.
- Commit/push si se hizo.
- Pendientes reales, si existen.

## Principios pedagógicos

Recordar siempre:

- Una explicación buena es lineal antes de ser completa.
- Un junior necesita saber qué mirar antes de mirar.
- Una cifra sin protocolo no es evidencia.
- Una captura sin lectura es decoración.
- Una fórmula sin intuición previa es una barrera.
- Un gráfico sin límite invita a exagerar.
- Un capítulo largo necesita checkpoints visuales.
- La homogeneidad reduce carga cognitiva.
- La variedad mantiene atención y permite explicar relaciones distintas.
- La imagen debe descargar memoria de trabajo, no añadir ruido.

## Prompts reutilizables

### Auditoría inicial

```text
Lee el documento base y el proyecto. Genera una auditoría pedagógica para un ingeniero junior.
Indica por capítulo si la explicación es lineal, qué habría que ampliar y qué visuales ayudarían.
No modifiques el original. Crea una versión de trabajo separada y un registro de seguimiento.
```

### Plan visual

```text
Crea un plan visual completo por capítulos.
Para cada capítulo indica qué imágenes conservarías, cuáles sustituirías, qué visuales nuevos faltan y qué función pedagógica tendrá cada uno.
Incluye infografías de cierre con fondo blanco, pero evita repetir siempre la misma plantilla.
Usa arquetipos variados: mapa, radiografía, flujo, storyboard, matriz, dashboard, gráfica explicada, microscopio de fórmula, diagnóstico y ruta reproducible.
```

### Ejecución por lotes

```text
Ejecuta el plan por lotes.
Por cada lote: crea visuales, renderízalos a PNG, revisa hoja de contacto, corrige desbordes, integra en HTML, comprueba rutas y actualiza seguimiento.
No generes PDF hasta que termine el lote acordado o hasta que lo autorice explícitamente.
```

### Anexos finales

```text
Añade anexos finales para que el manual sea útil fuera de la lectura:
instalación y ejecución, mapa del código, fragmentos de código esenciales, glosario operativo, lecturas recomendadas, licencia y autoría.
No reescribas todo el libro si basta con añadir anexos al final.
```

### Publicación

```text
Genera el PDF final, renderiza páginas clave y revisa que no haya textos fuera de caja, cortes, captions perdidos ni referencias antiguas.
Actualiza README al PDF final, regenera portada/preview, mueve versiones antiguas a Back, ejecuta build/test y prepara commit.
```
