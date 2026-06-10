from __future__ import annotations

from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BACK = DOCS / "Back"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


EXTRA_CSS = r"""

  /* ---------- v2: capa editorial pedagogica ---------- */
  .edition-ribbon{
    position:absolute; top:18mm; right:22mm; padding:7px 12px;
    border:1px solid rgba(255,255,255,.35); border-radius:999px;
    background:rgba(255,255,255,.14); color:#fff; font-size:9pt; font-weight:800;
    letter-spacing:.6px; text-transform:uppercase;
  }
  .reader-note{ border:1px solid var(--line); border-left:5px solid var(--accent,#2563eb);
    border-radius:13px; padding:13px 16px; margin:12px 0; background:#fff; }
  .reader-note .eyebrow{ text-transform:uppercase; letter-spacing:.7px; color:var(--accent,#2563eb);
    font-size:7.5pt; font-weight:800; margin-bottom:4px; }
  .reader-note .big{ font-size:14pt; line-height:1.25; font-weight:800; margin-bottom:6px; }
  .reading-map{ display:grid; grid-template-columns:repeat(4,1fr); gap:9px; margin:12px 0; }
  .reading-map .node{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px 11px; min-height:34mm; }
  .reading-map .n{ display:inline-grid; place-items:center; width:22px; height:22px; border-radius:999px;
    background:var(--accent-soft,#eff4fe); color:var(--accent,#2563eb); font-family:"SFMono-Regular",Menlo,monospace;
    font-weight:800; font-size:8pt; margin-bottom:6px; }
  .reading-map .h{ font-weight:800; font-size:9.8pt; margin-bottom:3px; }
  .reading-map p{ font-size:9.1pt; color:var(--slate); margin:0; }
  .chapter-frame{ display:grid; grid-template-columns:1.08fr .92fr; gap:12px; align-items:stretch; margin:10px 0 14px; }
  .chapter-frame .question{ border-radius:14px; padding:14px 16px; background:var(--accent-soft,#eff4fe);
    border:1px solid rgba(37,99,235,.15); }
  .chapter-frame .question .label{ font-size:8pt; text-transform:uppercase; letter-spacing:.7px;
    color:var(--accent,#2563eb); font-weight:800; margin-bottom:5px; }
  .chapter-frame .question .q{ font-size:15pt; line-height:1.22; font-weight:800; margin-bottom:8px; }
  .chapter-frame .answer{ display:grid; gap:8px; }
  .chapter-frame .answer .mini{ border:1px solid var(--line); border-radius:12px; background:#fff; padding:9px 11px; }
  .chapter-frame .answer .mini b{ color:var(--ink); }
  .concept-flow{ display:flex; align-items:stretch; gap:8px; margin:12px 0; }
  .concept-flow .step{ flex:1; border:1px solid var(--line); border-radius:12px; padding:10px; background:#fff; text-align:center; }
  .concept-flow .step .top{ font-size:8pt; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; font-weight:800; }
  .concept-flow .step .mid{ font-size:12pt; font-weight:800; margin:4px 0; color:var(--accent,#2563eb); }
  .concept-flow .step .bot{ font-size:8.6pt; color:var(--slate); }
  .concept-flow .arr{ align-self:center; font-size:14pt; color:var(--muted); }
  .micro-figure{ border:1px solid var(--line); border-radius:14px; padding:13px 15px; background:#fff; margin:12px 0;
    break-inside:avoid; page-break-inside:avoid; }
  .micro-figure .fig-title{ font-weight:800; font-size:11pt; margin-bottom:7px; color:var(--ink); }
  .micro-figure .fig-caption{ font-size:8.8pt; color:var(--muted); margin-top:8px; text-align:center; }
  .svg-wrap{ display:block; width:100%; height:auto; }
  .metric-guide{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:9px; margin:11px 0; }
  .metric-guide .m{ border:1px solid var(--line); border-radius:12px; padding:10px 11px; background:#fff; }
  .metric-guide .m .k{ font-family:"SFMono-Regular",Menlo,monospace; color:var(--accent,#2563eb); font-weight:800; font-size:10pt; }
  .metric-guide .m p{ font-size:9.1pt; margin:3px 0 0; color:var(--slate); }
  .two-col{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:11px 0; }
  .check-card,.trap-card,.limit-card{ border-radius:12px; padding:12px 14px; break-inside:avoid; page-break-inside:avoid; }
  .check-card{ background:#e9fbf4; border:1px solid #c7eede; }
  .trap-card{ background:#fdeef6; border:1px solid #f6cfe3; }
  .limit-card{ background:#fff8ec; border:1px solid #f6e2bd; }
  .check-card .h,.trap-card .h,.limit-card .h{ font-weight:800; margin-bottom:5px; }
  .check-card .h{ color:#0a6b4e; } .trap-card .h{ color:#a41d54; } .limit-card .h{ color:#7a5408; }
  .selfcheck{ border:1px solid var(--line); border-radius:13px; padding:12px 15px; background:#f8fafc; margin:12px 0;
    break-inside:avoid; page-break-inside:avoid; }
  .selfcheck .h{ font-weight:800; font-size:10.5pt; margin-bottom:6px; }
  .selfcheck ol{ margin:0; padding-left:18px; color:var(--slate); }
  .selfcheck li{ margin:4px 0; }
  .timeline-v2{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:12px 0; }
  .timeline-v2 .item{ border:1px solid var(--line); border-radius:12px; padding:10px; background:#fff; min-height:35mm; }
  .timeline-v2 .item .tag2{ font-size:7.5pt; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); font-weight:800; }
  .timeline-v2 .item .name{ font-size:11pt; color:var(--accent,#2563eb); font-weight:800; margin:3px 0; }
  .timeline-v2 .item p{ font-size:8.8pt; margin:0; color:var(--slate); }
  .annex-index{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:9px 0; }
  .annex-index .a{ border:1px solid var(--line); border-radius:12px; padding:8px 9px; background:#fff; }
  .annex-index .a .letter{ font-family:"SFMono-Regular",Menlo,monospace; color:var(--accent,#2563eb); font-weight:800; }
  .annex-index .a .name{ font-weight:800; margin:2px 0; }
  .annex-index .a p{ color:var(--slate); font-size:8.25pt; margin:0; line-height:1.35; }
  .glossary{ display:grid; grid-template-columns:1fr 1fr; gap:8px 12px; margin:10px 0; }
  .glossary .g{ border-bottom:1px solid #eef0f4; padding:6px 0 7px; }
  .glossary .g b{ color:var(--accent,#2563eb); font-family:"SFMono-Regular",Menlo,monospace; font-size:9.1pt; }
  .glossary .g span{ color:var(--slate); }
  .poster{ border:1px solid var(--line); border-radius:16px; padding:15px 17px; background:#fff; margin:14px 0;
    break-inside:avoid; page-break-inside:avoid; }
  .poster h3{ font-size:18pt; color:var(--ink); margin-bottom:8px; }
  .poster-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
  .poster-grid .p{ border-radius:12px; padding:11px; background:#f8fafc; border:1px solid #edf0f4; }
  .poster-grid .p .h{ font-weight:800; color:var(--accent,#2563eb); margin-bottom:4px; }
  .poster-grid .p p{ margin:0; color:var(--slate); font-size:9pt; }
  .caption-check{ color:var(--muted); font-size:8.7pt; text-align:center; margin:-3px 0 8px; }
  @media print{
    .reading-map{ grid-template-columns:repeat(4,1fr); }
    .chapter-frame,.two-col{ grid-template-columns:1fr 1fr; }
    .annex-index{ grid-template-columns:repeat(4,1fr); }
    .poster,.reader-note,.micro-figure,.selfcheck{ break-inside:avoid; page-break-inside:avoid; }
  }
"""


INTRO_V2 = r"""

<!-- ============ V2 · COMO LEER ============ -->
<section class="section">
  <div class="band b-blue"><div class="n">VERSIÓN 2 · GUÍA DE LECTURA</div><h2>Cómo leer este manual</h2>
    <div class="tag">La ruta: intuición → mecanismo → evidencia → límites → reproducción</div></div>
  <div class="pad" style="--accent:#2563eb;--accent-soft:#eff4fe">
    <div class="reader-note">
      <div class="eyebrow">Promesa pedagógica</div>
      <div class="big">No vamos a memorizar nombres de modelos: vamos a entender qué problema resuelve cada uno y qué precio paga por resolverlo.</div>
      <p>El laboratorio está hecho para un ingeniero junior de IA que quiere ver las ideas funcionando. Por eso cada capítulo sigue la misma gramática: empieza con una pregunta, baja a la intuición, abre la caja del modelo, traduce la fórmula al código real, enseña una captura o resultado y termina con límites y preguntas de autocomprobación.</p>
    </div>

    <h3 class="sub">El patrón que se repite en todos los capítulos</h3>
    <div class="reading-map">
      <div class="node"><span class="n">1</span><div class="h">Pregunta</div><p>¿Qué carencia del modelo anterior intenta resolver?</p></div>
      <div class="node"><span class="n">2</span><div class="h">Mecanismo</div><p>Qué aprende, qué entra, qué sale y dónde está la restricción.</p></div>
      <div class="node"><span class="n">3</span><div class="h">Evidencia</div><p>Capturas, curvas y métricas con una lectura explícita.</p></div>
      <div class="node"><span class="n">4</span><div class="h">Límite</div><p>Qué no demuestra la demo y qué error conceptual evitar.</p></div>
    </div>

    <h3 class="sub" style="--accent:#0c9f6e">Mapa mental del viaje</h3>
    <div class="timeline-v2" style="--accent:#0c9f6e">
      <div class="item"><div class="tag2">Base</div><div class="name">AE</div><p>Aprende una compresión útil. Enseña cuello de botella, reconstrucción y espacio latente.</p></div>
      <div class="item"><div class="tag2">Primer generador</div><div class="name">VAE</div><p>Convierte puntos en nubes para que el latente se pueda muestrear.</p></div>
      <div class="item"><div class="tag2">Competición</div><div class="name">GAN</div><p>Cambia el error por píxel por un juez aprendido: gana nitidez y pierde estabilidad.</p></div>
      <div class="item"><div class="tag2">Iteración</div><div class="name">Diffusion</div><p>Genera quitando ruido poco a poco: calidad y control a cambio de más pasos.</p></div>
      <div class="item"><div class="tag2">Síntesis</div><div class="name">Comparador</div><p>Pone capacidad, coste, estabilidad e interpretabilidad en la misma mesa.</p></div>
    </div>

    <h3 class="sub" style="--accent:#d97706">Qué mirar en las capturas</h3>
    <p>Una captura no es una prueba estadística; es una brújula visual. En cada imagen del laboratorio conviene leer tres capas: primero el flujo de datos, luego los controles que cambian el modelo y por último las señales de evidencia (pérdida, métrica, mapa latente o muestras). Cuando una figura muestra resultados, el caption aclara qué demuestra y qué lectura sería exagerada.</p>

  </div>
</section>

<!-- ============ V2 · DATOS Y LENGUAJE COMUN ============ -->
<section class="section">
  <div class="band b-ink"><div class="n">ANTES DE LOS MODELOS</div><h2>Datos, tensores y espacio latente</h2>
    <div class="tag">El vocabulario mínimo para que las cuatro pestañas tengan sentido</div></div>
  <div class="pad" style="--accent:#1f2937;--accent-soft:#f1f3f6">
    <h3 class="sub" style="--accent:#2563eb">Qué es una imagen para una red neuronal</h3>
    <p>Para nosotros una cara es una expresión, un peinado o un estilo. Para la red es un tensor: una tabla de números. En este proyecto cada imagen tiene 64×64 píxeles y 3 canales de color, así que cada ejemplo entra como 12.288 valores entre 0 y 1. La red no recibe etiquetas ni nombres de rasgos; solo recibe esa matriz de números y una tarea de aprendizaje.</p>
    <div class="micro-figure" style="--accent:#2563eb">
      <div class="fig-title">Figura guía: de píxeles a aprendizaje no supervisado</div>
      <svg class="svg-wrap" viewBox="0 0 900 210" role="img" aria-label="Flujo de imagen a tensor y objetivo">
        <rect x="20" y="42" width="135" height="135" rx="18" fill="#EAF1FE" stroke="#2563EB"/>
        <g fill="#2563EB" opacity=".95">
          <circle cx="60" cy="88" r="14"/><circle cx="114" cy="88" r="14"/>
          <path d="M54 128 Q88 156 122 128" fill="none" stroke="#2563EB" stroke-width="8" stroke-linecap="round"/>
          <path d="M38 70 Q88 20 138 70" fill="none" stroke="#2563EB" stroke-width="10" stroke-linecap="round"/>
        </g>
        <text x="87" y="197" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">Cara 64×64</text>
        <path d="M170 108 H255" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowv2)"/>
        <rect x="275" y="44" width="170" height="132" rx="14" fill="#fff" stroke="#EDEFF4"/>
        <text x="360" y="78" text-anchor="middle" font-size="18" font-weight="800" fill="#111827">Tensor</text>
        <text x="360" y="108" text-anchor="middle" font-size="24" font-family="monospace" fill="#2563EB">64×64×3</text>
        <text x="360" y="140" text-anchor="middle" font-size="16" fill="#5B6472">12.288 valores</text>
        <path d="M460 108 H545" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowv2)"/>
        <rect x="565" y="44" width="145" height="132" rx="14" fill="#F1ECFC" stroke="#8B5CF6"/>
        <text x="637" y="82" text-anchor="middle" font-size="18" font-weight="800" fill="#111827">Modelo</text>
        <text x="637" y="116" text-anchor="middle" font-size="16" fill="#5B6472">aprende</text>
        <text x="637" y="144" text-anchor="middle" font-size="16" fill="#5B6472">estructura</text>
        <path d="M724 108 H810" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowv2)"/>
        <rect x="825" y="44" width="55" height="132" rx="14" fill="#E5F6F0" stroke="#10B981"/>
        <text x="852" y="104" text-anchor="middle" font-size="20" font-family="monospace" font-weight="800" fill="#10B981">z</text>
        <text x="852" y="130" text-anchor="middle" font-size="12" fill="#5B6472">latente</text>
        <defs><marker id="arrowv2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8A93A2"/></marker></defs>
      </svg>
      <div class="fig-caption">Qué demuestra: todos los modelos comparten el mismo material de partida. Qué no demuestra: que el modelo “entienda” ojos o pelo como conceptos humanos; solo aprende regularidades útiles en los píxeles.</div>
    </div>

    <h3 class="sub" style="--accent:#8B5CF6">Latente no significa mágico</h3>
    <p>El espacio latente es una representación comprimida. Si dos imágenes acaban cerca en ese espacio, significa que el modelo las usa de forma parecida para su tarea. En el AE y VAE esa tarea incluye reconstruir; en la GAN el latente es ruido de entrada; en difusión no hay un latente compacto equivalente, sino una trayectoria de ruido. Esta diferencia evita una confusión habitual: no todos los modelos generativos “tienen el mismo tipo de z”.</p>
    <table class="compare">
      <tr><th>Modelo</th><th>Qué representa su espacio interno</th><th>Cómo leerlo en la app</th><th>Riesgo de interpretación</th></tr>
      <tr><td>AE</td><td>Resumen determinista de una imagen</td><td>Mapa latente, vecinos, interpolación</td><td>Confundir reconstruir bien con generar bien</td></tr>
      <tr><td>VAE</td><td>Distribución ordenada alrededor de cada imagen</td><td>Muestreo del prior y traversal</td><td>Pedir nitidez de GAN a un modelo regularizado</td></tr>
      <tr><td>GAN</td><td>Ruido que el generador transforma en imagen</td><td>Semilla, truncation, interpolación</td><td>Olvidar que no hay encoder ni reconstrucción</td></tr>
      <tr><td>Diffusion</td><td>Estados intermedios de una cadena de denoising</td><td>Forward/reverse filmstrip</td><td>Creer que genera de una sola pasada</td></tr>
    </table>

    <div class="selfcheck">
      <div class="h">Autocomprobación antes de seguir</div>
      <ol>
        <li>¿Cuántos números tiene una imagen de entrada y por qué eso importa?</li>
        <li>¿Qué diferencia hay entre “un punto z” y “una nube alrededor de z”?</li>
        <li>¿Por qué una captura bonita no sustituye a una métrica o a un protocolo?</li>
      </ol>
    </div>
  </div>
</section>
"""


AE_FRAME = r"""

    <div class="chapter-frame">
      <div class="question">
        <div class="label">Pregunta del capítulo</div>
        <div class="q">¿Qué tiene que aprender una red para copiar una cara usando mucha menos información?</div>
        <p>El Autoencoder es el laboratorio del cuello de botella: si el resumen es pequeño, la red debe decidir qué rasgos sobreviven y qué detalles se pierden.</p>
      </div>
      <div class="answer">
        <div class="mini"><b>Entrada:</b> una imagen 64×64×3.</div>
        <div class="mini"><b>Salida:</b> la misma imagen reconstruida.</div>
        <div class="mini"><b>Señal de aprendizaje:</b> error entre original y reconstrucción.</div>
        <div class="mini"><b>Lección:</b> comprimir no es guardar todo; es elegir estructura.</div>
      </div>
    </div>

    <div class="micro-figure" style="--accent:#2563eb">
      <div class="fig-title">Microscopio del cuello de botella</div>
      <div class="concept-flow">
        <div class="step"><div class="top">imagen</div><div class="mid">12.288</div><div class="bot">valores RGB</div></div>
        <div class="arr">→</div>
        <div class="step"><div class="top">encoder</div><div class="mid">reduce</div><div class="bot">64→32→16→8→4</div></div>
        <div class="arr">→</div>
        <div class="step" style="border-color:#F59E0B;background:#FCEFD7"><div class="top">latente</div><div class="mid" style="color:#D97706">128</div><div class="bot">números por defecto</div></div>
        <div class="arr">→</div>
        <div class="step"><div class="top">decoder</div><div class="mid">expande</div><div class="bot">4→8→16→32→64</div></div>
      </div>
      <div class="fig-caption">Lectura: la red comprime unas 96 veces. Si la reconstrucción conserva expresión, color y silueta, el latente ha capturado estructura; si pierde pelo fino, es el precio de la compresión.</div>
    </div>
"""


AE_WRAP = r"""

    <div class="metric-guide" style="--accent:#2563eb">
      <div class="m"><div class="k">MSE / L1</div><p>Miden error píxel a píxel. Son útiles para reconstrucción, pero favorecen promedios suaves cuando hay ambigüedad.</p></div>
      <div class="m"><div class="k">PSNR</div><p>Traduce el error a decibelios. En este proyecto ayuda a comparar variantes AE con el mismo protocolo.</p></div>
      <div class="m"><div class="k">SSIM</div><p>Mira estructura visual: bordes, contraste y formas. Complementa al error numérico por píxel.</p></div>
    </div>
    <div class="two-col">
      <div class="trap-card"><div class="h">Error común</div>“La U-Net entiende mejor el latente porque reconstruye mejor”. En realidad sus atajos llevan detalle alrededor del cuello; por eso reconstruye nítido, pero parte de la información no pasa por z.</div>
      <div class="limit-card"><div class="h">Límite honesto</div>PSNR y SSIM miden fidelidad de reconstrucción, no creatividad. Un AE puede tener buenas métricas y aun así no ser un buen generador desde el prior.</div>
    </div>
    <div class="selfcheck">
      <div class="h">Autocomprobación AE</div>
      <ol>
        <li>¿Por qué aumentar parámetros no solucionó la borrosidad de la variante básica?</li>
        <li>¿Qué diferencia hay entre distancia en z y distancia en la proyección 2D?</li>
        <li>¿Qué esperarías que ocurra si subes latent_dim de 128 a 512?</li>
      </ol>
    </div>
"""


VAE_FRAME = r"""

    <div class="chapter-frame">
      <div class="question">
        <div class="label">Pregunta del capítulo</div>
        <div class="q">¿Cómo pasamos de reconstruir caras conocidas a generar caras nuevas?</div>
        <p>El VAE conserva el encoder-decoder, pero cambia la geometría del latente: cada imagen deja de ser un punto exacto y se convierte en una distribución.</p>
      </div>
      <div class="answer">
        <div class="mini"><b>Entrada:</b> imagen real.</div>
        <div class="mini"><b>Salida:</b> reconstrucción y muestras nuevas.</div>
        <div class="mini"><b>Objetivo:</b> recon_loss + β·KL.</div>
        <div class="mini"><b>Lección:</b> generar exige ordenar el espacio, no solo memorizar.</div>
      </div>
    </div>

    <div class="micro-figure" style="--accent:#8B5CF6">
      <div class="fig-title">De puntos sueltos a nubes muestreables</div>
      <svg class="svg-wrap" viewBox="0 0 900 260" role="img" aria-label="Comparación AE VAE latente">
        <rect x="35" y="35" width="360" height="180" rx="18" fill="#fff" stroke="#EDEFF4"/>
        <text x="215" y="66" text-anchor="middle" font-size="20" font-weight="800" fill="#111827">AE: puntos</text>
        <g fill="#2563EB">
          <circle cx="115" cy="125" r="8"/><circle cx="180" cy="92" r="8"/><circle cx="245" cy="156" r="8"/><circle cx="305" cy="108" r="8"/>
        </g>
        <path d="M92 178 C150 138, 220 196, 325 145" fill="none" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="6 7"/>
        <text x="215" y="204" text-anchor="middle" font-size="15" fill="#5B6472">Entre puntos puede haber huecos</text>
        <rect x="505" y="35" width="360" height="180" rx="18" fill="#fff" stroke="#EDEFF4"/>
        <text x="685" y="66" text-anchor="middle" font-size="20" font-weight="800" fill="#111827">VAE: nubes</text>
        <g>
          <circle cx="605" cy="125" r="48" fill="#F1ECFC" stroke="#8B5CF6"/><circle cx="670" cy="103" r="48" fill="#F1ECFC" stroke="#8B5CF6"/>
          <circle cx="735" cy="150" r="48" fill="#F1ECFC" stroke="#8B5CF6"/><circle cx="782" cy="112" r="48" fill="#F1ECFC" stroke="#8B5CF6"/>
          <circle cx="605" cy="125" r="6" fill="#8B5CF6"/><circle cx="670" cy="103" r="6" fill="#8B5CF6"/><circle cx="735" cy="150" r="6" fill="#8B5CF6"/><circle cx="782" cy="112" r="6" fill="#8B5CF6"/>
        </g>
        <text x="685" y="204" text-anchor="middle" font-size="15" fill="#5B6472">El prior N(0,I) cae en zonas habitadas</text>
      </svg>
      <div class="fig-caption">Qué mirar: el término KL no “mejora la reconstrucción” directamente; regulariza el mapa para que muestrear tenga sentido.</div>
    </div>
"""


VAE_WRAP = r"""

    <div class="two-col">
      <div class="trap-card"><div class="h">Error común</div>“Si β sube, el VAE siempre mejora”. No: β alto ordena más el latente, pero suele empeorar la nitidez. β es un mando de compromiso, no de calidad absoluta.</div>
      <div class="limit-card"><div class="h">Límite honesto</div>Un VAE genera diversidad razonable, pero su pérdida por reconstrucción y la regularización KL tienden a suavizar detalles. No está diseñado para ganar en nitidez a una GAN o a difusión.</div>
    </div>
    <div class="selfcheck">
      <div class="h">Autocomprobación VAE</div>
      <ol>
        <li>¿Por qué el encoder devuelve μ y logσ² en vez de un único z?</li>
        <li>¿Qué papel cumple el truco de reparametrización durante el entrenamiento?</li>
        <li>¿Qué sacrificas al aumentar β y qué ganas?</li>
      </ol>
    </div>
"""


GAN_FRAME = r"""

    <div class="chapter-frame">
      <div class="question">
        <div class="label">Pregunta del capítulo</div>
        <div class="q">¿Y si dejamos de comparar píxeles y entrenamos un juez que detecte falsificaciones?</div>
        <p>La GAN cambia la señal de aprendizaje. Ya no busca parecerse a una imagen concreta: busca producir imágenes que un discriminador no pueda distinguir de las reales.</p>
      </div>
      <div class="answer">
        <div class="mini"><b>Entrada:</b> ruido z.</div>
        <div class="mini"><b>Salida:</b> imagen generada.</div>
        <div class="mini"><b>Dos redes:</b> generador y discriminador.</div>
        <div class="mini"><b>Lección:</b> nitidez y estabilidad están en tensión.</div>
      </div>
    </div>

    <div class="micro-figure" style="--accent:#EC4899">
      <div class="fig-title">El duelo adversarial en una sola mirada</div>
      <svg class="svg-wrap" viewBox="0 0 900 250" role="img" aria-label="Bucle GAN">
        <rect x="55" y="82" width="160" height="82" rx="16" fill="#FCE9F2" stroke="#EC4899"/>
        <text x="135" y="114" text-anchor="middle" font-size="20" font-weight="800" fill="#111827">Ruido z</text>
        <text x="135" y="142" text-anchor="middle" font-size="15" fill="#5B6472">semilla</text>
        <path d="M230 123 H315" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowgan)"/>
        <rect x="335" y="66" width="180" height="114" rx="18" fill="#fff" stroke="#EC4899"/>
        <text x="425" y="103" text-anchor="middle" font-size="22" font-weight="800" fill="#111827">Generador</text>
        <text x="425" y="134" text-anchor="middle" font-size="15" fill="#5B6472">aprende a engañar</text>
        <path d="M530 123 H610" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowgan)"/>
        <rect x="630" y="66" width="200" height="114" rx="18" fill="#fff" stroke="#EC4899"/>
        <text x="730" y="103" text-anchor="middle" font-size="22" font-weight="800" fill="#111827">Discriminador</text>
        <text x="730" y="134" text-anchor="middle" font-size="15" fill="#5B6472">real o falsa</text>
        <path d="M730 193 C640 235, 470 235, 425 190" fill="none" stroke="#EC4899" stroke-width="3" stroke-dasharray="8 7" marker-end="url(#arrowganpink)"/>
        <text x="570" y="228" text-anchor="middle" font-size="15" fill="#A41D54">la señal vuelve como gradiente</text>
        <defs>
          <marker id="arrowgan" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8A93A2"/></marker>
          <marker id="arrowganpink" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#EC4899"/></marker>
        </defs>
      </svg>
      <div class="fig-caption">Qué demuestra: la GAN aprende con una señal aprendida, no con un objetivo fijo por píxel. Qué no demuestra: que el entrenamiento vaya a converger siempre.</div>
    </div>
"""


GAN_WRAP = r"""

    <div class="two-col">
      <div class="trap-card"><div class="h">Error común</div>“Si D loss baja, todo va bien”. En una GAN no hay una única pérdida que deba bajar siempre: si el discriminador aplasta al generador, la señal de aprendizaje puede dejar de ser útil.</div>
      <div class="limit-card"><div class="h">Límite honesto</div>Las muestras nítidas no garantizan cobertura del dataset. Puede haber colapso de modo: muchas caras bonitas pero poca diversidad real.</div>
    </div>
    <div class="selfcheck">
      <div class="h">Autocomprobación GAN</div>
      <ol>
        <li>¿Por qué una GAN no reconstruye una imagen concreta?</li>
        <li>¿Qué significa que el generador intente que D(fake) parezca real?</li>
        <li>¿Por qué truncation puede mejorar calidad visual reduciendo diversidad?</li>
      </ol>
    </div>
"""


DIFF_FRAME = r"""

    <div class="chapter-frame">
      <div class="question">
        <div class="label">Pregunta del capítulo</div>
        <div class="q">¿Podemos generar una imagen aprendiendo solo a deshacer ruido?</div>
        <p>Difusión convierte la generación en una cadena de pasos pequeños: ensuciar es fácil y conocido; aprender a limpiar cada nivel de ruido hace posible generar desde ruido puro.</p>
      </div>
      <div class="answer">
        <div class="mini"><b>Entrada al muestrear:</b> ruido puro.</div>
        <div class="mini"><b>Red:</b> UNet condicionada en el tiempo t.</div>
        <div class="mini"><b>Objetivo:</b> predecir el ruido ε añadido.</div>
        <div class="mini"><b>Lección:</b> calidad por iteración, no por una sola pasada.</div>
      </div>
    </div>

    <div class="micro-figure" style="--accent:#06B6D4">
      <div class="fig-title">Forward fácil, reverse aprendido</div>
      <svg class="svg-wrap" viewBox="0 0 900 230" role="img" aria-label="Proceso diffusion forward reverse">
        <text x="95" y="42" text-anchor="middle" font-size="18" font-weight="800" fill="#111827">x0</text>
        <text x="805" y="42" text-anchor="middle" font-size="18" font-weight="800" fill="#111827">xT</text>
        <g>
          <rect x="40" y="65" width="110" height="95" rx="16" fill="#E7F7FB" stroke="#06B6D4"/>
          <circle cx="78" cy="105" r="9" fill="#06B6D4"/><circle cx="113" cy="105" r="9" fill="#06B6D4"/><path d="M76 130 Q96 146 118 130" fill="none" stroke="#06B6D4" stroke-width="5" stroke-linecap="round"/>
          <rect x="220" y="65" width="110" height="95" rx="16" fill="#E7F7FB" stroke="#06B6D4" opacity=".9"/>
          <circle cx="258" cy="105" r="9" fill="#06B6D4"/><circle cx="293" cy="105" r="9" fill="#06B6D4"/><path d="M256 130 Q276 146 298 130" fill="none" stroke="#06B6D4" stroke-width="5" stroke-linecap="round"/><circle cx="236" cy="84" r="4" fill="#64748B"/><circle cx="313" cy="145" r="5" fill="#64748B"/>
          <rect x="400" y="65" width="110" height="95" rx="16" fill="#E7F7FB" stroke="#06B6D4" opacity=".75"/>
          <circle cx="437" cy="93" r="5" fill="#64748B"/><circle cx="477" cy="118" r="4" fill="#64748B"/><circle cx="425" cy="138" r="6" fill="#64748B"/><path d="M438 131 Q456 143 479 132" fill="none" stroke="#06B6D4" stroke-width="4" stroke-linecap="round" opacity=".5"/>
          <rect x="580" y="65" width="110" height="95" rx="16" fill="#F8FAFC" stroke="#94A3B8"/>
          <circle cx="608" cy="86" r="6" fill="#64748B"/><circle cx="660" cy="102" r="5" fill="#64748B"/><circle cx="632" cy="139" r="7" fill="#64748B"/><circle cx="675" cy="148" r="4" fill="#64748B"/>
          <rect x="760" y="65" width="110" height="95" rx="16" fill="#F1F5F9" stroke="#94A3B8"/>
          <g fill="#64748B"><circle cx="782" cy="84" r="5"/><circle cx="828" cy="93" r="7"/><circle cx="850" cy="132" r="4"/><circle cx="796" cy="145" r="6"/><circle cx="818" cy="122" r="3"/></g>
        </g>
        <path d="M160 112 H210" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowdiff)"/>
        <path d="M340 112 H390" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowdiff)"/>
        <path d="M520 112 H570" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowdiff)"/>
        <path d="M700 112 H750" stroke="#8A93A2" stroke-width="3" marker-end="url(#arrowdiff)"/>
        <path d="M750 185 H160" stroke="#06B6D4" stroke-width="3" stroke-dasharray="8 7" marker-end="url(#arrowdiffblue)"/>
        <text x="455" y="205" text-anchor="middle" font-size="15" fill="#0891B2">la UNet aprende el camino inverso paso a paso</text>
        <defs>
          <marker id="arrowdiff" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#8A93A2"/></marker>
          <marker id="arrowdiffblue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#06B6D4"/></marker>
        </defs>
      </svg>
      <div class="fig-caption">Qué mirar: en entrenamiento se elige un t aleatorio, se añade ruido y la red predice ε. Al generar, se aplica esa predicción muchas veces en orden inverso.</div>
    </div>
"""


DIFF_WRAP = r"""

    <div class="two-col">
      <div class="trap-card"><div class="h">Error común</div>“Difusión aprende a dibujar directamente una cara”. Más preciso: aprende a estimar ruido. La cara aparece porque repetir esa estimación invierte gradualmente el proceso de degradación.</div>
      <div class="limit-card"><div class="h">Límite honesto</div>Menos pasos DDIM hacen la demo interactiva, pero no son gratis: aceleran el muestreo a costa de calidad o diversidad si se fuerzan demasiado.</div>
    </div>
    <div class="selfcheck">
      <div class="h">Autocomprobación Diffusion</div>
      <ol>
        <li>¿Por qué el forward q(x_t|x0) no necesita aprenderse?</li>
        <li>¿Qué recibe la UNet además de la imagen ruidosa?</li>
        <li>¿Qué cambia al pasar de DDPM a DDIM en una demo interactiva?</li>
      </ol>
    </div>
"""


COMPARATOR_PLUS = r"""

    <div class="poster" style="--accent:#2563eb">
      <h3>La síntesis: cuatro respuestas a una misma pregunta</h3>
      <div class="poster-grid">
        <div class="p"><div class="h">AE</div><p>Si quiero entender compresión y similitud, empiezo aquí.</p></div>
        <div class="p"><div class="h">VAE</div><p>Si necesito un generador interpretable y muestreable, acepto suavidad.</p></div>
        <div class="p"><div class="h">GAN</div><p>Si busco nitidez visual, vigilo estabilidad y diversidad.</p></div>
        <div class="p"><div class="h">Diffusion</div><p>Si busco calidad y proceso explicable, pago coste de muestreo.</p></div>
      </div>
    </div>
    <div class="two-col">
      <div class="check-card"><div class="h">Lectura correcta del comparador</div>No decide “el mejor modelo universal”. Decide qué compromiso encaja con una tarea: reconstruir, generar, explorar latentes, demostrar entrenamiento o comparar estabilidad.</div>
      <div class="trap-card"><div class="h">Lectura equivocada</div>Comparar una reconstrucción AE con una muestra GAN como si fueran la misma tarea. El primero responde “¿puedo copiar esta imagen?”; el segundo “¿puedo inventar una imagen plausible?”.</div>
    </div>
"""


GLOSSARY_SECTION = r"""

<!-- ============ ANEXO D ============ -->
<section class="section">
  <div class="band b-violet"><div class="n">ANEXO D</div><h2>Glosario operativo y lecturas</h2>
    <div class="tag">Para volver al manual semanas después sin perder el hilo</div></div>
  <div class="pad" style="--accent:#8B5CF6;--accent-soft:#F1ECFC">
    <h3 class="sub">Glosario de trabajo</h3>
    <div class="glossary">
      <div class="g"><b>encoder</b> <span>Red que comprime una imagen a una representación interna.</span></div>
      <div class="g"><b>decoder</b> <span>Red que convierte una representación en imagen.</span></div>
      <div class="g"><b>z</b> <span>Vector latente. En AE/VAE resume; en GAN es ruido de entrada.</span></div>
      <div class="g"><b>latent_dim</b> <span>Número de dimensiones del vector z. Más no siempre significa mejor.</span></div>
      <div class="g"><b>MSE</b> <span>Error cuadrático medio. Penaliza fuerte errores grandes y favorece promedios.</span></div>
      <div class="g"><b>L1</b> <span>Error absoluto medio. Suele preservar bordes algo mejor que MSE.</span></div>
      <div class="g"><b>KL</b> <span>Regularización que acerca el posterior del VAE al prior N(0,I).</span></div>
      <div class="g"><b>β-VAE</b> <span>VAE donde β controla reconstrucción frente a latente ordenado.</span></div>
      <div class="g"><b>prior</b> <span>Distribución desde la que muestreamos antes de decodificar.</span></div>
      <div class="g"><b>posterior</b> <span>Distribución latente asociada a una imagen concreta.</span></div>
      <div class="g"><b>truncation</b> <span>Recortar z para favorecer zonas típicas: más calidad media, menos diversidad.</span></div>
      <div class="g"><b>mode collapse</b> <span>Fallo GAN donde muchas semillas producen pocas variantes reales.</span></div>
      <div class="g"><b>schedule</b> <span>Regla que decide cuánto ruido añade difusión en cada paso.</span></div>
      <div class="g"><b>DDIM</b> <span>Muestreador de difusión que permite saltar pasos para hacer demos rápidas.</span></div>
      <div class="g"><b>SSE</b> <span>Server-Sent Events: canal para enviar progreso epoch a epoch al frontend.</span></div>
      <div class="g"><b>MPS</b> <span>Backend de PyTorch para la GPU de Apple Silicon.</span></div>
    </div>

    <h3 class="sub" style="--accent:#EC4899">Diferencias que suelen confundirse</h3>
    <table class="compare">
      <tr><th>Confusión</th><th>Diferencia útil</th></tr>
      <tr><td>Reconstruir vs generar</td><td>Reconstruir parte de una imagen; generar parte de ruido o del prior.</td></tr>
      <tr><td>Latente AE vs latente GAN</td><td>El AE codifica imágenes reales; la GAN transforma ruido en imágenes.</td></tr>
      <tr><td>Captura vs evidencia</td><td>La captura orienta; la evidencia necesita métrica, protocolo y límite.</td></tr>
      <tr><td>Más parámetros vs mejor diseño</td><td>El AE grande mostró que más capacidad no arregla un cuello mal planteado.</td></tr>
      <tr><td>Calidad vs diversidad</td><td>Una galería nítida puede cubrir pocos modos; una galería diversa puede ser menos pulida.</td></tr>
    </table>

    <h3 class="sub" style="--accent:#0c9f6e">Lecturas recomendadas</h3>
    <div class="defs" style="--accent:#0c9f6e">
      <dt>AE/VAE</dt><dd>Kingma & Welling, “Auto-Encoding Variational Bayes”: el origen práctico del truco de reparametrización.</dd>
      <dt>GAN</dt><dd>Goodfellow et al., “Generative Adversarial Nets”: la formulación del juego generador-discriminador.</dd>
      <dt>DCGAN</dt><dd>Radford et al., “Unsupervised Representation Learning with Deep Convolutional GANs”: el patrón convolucional usado aquí.</dd>
      <dt>DDPM</dt><dd>Ho et al., “Denoising Diffusion Probabilistic Models”: el objetivo de predicción de ruido.</dd>
      <dt>DDIM</dt><dd>Song et al., “Denoising Diffusion Implicit Models”: la idea que permite muestrear con menos pasos.</dd>
    </div>

    <div class="selfcheck">
      <div class="h">Preguntas finales para no perder rigor</div>
      <ol>
        <li>¿Qué tarea exacta evalúa cada métrica o captura?</li>
        <li>¿Qué cambia si uso otra seed, otro subconjunto o más epochs?</li>
        <li>¿La visualización demuestra aprendizaje o solo enseña una interfaz funcionando?</li>
        <li>¿Dónde está el compromiso principal del modelo: compresión, orden, estabilidad o coste?</li>
      </ol>
    </div>
  </div>
</section>
"""


def build_report() -> None:
    html = (DOCS / "report.html").read_text(encoding="utf-8")
    html = html.replace("<title>Laboratorio de Modelos Generativos</title>", "<title>Laboratorio de Modelos Generativos · Manual visual v2</title>")
    html = html.replace("</style>", EXTRA_CSS + "\n</style>")
    html = html.replace('<div class="kicker">Recorrido didáctico</div>', '<div class="edition-ribbon">Versión 2</div>\n  <div class="kicker">Manual técnico-pedagógico</div>')
    html = html.replace("<h1>Cómo aprende<br/>una IA a crear<br/>imágenes</h1>", "<h1>Manual visual de<br/>modelos generativos</h1>")
    html = html.replace(
        "Un viaje visual por los cuatro grandes modelos generativos —de lo más sencillo a lo más avanzado— explicado para que se entienda sin necesidad de saber de inteligencia artificial. Cada tecnicismo, aclarado al momento.",
        "Versión 2 ampliada: un recorrido visual, reproducible y pedagógico por Autoencoder, VAE, GAN y Diffusion. Cada capítulo explica qué problema resuelve el modelo, qué mecanismo usa, qué mirar en la interfaz, qué evidencia aporta y qué límite no conviene olvidar.",
    )
    html = html.replace(
        "<span>Laboratorio interactivo · se ejecuta en un portátil normal</span>",
        "<span>Laboratorio interactivo local · edición v2</span>",
    )
    html = html.replace(
        "<span>Aprende con 43.102 caras de dibujo (anime) de 64×64 píxeles</span>",
        "<span>Junio de 2026 · 43.102 caras anime de 64×64</span>",
    )
    html = html.replace("<!-- ============ AUTOENCODER ============ -->", INTRO_V2 + "\n<!-- ============ AUTOENCODER ============ -->")

    html = html.replace('<div class="pad" style="--accent:#2563eb;--accent-soft:#eff4fe">\n\n    <h3 class="sub">¿Qué es un autoencoder?</h3>', '<div class="pad" style="--accent:#2563eb;--accent-soft:#eff4fe">' + AE_FRAME + '\n\n    <h3 class="sub">¿Qué es un autoencoder?</h3>', 1)
    html = html.replace("  </div>\n</section>\n\n<!-- ============ VAE ============ -->", AE_WRAP + "\n  </div>\n</section>\n\n<!-- ============ VAE ============ -->", 1)

    html = html.replace('<div class="pad" style="--accent:#7c3aed;--accent-soft:#f4effe">\n\n    <h3 class="sub">¿Qué es un VAE?</h3>', '<div class="pad" style="--accent:#7c3aed;--accent-soft:#f4effe">' + VAE_FRAME + '\n\n    <h3 class="sub">¿Qué es un VAE?</h3>', 1)
    html = html.replace("  </div>\n</section>\n\n<!-- ============ GAN ============ -->", VAE_WRAP + "\n  </div>\n</section>\n\n<!-- ============ GAN ============ -->", 1)

    html = html.replace('<div class="pad" style="--accent:#db2777;--accent-soft:#fdeef6">\n\n    <h3 class="sub">¿Qué es una GAN?</h3>', '<div class="pad" style="--accent:#db2777;--accent-soft:#fdeef6">' + GAN_FRAME + '\n\n    <h3 class="sub">¿Qué es una GAN?</h3>', 1)
    html = html.replace("  </div>\n</section>\n\n<!-- ============ DIFFUSION ============ -->", GAN_WRAP + "\n  </div>\n</section>\n\n<!-- ============ DIFFUSION ============ -->", 1)

    html = html.replace('<div class="pad" style="--accent:#0891b2;--accent-soft:#e7f7fb">\n\n    <h3 class="sub">¿Qué es un modelo de difusión?</h3>', '<div class="pad" style="--accent:#0891b2;--accent-soft:#e7f7fb">' + DIFF_FRAME + '\n\n    <h3 class="sub">¿Qué es un modelo de difusión?</h3>', 1)
    html = html.replace("  </div>\n</section>\n\n<!-- ============ CIERRE ============ -->", DIFF_WRAP + "\n  </div>\n</section>\n\n<!-- ============ CIERRE ============ -->", 1)

    html = html.replace("    <h3 class=\"sub\" style=\"--accent:#0c9f6e\">Las dos ideas que conviene llevarse</h3>", COMPARATOR_PLUS + "\n    <h3 class=\"sub\" style=\"--accent:#0c9f6e\">Las dos ideas que conviene llevarse</h3>", 1)

    html = html.replace('<div class="band b-green"><div class="n">PONLO EN MARCHA</div><h2>Descárgalo y pruébalo</h2>', '<div class="band b-green"><div class="n">ANEXO A</div><h2>Instalación y ejecución reproducible</h2>', 1)
    html = html.replace('<div class="band b-ink"><div class="n">BAJO EL CAPÓ</div><h2>Cómo está organizado el código</h2>', '<div class="band b-ink"><div class="n">ANEXO B</div><h2>Mapa del código</h2>', 1)
    html = html.replace('<div class="band b-blue"><div class="n">EL CORAZÓN DEL CÓDIGO</div><h2>El código clave, modelo a modelo</h2>', '<div class="band b-blue"><div class="n">ANEXO C</div><h2>Código esencial, modelo a modelo</h2>', 1)
    html = html.replace("<!-- ============ LICENCIA ============ -->", GLOSSARY_SECTION + "\n<!-- ============ LICENCIA ============ -->")
    html = html.replace('<div class="band b-ink"><div class="n">CRÉDITOS</div><h2>Licencia y autoría</h2>', '<div class="band b-ink"><div class="n">LICENCIA Y AUTORÍA</div><h2>Cierre editorial</h2>', 1)
    html = html.replace("Hecho para aprender, compartir y experimentar. ✦", "Hecho para aprender, compartir y experimentar. Versión 2.")

    # Etiquetas de lectura para capturas reales.
    html = html.replace("<figure><img src=\"assets/tab_autoencoder.jpg\" />", "<div class=\"caption-check\">Qué mirar: flujo original-reconstrucción, mapa de diferencia, mapa latente y controles de reentrenamiento.</div>\n    <figure><img src=\"assets/tab_autoencoder.jpg\" />", 1)
    html = html.replace("<figure><img src=\"assets/tab_vae.jpg\" />", "<div class=\"caption-check\">Qué mirar: la misma gramática del AE más el panel de generación desde el prior.</div>\n    <figure><img src=\"assets/tab_vae.jpg\" />", 1)
    html = html.replace("<figure><img src=\"assets/tab_gan.jpg\" />", "<div class=\"caption-check\">Qué mirar: muestras desde ruido, curva G/D y controles de truncation/interpolación.</div>\n    <figure><img src=\"assets/tab_gan.jpg\" />", 1)
    html = html.replace("<figure><img src=\"assets/tab_diffusion.jpg\" />", "<div class=\"caption-check\">Qué mirar: forward de ruido, reverse de denoising y coste de pasos de muestreo.</div>\n    <figure><img src=\"assets/tab_diffusion.jpg\" />", 1)

    write(DOCS / "report_v2.html", html)


def build_working_docs() -> None:
    today = date.today().isoformat()
    audit = f"""
# Auditoría pedagógica del manual v2

Fecha: {today}

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
"""

    plan = """
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
- Buscar restos de `TODO`, `version-codex`, `imagen_obsoleta` y referencias internas antiguas.
"""

    tracking = f"""
# Seguimiento manual v2

Estado inicial: en curso.

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

- {today}: leídas instrucciones del proyecto, protocolo de generación de libros, PDF/HTML actual y estructura del código.
- {today}: planificada v2 como capa pedagógica sobre `report.html`, con nuevos bloques de lectura, visuales SVG y anexos formales.
- {today}: generado `report_v2.html` desde `docs/Back/build_manual_v2.py`.

## Assets creados

- Visuales nuevos embebidos como SVG/HTML en `docs/report_v2.html`.
- QA visual en `docs/Back/qa_visual/v2/` tras generar el PDF.

## Decisiones de publicación

- No se actualiza `README.md` ni se mueve la versión anterior hasta revisión humana.
- No se hace commit porque el usuario pidió crear la v2, no publicar/cerrar una fase de producto.
"""

    write(BACK / "auditoria_manual_v2.md", audit)
    write(BACK / "plan_visual_manual_v2.md", plan)
    write(BACK / "seguimiento_manual_v2.md", tracking)


if __name__ == "__main__":
    build_report()
    build_working_docs()
    print("manual v2 generado")
