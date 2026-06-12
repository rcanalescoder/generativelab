"""Construye docs/report_v3.html a partir de docs/report_v2.html.

La v3 añade la capa «ciclo de calidad»: el capítulo que cuenta, con cifras y antes/después,
qué cambios de algoritmos y de parámetros llevaron el VAE de KID 608→121, el GAN de 200→37,5
y Diffusion de 175→19,4. Fuente de los datos: «Lista de Mejoras.md» y experiments/ledger.jsonl.

Uso:  python3 docs/Back/build_manual_v3.py   (luego, PDF con Chrome headless)
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

EXTRA_CSS = r"""

  /* ---------- v3: ciclo de calidad ---------- */
  .ba-pair{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin:12px 0; break-inside:avoid; page-break-inside:avoid; }
  .ba-pair figure{ margin:0; }
  .ba-tag{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:8pt; font-weight:800;
    letter-spacing:.5px; text-transform:uppercase; margin-bottom:6px; }
  .ba-tag.antes{ background:#fdeef6; color:#a41d54; border:1px solid #f6cfe3; }
  .ba-tag.despues{ background:#e9fbf4; color:#0a6b4e; border:1px solid #c7eede; }
  .delta-strip{ display:grid; grid-template-columns:repeat(3,1fr); gap:9px; margin:12px 0; }
  .delta-strip .d{ border:1px solid var(--line); border-radius:12px; background:#fff; padding:10px 12px; text-align:center; }
  .delta-strip .d .k{ font-size:8pt; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); font-weight:800; }
  .delta-strip .d .v{ font-family:"SFMono-Regular",Menlo,monospace; font-size:12.5pt; font-weight:800; margin-top:3px; }
  .delta-strip .d .v .from{ color:#a41d54; }
  .delta-strip .d .v .to{ color:#0a6b4e; }
  .delta-strip .d p{ font-size:8.4pt; color:var(--slate); margin:4px 0 0; }
  .tipo{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:7.6pt; font-weight:800;
    letter-spacing:.4px; text-transform:uppercase; white-space:nowrap; }
  .tipo.alg{ background:#f4effe; color:#5b21b6; border:1px solid #e3d5fb; }
  .tipo.par{ background:#eff4fe; color:#1d4ed8; border:1px solid #d6e2fb; }
  .tipo.infra{ background:#e7f7fb; color:#0e7490; border:1px solid #c8ecf4; }
  .metric-guide, .metric-guide .m, .concept-flow, .chapter-frame, .delta-strip,
  .two-col, .selfcheck, .reader-note{ break-inside:avoid; page-break-inside:avoid; }
  @media print{ .ba-pair{ grid-template-columns:1fr 1fr; } }
"""

# --------------------------------------------------------------------------------------
# Cápsulas «versión 3» al final de los capítulos VAE / GAN / Diffusion
# --------------------------------------------------------------------------------------
CAPSULE_VAE = r"""
    <div class="reader-note" style="--accent:#7c3aed">
      <div class="eyebrow">Actualización · versión 3</div>
      <div class="big">Este capítulo te enseñó un VAE que generaba manchas marrones. Ya no.</div>
      <p>El ciclo de calidad de la v3 encontró la causa exacta —un desequilibrio entre el error de
      reconstrucción y la KL dentro de la pérdida— y la arregló con tres cambios pequeños. Mismo
      modelo, mismos datos: <b>KID 608 → 121</b> y la diversidad pasó de 0,15 (todas iguales) a ≈1
      (tan variadas como el dataset). La historia completa, con el antes/después y la tabla de
      cambios, está en el capítulo <b>«Versión 3 · La historia»</b>.</p>
    </div>
"""

CAPSULE_GAN = r"""
    <div class="reader-note" style="--accent:#db2777">
      <div class="eyebrow">Actualización · versión 3</div>
      <div class="big">La GAN «tosca» de este capítulo ahora dibuja las caras más nítidas del laboratorio clásico.</div>
      <p>En la v3 no cambió el juego adversarial: cambiaron cuatro detalles de entrenamiento
      (promediado EMA de los pesos, etiquetas suavizadas, augmentación DiffAugment y su presupuesto
      real de datos y epochs). Resultado medido: <b>KID 199,5 → 37,5</b>. Y de regalo, una lección de
      método: los trucos de estabilidad <i>empeoran</i> los entrenamientos cortos antes de ganar en
      los largos. Detalles en <b>«Versión 3 · La historia»</b>.</p>
    </div>
"""

CAPSULE_DIFF = r"""
    <div class="reader-note" style="--accent:#0891b2">
      <div class="eyebrow">Actualización · versión 3</div>
      <div class="big">A difusión solo le faltaba una cosa: tiempo de entrenamiento de verdad.</div>
      <p>La receta ya era correcta (EMA, schedule cosine, muestreo DDIM). La v3 le dio el dataset
      completo, un arranque suave del optimizador y —la pieza práctica clave— entrenamiento
      <b>reanudable</b> para poder acumular sesiones largas sin miedo a perderlas. Resultado:
      <b>KID 175 → 19,4</b>, el mejor del laboratorio. Detalles en <b>«Versión 3 · La historia»</b>.</p>
    </div>
"""

# --------------------------------------------------------------------------------------
# Capítulo grande «Versión 3 · La historia» (5 secciones)
# --------------------------------------------------------------------------------------
V3_PART = r"""

<!-- ============ V3 · LA HISTORIA ============ -->
<section class="section">
  <div class="band b-ink"><div class="n">VERSIÓN 3 · LA HISTORIA</div><h2>El ciclo de calidad: de manchas a caras</h2>
    <div class="tag">Qué cambiamos en cada modelo, cómo lo medimos y la prueba de que funcionó</div></div>
  <div class="pad" style="--accent:#2563eb;--accent-soft:#eff4fe">

    <div class="chapter-frame">
      <div class="question">
        <div class="label">La pregunta de esta parte</div>
        <div class="q">Si el laboratorio estaba «bien hecho»… ¿por qué las caras generadas eran tan malas?</div>
        <p>La versión 2 enseñaba los mecanismos correctamente, pero las muestras decepcionaban:
        el VAE pintaba 64 manchas marrones idénticas, la GAN era tosca y difusión se quedaba blanda.
        La v3 es la respuesta: un <b>ciclo de experimentos</b> que diagnostica, cambia una cosa cada
        vez y solo se queda con lo que mejora una métrica objetiva.</p>
      </div>
      <div class="answer">
        <div class="mini"><b>Diagnóstico 1 · VAE:</b> un <b>bug de escala</b> en la pérdida colapsaba el generador (no era falta de capacidad).</div>
        <div class="mini"><b>Diagnóstico 2 · GAN:</b> receta de 2016 sin los <b>estabilizadores</b> modernos, y entrenada con un tercio de los datos.</div>
        <div class="mini"><b>Diagnóstico 3 · Diffusion:</b> código correcto, <b>presupuesto de entrenamiento</b> corto.</div>
        <div class="mini"><b>Diagnóstico 0 (el importante):</b> el laboratorio <b>no medía la calidad de generación</b> — solo la de reconstrucción. Sin métrica, toda mejora es opinión.</div>
      </div>
    </div>

    <h3 class="sub">El ciclo, en cuatro pasos</h3>
    <p class="lead">Es el mismo bucle de realimentación que prometía la introducción, ahora con todas las piezas: cada hipótesis se entrena acotada, se evalúa con el mismo protocolo y se decide con el marcador delante.</p>
    <div class="concept-flow">
      <div class="step"><div class="top">1 · hipótesis</div><div class="mid">Proponer</div><div class="bot">un cambio concreto de algoritmo o parámetro</div></div>
      <div class="arr">→</div>
      <div class="step"><div class="top">2 · run acotado</div><div class="mid">Entrenar</div><div class="bot">screen corto o presupuesto completo, semilla fija</div></div>
      <div class="arr">→</div>
      <div class="step"><div class="top">3 · protocolo fijo</div><div class="mid">Evaluar</div><div class="bot">KID + diversidad + rejilla 8×8 de semilla fija</div></div>
      <div class="arr">→</div>
      <div class="step"><div class="top">4 · marcador</div><div class="mid">Decidir</div><div class="bot">se queda solo lo que mejora; todo va al ledger</div></div>
    </div>

    <h3 class="sub">Las reglas del juego: medir antes de opinar</h3>
    <p>La pieza nueva de la v3 es la métrica. La reconstrucción (PSNR/SSIM) no sirve para juzgar
    la <i>generación</i>: el VAE colapsado reconstruía «aceptablemente» mientras generaba manchas.
    Por eso todos los experimentos de esta parte se juzgan así:</p>
    <div class="metric-guide">
      <div class="m"><span class="k">KID ×1000</span><p><b>La nota principal.</b> Compara la distribución de caras generadas con la de caras reales usando los «ojos» de una red ya entrenada (InceptionV3). <b>Menor = mejor</b>; 0 sería indistinguible del dataset.</p></div>
      <div class="m"><span class="k">Diversidad</span><p><b>El detector de colapso.</b> Distancia media entre las propias muestras, relativa a la del dataset real: ≈1 significa «tan variadas como las reales»; ≈0, «todas iguales».</p></div>
      <div class="m"><span class="k">Rejilla 8×8</span><p><b>El control visual.</b> 64 muestras con la misma semilla en todos los runs de un modelo: lo que cambia entre dos rejillas es el modelo, no el azar.</p></div>
    </div>
    <div class="note">Protocolo fijo en todos los experimentos: <b>2.048 caras generadas frente a 2.048 reales</b> de un conjunto apartado (semilla 1234), KID por subconjuntos con estimador insesgado. En difusión, 512 muestras (generar es caro) — por eso difusión solo se compara contra difusión. Cada run queda registrado en <span class="mono">experiments/ledger.jsonl</span>; en total, <b>17 experimentos</b> en una tarde.</div>

    <div class="selfcheck">
      <div class="h">Para situarte antes de seguir</div>
      <ol>
        <li>¿Por qué un VAE con buen PSNR de reconstrucción puede estar generando basura? (Pista: reconstruir parte de una foto; generar parte de ruido.)</li>
        <li>¿Qué detecta la diversidad que el KID solo no deja claro?</li>
      </ol>
    </div>
  </div>
</section>

<!-- ============ V3 · VAE ============ -->
<section class="section">
  <div class="band b-violet"><div class="n">VERSIÓN 3 · MEJORA 01</div><h2>VAE: el bug que pintaba manchas</h2>
    <div class="tag">recon + β·KL — pero un término iba «por píxel» y el otro «por dimensión»</div></div>
  <div class="pad" style="--accent:#7c3aed;--accent-soft:#f4effe">

    <p>La pérdida del VAE suma dos fuerzas: <b>reconstruir bien</b> (error entre la cara y su copia)
    y <b>mantener ordenada la memoria</b> (la KL, que empuja el espacio latente hacia una campana
    de Gauss). El capítulo del VAE explicó ese equilibrio. El bug estaba en cómo se sumaban:
    el error de reconstrucción se <b>promediaba</b> sobre los 12.288 valores de la imagen, pero la
    KL se <b>sumaba</b> sobre las 128 dimensiones del latente. Resultado: con β=1, la regularización
    pesaba <b>miles de veces más</b> de lo que la teoría manda.</p>
    <p>El optimizador hizo exactamente lo que se le pedía: aplastó la KL hasta cero. Eso significa
    que el latente <span class="mono">z</span> dejó de transportar información — y un decoder al que
    <span class="mono">z</span> no le dice nada solo puede hacer una cosa: pintar siempre la
    <b>«cara promedio»</b> del dataset. Por eso las 64 muestras eran la misma mancha marrón.</p>

    <div class="ba-pair">
      <figure><span class="ba-tag antes">Antes · v2</span><img src="assets/v3_vae_antes.jpg" /><figcaption><b>El colapso, a la vista:</b> 64 muestras del prior con la config demo v2 (12 epochs). Todas son prácticamente la misma cara. KID 608 · diversidad 0,145.</figcaption></figure>
      <figure><span class="ba-tag despues">Después · v3</span><img src="assets/v3_vae_despues.jpg" /><figcaption><b>El mismo tipo de modelo, arreglado:</b> variante grande, 40 epochs, β=0,5. Caras suaves —es lo propio de un VAE— pero distintas y reconocibles. KID 121 · diversidad 0,976.</figcaption></figure>
    </div>

    <h3 class="sub">Qué se cambió exactamente</h3>
    <table class="params">
      <thead><tr><th>Cambio</th><th>En la v2</th><th>En la v3</th><th>Tipo</th></tr></thead>
      <tbody>
        <tr><td><b>Escala de la KL en la pérdida</b> <span class="mono">(el bug)</span></td><td>KL sumada sobre 128 dims frente a un error promediado por píxel → β=1 equivalía a β≈12.000</td><td>KL dividida entre 12.288 (los valores de una imagen): misma escala «por píxel» que el error; β=1 vuelve a significar ELBO equilibrado</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Calentamiento de β</b> (KL annealing)</td><td>No existía: la regularización entraba a tope desde el primer paso</td><td>β sube linealmente de 0 a su valor durante el primer 30 % de las epochs: primero aprender a reconstruir, después ordenar</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Guarda numérica de logσ²</b></td><td>Sin límite — la variante grande explotaba al arrancar (KL ≈ 4·10¹⁷)</td><td><span class="mono">clamp(−8, 8)</span> en el encoder: inocuo en el rango normal, elimina la explosión de raíz</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>β del demo</b></td><td>1,0 (que en realidad valía ≈12.000)</td><td>0,5 — elegido con un barrido medido {0,5 · 1 · 2}: 241 / 255 / 292 en KID</td><td><span class="tipo par">parámetro</span></td></tr>
        <tr><td><b>Epochs del demo</b></td><td>12</td><td>40 (dataset completo)</td><td><span class="tipo par">parámetro</span></td></tr>
        <tr><td><b>Arquitectura del demo</b></td><td>basico (base 32)</td><td>grande (base 64 + bloques residuales)</td><td><span class="tipo par">parámetro</span></td></tr>
      </tbody>
    </table>
    <div class="caption-check">Sin cambios: latent_dim 128, lr 10⁻³, batch 256, pérdida MSE. El slider de β de la app conserva su rango — ahora con su significado de libro.</div>

    <h3 class="sub">La evidencia, paso a paso</h3>
    <table class="compare">
      <thead><tr><th>Run (mismos datos, misma semilla)</th><th>KID ×1000 ↓</th><th>Diversidad</th><th>PSNR recon (dB)</th></tr></thead>
      <tbody>
        <tr><td>v2 demo · 12 epochs · β=1 sin reescalar</td><td><b>608,5</b></td><td>0,145 (colapso)</td><td>12,1</td></tr>
        <tr><td>v3 · <b>mismo presupuesto</b> (12 epochs) · KL/píxel · β=0,5</td><td><b>241,5</b></td><td>1,097</td><td>17,9</td></tr>
        <tr><td>v3 final · 40 epochs · variante grande · β=0,5</td><td><b>121,3</b></td><td>0,976</td><td>19,1</td></tr>
      </tbody>
    </table>
    <div class="delta-strip">
      <div class="d"><div class="k">KID ×1000</div><div class="v"><span class="from">608</span> → <span class="to">121</span></div><p>−80 % con tres cambios de pérdida y más epochs</p></div>
      <div class="d"><div class="k">Diversidad</div><div class="v"><span class="from">0,15</span> → <span class="to">0,98</span></div><p>de «todas iguales» a «como el dataset»</p></div>
      <div class="d"><div class="k">PSNR recon</div><div class="v"><span class="from">12,1</span> → <span class="to">19,1</span> dB</div><p>arreglar la generación también mejoró la copia</p></div>
    </div>

    <div class="two-col">
      <div class="trap-card"><div class="h">La anécdota que enseña</div>Al lanzar el entrenamiento largo de la variante grande, la KL <b>explotó</b> en la primera epoch (≈4·10¹⁷). Con el calentamiento, β empieza casi en 0… y entonces <i>nada</i> ancla la varianza del encoder: <span class="mono">exp(logσ²)</span> se desborda. La guarda <span class="mono">clamp(−8,8)</span> lo corta de raíz. Lección: <b>el warmup de KL necesita una guarda numérica</b>.</div>
      <div class="limit-card"><div class="h">El límite honesto</div>Un VAE «vainilla» a 64×64 produce caras <b>suaves</b>, no nítidas: promedia donde hay incertidumbre. Eso no es un fallo de esta implementación — es la firma de la familia, y la razón de que existan GAN y difusión. El comparador del laboratorio lo muestra tal cual.</div>
    </div>
  </div>
</section>

<!-- ============ V3 · GAN ============ -->
<section class="section">
  <div class="band b-pink"><div class="n">VERSIÓN 3 · MEJORA 02</div><h2>GAN: cuatro estabilizadores y su presupuesto real</h2>
    <div class="tag">El juego adversarial no cambió — cambió cómo lo entrenamos</div></div>
  <div class="pad" style="--accent:#db2777;--accent-soft:#fdeef6">

    <p>La GAN de la v2 era el DCGAN «de libro» (2016): correcto, pero sin los trucos que una década
    de práctica convirtió en estándar. Y además entrenaba con 15.000 de las 43.102 caras. La v3
    añadió cuatro estabilizadores —cada uno ataca un modo de fallo concreto del duelo— y le dio
    su presupuesto real.</p>

    <div class="ba-pair">
      <figure><span class="ba-tag antes">Antes · v2</span><img src="assets/v3_gan_antes.jpg" /><figcaption><b>Demo v2</b> (25 epochs, 15k imágenes): hay caras, pero con manchas, deformaciones y zonas lavadas. KID 199,5.</figcaption></figure>
      <figure><span class="ba-tag despues">Después · v3</span><img src="assets/v3_gan_despues.jpg" /><figcaption><b>Demo v3</b> (60 epochs, dataset completo, EMA+smoothing+DiffAugment): caras nítidas, ojos detallados, estilos variados. KID 37,5.</figcaption></figure>
    </div>

    <h3 class="sub">Qué se cambió exactamente</h3>
    <table class="params">
      <thead><tr><th>Cambio</th><th>En la v2</th><th>En la v3</th><th>Tipo</th></tr></thead>
      <tbody>
        <tr><td><b>Pesos con los que se generan las caras</b></td><td>Los del último paso de entrenamiento (oscilan con cada lote)</td><td>Media móvil <b>EMA</b> de los pesos del generador (decay 0,999): se muestrea con la versión «promediada y serena» de G</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Etiqueta de «real» para el juez</b></td><td>1,0</td><td>0,9 (<b>label smoothing</b>): impide que D se vuelva sobreconfiado y deje a G sin gradiente útil</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Lo que ve el juez</b></td><td>Las imágenes tal cual (D puede memorizar el dataset)</td><td><b>DiffAugment</b>: color + traslación + recorte aleatorios, idénticos para reales y falsas, diferenciables — D ya no puede memorizar y el juego no se desequilibra</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Learning rates</b></td><td>Una sola lr (2·10⁻⁴) para G y D</td><td><span class="mono">lr_G</span> y <span class="mono">lr_D</span> separadas (sliders nuevos en la app). El A/B medido mantuvo 2·10⁻⁴/2·10⁻⁴ y <b>descartó TTUR</b> (1·10⁻⁴/2·10⁻⁴ quedó peor: 56,5 vs 52,2)</td><td><span class="tipo par">parámetro</span></td></tr>
        <tr><td><b>Datos y duración del demo</b></td><td>15.000 imágenes × 25 epochs</td><td>43.102 imágenes × 60 epochs</td><td><span class="tipo par">parámetro</span></td></tr>
      </tbody>
    </table>

    <h3 class="sub">La evidencia — y una lección de método</h3>
    <table class="compare">
      <thead><tr><th>Run</th><th>Presupuesto</th><th>KID ×1000 ↓</th></tr></thead>
      <tbody>
        <tr><td>v2 demo</td><td>25 epochs · 15k imgs</td><td><b>199,5</b></td></tr>
        <tr><td>receta v2, presupuesto real (control)</td><td>40 epochs · 43k imgs</td><td><b>70,9</b></td></tr>
        <tr><td>estabilizadores v3</td><td>40 epochs · 43k imgs</td><td><b>52,2</b></td></tr>
        <tr><td>estabilizadores v3 (demo final)</td><td>60 epochs · 43k imgs</td><td><b>37,5</b></td></tr>
      </tbody>
    </table>
    <p class="lead">Lectura: solo darle a la GAN sus datos y epochs ya baja 199→71; los estabilizadores añaden otro −26 %; y estirar a 60 epochs (posible <i>porque</i> ahora es estable) deja el 37,5.</p>

    <div class="two-col">
      <div class="trap-card"><div class="h">Los screens cortos mintieron</div>Probamos primero cada estabilizador con runs cortos (2.950 pasos), y <b>todos perdieron</b> contra la receta v2 (224–414 vs 200 de KID). ¿Por qué? La EMA promedia ~1.000 pasos —un tercio del run— y llega «con retraso»; el smoothing y la augmentación frenan la convergencia temprana. Son <b>apuestas a largo plazo</b>: a 13.500 pasos la foto se invierte por completo.</div>
      <div class="check-card"><div class="h">La regla que queda</div>Las mejoras de <b>estabilidad</b> no se evalúan con entrenamientos cortos: hay que medirlas al <b>presupuesto real</b>. (Y su simétrica: un truco que gana en el screen puede no aportar nada al final — TTUR pasó exactamente eso.)</div>
    </div>
  </div>
</section>

<!-- ============ V3 · DIFFUSION ============ -->
<section class="section">
  <div class="band b-cyan"><div class="n">VERSIÓN 3 · MEJORA 03</div><h2>Diffusion: el mismo código, el presupuesto que merecía</h2>
    <div class="tag">Cuando la receta ya es buena, la palanca es el cómputo — y poder reanudarlo</div></div>
  <div class="pad" style="--accent:#0891b2;--accent-soft:#e7f7fb">

    <p>Difusión era el caso opuesto al VAE: aquí <b>no había ningún bug</b>. La implementación ya
    traía las buenas prácticas (promediado EMA, schedule cosine, muestreo DDIM, atención en la
    variante nítida). Lo que faltaba era prosaico: la variante de calidad apenas se había entrenado
    (60 epochs sobre 18.000 imágenes en su mejor tirada) y cualquier corte obligaba a empezar de cero.</p>

    <div class="ba-pair">
      <figure><span class="ba-tag antes">Antes · v2</span><img src="assets/v3_diff_antes.jpg" /><figcaption><b>Demo v2</b> (variante ágil: trabaja a 32×32 y se reescala): caras correctas pero blandas, sin detalle fino. KID 175.</figcaption></figure>
      <figure><span class="ba-tag despues">Después · v3</span><img src="assets/v3_diff_despues.jpg" /><figcaption><b>Demo v3</b> (variante nítida a 64×64 con atención, 28 epochs con el dataset completo): detalle fino, alto contraste y la mayor variedad del laboratorio. KID 19,4.</figcaption></figure>
    </div>

    <h3 class="sub">Qué se cambió exactamente</h3>
    <table class="params">
      <thead><tr><th>Cambio</th><th>En la v2</th><th>En la v3</th><th>Tipo</th></tr></thead>
      <tbody>
        <tr><td><b>Variante del demo</b></td><td>ágil: UNet a 32×32, reescalada a 64 para mostrar (emborrona)</td><td>nítida: 64×64 nativo, base 96, auto-atención en las resoluciones 16 y 8</td><td><span class="tipo par">parámetro</span></td></tr>
        <tr><td><b>Datos de entrenamiento</b></td><td>18.000–20.000 imágenes</td><td>43.102 (dataset completo)</td><td><span class="tipo par">parámetro</span></td></tr>
        <tr><td><b>Arranque del optimizador</b></td><td>lr fija desde el primer paso</td><td><b>warmup</b> lineal de lr durante 500 pasos (estabiliza el inicio de la UNet)</td><td><span class="tipo alg">algoritmo</span></td></tr>
        <tr><td><b>Sesiones de entrenamiento</b></td><td>Una tirada: si se corta, se pierde</td><td><b>Reanudable</b>: cada 4 epochs guarda modelo + optimizador + EMA; <span class="mono">--resume</span> continúa donde se quedó</td><td><span class="tipo infra">infraestructura</span></td></tr>
        <tr><td><b>Lo que no se tocó</b></td><td colspan="2">EMA (decay 0,999), schedule cosine con T=200 y muestreo DDIM ya estaban bien en la v2 — se conservan tal cual</td><td><span class="tipo alg">sin cambios</span></td></tr>
      </tbody>
    </table>

    <h3 class="sub">La evidencia</h3>
    <table class="compare">
      <thead><tr><th>Run (eval: 512 muestras, DDIM 80 pasos)</th><th>KID ×1000 ↓</th><th>Diversidad</th></tr></thead>
      <tbody>
        <tr><td>v2 demo · ágil 32×32 · 30 epochs · 20k imgs</td><td><b>175,1</b></td><td>1,130</td></tr>
        <tr><td>v3 demo · nítida 64×64 · 28 epochs · 43k imgs · warmup</td><td><b>19,4</b></td><td>0,984</td></tr>
      </tbody>
    </table>
    <div class="note"><b>El mejor KID del laboratorio</b> — coherente con lo que el capítulo de difusión anticipaba: muchos pasos pequeños y un objetivo estable escalan mejor que un duelo. Y aún hay margen: al cortar, la pérdida seguía bajando (0,0616 ↘). Cada epoch extra cuesta ~2 min en un Mac M-series; <span class="mono">--resume</span> permite acumularlas por noches.</div>
  </div>
</section>

<!-- ============ V3 · VEREDICTO ============ -->
<section class="section">
  <div class="band b-green"><div class="n">VERSIÓN 3 · VEREDICTO</div><h2>El marcador final, y cómo repetirlo en casa</h2>
    <div class="tag">Tres modelos, una tarde de experimentos, todo medido y registrado</div></div>
  <div class="pad" style="--accent:#0c9f6e;--accent-soft:#e9fbf4">

    <table class="compare">
      <thead><tr><th>Modelo</th><th>KID ×1000 · v2</th><th>KID ×1000 · v3</th><th>Mejora</th><th>Diversidad v3</th></tr></thead>
      <tbody>
        <tr><td>VAE</td><td>608,5</td><td><b>121,3</b></td><td>−80 %</td><td>0,98</td></tr>
        <tr><td>GAN</td><td>199,5</td><td><b>37,5</b></td><td>−81 %</td><td>0,91</td></tr>
        <tr><td>Diffusion</td><td>175,1</td><td><b>19,4</b></td><td>−89 %</td><td>0,98</td></tr>
      </tbody>
    </table>
    <div class="caption-check">Misma vara de medir en cada pareja v2/v3 (mismas semillas y mismo protocolo). El ranking final —difusión > GAN > VAE— es el esperado por la teoría; ahora además está medido.</div>

    <h3 class="sub">El ciclo, fase a fase</h3>
    <div class="timeline-v2">
      <div class="item"><div class="tag2">V3.0</div><div class="name">Arnés</div><p>KID + diversidad + rejillas de semilla fija; runner que entrena, evalúa y registra; baselines v2 medidos.</p></div>
      <div class="item"><div class="tag2">V3.1</div><div class="name">VAE</div><p>El bug de escala: KL por píxel, warmup de β, guarda de logσ². 608 → 121.</p></div>
      <div class="item"><div class="tag2">V3.2</div><div class="name">GAN</div><p>EMA + smoothing + DiffAugment + presupuesto real. Ablación honesta. 199 → 37,5.</p></div>
      <div class="item"><div class="tag2">V3.3</div><div class="name">Diffusion</div><p>Nítida a 64×64 con dataset completo, warmup y reanudación. 175 → 19,4.</p></div>
      <div class="item"><div class="tag2">V3.4</div><div class="name">Cierre</div><p>Demos regenerados, marcador final, este capítulo y la «Lista de Mejoras» como memoria del ciclo.</p></div>
    </div>

    <div class="poster">
      <h3>Cuatro lecciones que viajan a cualquier proyecto</h3>
      <div class="poster-grid">
        <div class="p"><div class="h">1 · Mide lo que importa</div><p>Las métricas de reconstrucción no ven el colapso de generación. Hasta que existió el KID, toda mejora era una opinión.</p></div>
        <div class="p"><div class="h">2 · El presupuesto es parte del experimento</div><p>Los estabilizadores pierden en runs cortos y arrasan en largos. Evalúa cada cambio en el régimen donde va a vivir.</p></div>
        <div class="p"><div class="h">3 · Primero la pérdida, luego el cómputo</div><p>Más capacidad no arregló el VAE roto (254≈255); tras arreglar la pérdida, la misma capacidad rindió (121). El orden importa.</p></div>
        <div class="p"><div class="h">4 · La infraestructura también es ciencia</div><p>Entrenamiento reanudable, semillas fijas y un ledger convierten «me suena que mejoró» en una tabla que cualquiera puede auditar.</p></div>
      </div>
    </div>

    <h3 class="sub">La ruta reproducible</h3>
    <p class="lead">Todo el ciclo se repite con tres comandos (y queda registrado solo):</p>
    <div class="code"><div class="fn">terminal<span class="role">bash</span></div><pre><span class="cm"># desde backend/ con el venv activado — cada run escribe su KID, su rejilla y su config</span>
python -m app.experiments.runner --model vae --tag demo --mode full \
    --hp '{"epochs":40,"beta":0.5,"arch":"grande"}' --save-ckpt app/checkpoints/vae_grande_demo.pt
python -m app.experiments.runner --model gan --tag demo --mode full \
    --hp '{"epochs":60}' --save-ckpt app/checkpoints/gan_basico_demo.pt
python -m app.experiments.train_diffusion_v3 --epochs 28      <span class="cm"># reanudable con --resume</span></pre></div>
    <div class="tint"><div class="t">Dónde queda la memoria del ciclo</div>Cada experimento: <span class="mono">experiments/ledger.jsonl</span> · marcador: <span class="mono">experiments/LEADERBOARD.md</span> · rejillas: <span class="mono">experiments/grids/</span> · la historia narrada, con todas las tablas de este capítulo: <span class="mono">Lista de Mejoras.md</span> (raíz del repositorio).</div>

    <div class="selfcheck">
      <div class="h">Autocomprobación del capítulo</div>
      <ol>
        <li>¿Por qué β=1 «significaba» en la práctica β≈12.000 en el VAE de la v2? ¿Qué hizo el optimizador al respecto?</li>
        <li>La EMA promedia los pesos del generador. ¿Por qué eso <i>empeora</i> un entrenamiento de 2.950 pasos y <i>mejora</i> uno de 13.500?</li>
        <li>DiffAugment transforma también las imágenes reales. ¿Por qué eso no «hace trampa» en el duelo G/D?</li>
        <li>Difusión ganó sin tocar su algoritmo. ¿Qué dos cosas concretas explicaron su salto de 175 a 19,4?</li>
      </ol>
    </div>
  </div>
</section>
"""

# --------------------------------------------------------------------------------------
# Anexo A · paso 4 actualizado (recetas v3)
# --------------------------------------------------------------------------------------
OLD_DEMO_BLOCK = """    <div class="code"><div class="fn">terminal<span class="role">bash</span></div><pre><span class="cm"># desde backend/ con el venv activado</span>
python -m app.services.ae_service          <span class="cm"># Autoencoder</span>
python -m app.services.vae_service         <span class="cm"># VAE</span>
python -m app.services.gan_service         <span class="cm"># GAN</span>
python -m app.services.diffusion_service   <span class="cm"># Diffusion</span></pre></div>"""

NEW_DEMO_BLOCK = """    <div class="code"><div class="fn">terminal<span class="role">bash</span></div><pre><span class="cm"># desde backend/ con el venv activado — recetas v3 (las del capítulo «Versión 3 · La historia»)</span>
python -m app.experiments.runner --model vae --tag demo --mode full \\
    --hp '{"epochs":40,"beta":0.5,"arch":"grande"}' --save-ckpt app/checkpoints/vae_grande_demo.pt
python -m app.experiments.runner --model gan --tag demo --mode full \\
    --hp '{"epochs":60}' --save-ckpt app/checkpoints/gan_basico_demo.pt
python -m app.experiments.train_diffusion_v3 --epochs 28      <span class="cm"># reanudable con --resume</span>
python -m app.services.ae_service          <span class="cm"># Autoencoder (sin cambios en v3)</span></pre></div>
    <div class="note">Cada run del <i>runner</i> queda registrado en <span class="mono">experiments/ledger.jsonl</span> (KID, diversidad, config, semillas) con su rejilla en <span class="mono">experiments/grids/</span>. El marcador vivo está en <span class="mono">experiments/LEADERBOARD.md</span> y la historia completa del ciclo en <span class="mono">«Lista de Mejoras.md»</span>.</div>"""


def build() -> None:
    # la fuente v2 vive archivada en Back/ (protocolo: las versiones previas se mueven ahí)
    src = DOCS / "Back" / "report_v2.html"
    if not src.exists():
        src = DOCS / "report_v2.html"
    html = src.read_text(encoding="utf-8")

    # título, portada y pie de edición
    html = html.replace(
        "<title>Laboratorio de Modelos Generativos · Manual visual v2</title>",
        "<title>Laboratorio de Modelos Generativos · Manual visual v3</title>",
    )
    html = html.replace('<div class="edition-ribbon">Versión 2</div>', '<div class="edition-ribbon">Versión 3</div>')
    html = html.replace(
        '<div class="sub">Versión 2 ampliada: un recorrido visual, reproducible y pedagógico por Autoencoder, VAE, GAN y Diffusion. Cada capítulo explica qué problema resuelve el modelo, qué mecanismo usa, qué mirar en la interfaz, qué evidencia aporta y qué límite no conviene olvidar.</div>',
        '<div class="sub">Versión 3: el recorrido visual y reproducible por Autoencoder, VAE, GAN y Diffusion — ahora con el <b>ciclo de calidad</b> que llevó las caras generadas de «manchas» a resultados medibles: VAE 608→121, GAN 200→37,5 y Diffusion 175→19,4 en KID. Cada capítulo explica el mecanismo; el capítulo nuevo explica, cambio a cambio, cómo se mejoró.</div>',
    )
    html = html.replace(
        "<div class=\"foot\"><span>Laboratorio interactivo local · edición v2</span>",
        "<div class=\"foot\"><span>Laboratorio interactivo local · edición v3</span>",
    )
    # CSS v3
    html = html.replace("</style>", EXTRA_CSS + "\n</style>")
    # puntero en la introducción
    html = html.replace(
        "No hay nada inventado ni maquillado.</div>",
        "No hay nada inventado ni maquillado. <b>La versión 3 lleva ese bucle hasta el final:</b> el capítulo «Versión 3 · La historia» documenta el ciclo completo —diagnóstico, cambios de algoritmos y parámetros, y cifras— que llevó el VAE de KID 608 a 121, la GAN de 200 a 37,5 y difusión de 175 a 19,4.</div>",
    )
    # cápsulas al cierre de cada capítulo de modelo
    html = html.replace(
        "  </div>\n</section>\n\n<!-- ============ GAN ============ -->",
        CAPSULE_VAE + "  </div>\n</section>\n\n<!-- ============ GAN ============ -->",
        1,
    )
    html = html.replace(
        "  </div>\n</section>\n\n<!-- ============ DIFFUSION ============ -->",
        CAPSULE_GAN + "  </div>\n</section>\n\n<!-- ============ DIFFUSION ============ -->",
        1,
    )
    html = html.replace(
        "  </div>\n</section>\n\n<!-- ============ CIERRE ============ -->",
        CAPSULE_DIFF + "  </div>\n</section>\n\n<!-- ============ CIERRE ============ -->",
        1,
    )
    # capítulo grande v3 antes de los anexos
    html = html.replace("<!-- ============ INSTALACIÓN ============ -->", V3_PART + "\n<!-- ============ INSTALACIÓN ============ -->")
    # Anexo A · paso 4 con las recetas v3
    assert OLD_DEMO_BLOCK in html, "ancla del bloque de demos no encontrada"
    html = html.replace(OLD_DEMO_BLOCK, NEW_DEMO_BLOCK, 1)
    # cierre editorial
    html = html.replace(
        "Hecho para aprender, compartir y experimentar. Versión 2.",
        "Hecho para aprender, compartir y experimentar. Versión 3.",
    )

    out = DOCS / "report_v3.html"
    out.write_text(html, encoding="utf-8")
    print(f"escrito {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    build()
