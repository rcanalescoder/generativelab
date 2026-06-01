import { toPng } from 'html-to-image'

/** Exporta un nodo del DOM a PNG y lo descarga. Pensado para capturas limpias. */
export async function exportNodeToPng(node: HTMLElement, filename: string): Promise<void> {
  const dataUrl = await toPng(node, {
    pixelRatio: 2,
    cacheBust: true,
    backgroundColor: '#ffffff',
    // No intentar inlinear las @font-face de Google Fonts (cross-origin → SecurityError).
    // El lienzo se rasteriza en el mismo navegador, donde Geist ya está cargada.
    skipFonts: true,
    // Excluye el cromo (botones "i", pills…) y lo marcado como no-exportable.
    filter: (el) =>
      !(el instanceof HTMLElement && (el.dataset.noExport === 'true' || el.dataset.chrome === 'true')),
  })
  const a = document.createElement('a')
  a.href = dataUrl
  a.download = filename.endsWith('.png') ? filename : `${filename}.png`
  a.click()
}
