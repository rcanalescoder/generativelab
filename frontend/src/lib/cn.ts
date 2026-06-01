/** Une clases condicionalmente, descartando valores falsy. Sin dependencias. */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}
