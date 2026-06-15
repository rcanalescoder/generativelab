#!/usr/bin/env bash
#
# parar.sh — Para adecuadamente el backend y frontend del Generative Models Visual Lab.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_PORT=8000
FRONT_PORT=5173
LOG_DIR="$ROOT/.logs"

echo "── Deteniendo Generative Models Visual Lab ──"

# Función para detener un proceso por PID
stop_pid() {
  local pid_file="$1" name="$2"
  if [ -f "$pid_file" ]; then
    local pid
    pid=$(cat "$pid_file")
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "⏹  Parando ${name} (PID: ${pid})…"
      kill "$pid" 2>/dev/null || true
      sleep 1
      if ps -p "$pid" > /dev/null 2>&1; then
        echo "⚠️  Forzando parada de ${name} (PID: ${pid})…"
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pid_file"
  fi
}

# Función para detener por puerto (por si acaso o para procesos huérfanos)
stop_port() {
  local port="$1" name="$2" pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "⏹  Parando procesos huérfanos de ${name} en puerto :${port} (PIDs: ${pids})…"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

# 1) Parar por PID si están los archivos
stop_pid "$LOG_DIR/backend.pid" "backend"
stop_pid "$LOG_DIR/frontend.pid" "frontend"

# 2) Doble comprobación/seguridad por puertos
stop_port "$BACK_PORT" "backend"
stop_port "$FRONT_PORT" "frontend"

# 3) Limpieza adicional por nombre de proceso
pkill -f "uvicorn app.main:app" 2>/dev/null || true

echo "✅ Servicios detenidos correctamente."
