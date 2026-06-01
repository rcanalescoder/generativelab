#!/usr/bin/env bash
#
# arrancar.sh — Arranca (o REINICIA) el Generative Models Visual Lab:
#   • backend  → FastAPI + PyTorch (MPS) en  http://localhost:8000
#   • frontend → Vite (React)        en  http://localhost:5173
#
# Idempotente: si ya hay algo escuchando en esos puertos, lo PARA antes de
# volver a arrancar. Así, ejecutarlo siempre te deja una instancia limpia.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACK_PORT=8000
FRONT_PORT=5173
LOG_DIR="$ROOT/.logs"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------- parar puerto
stop_port() {
  local port="$1" name="$2" pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "⏹  Parando ${name} (escuchaba en :${port} — PIDs: ${pids})…"
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

echo "── Generative Models Visual Lab ──"

# 1) Parar lo que hubiera arrancado
stop_port "$BACK_PORT" "backend"
stop_port "$FRONT_PORT" "frontend"
pkill -f "uvicorn app.main:app" 2>/dev/null || true   # por si quedó algún resto

# 2) Comprobaciones mínimas
if [ ! -x "$ROOT/backend/.venv/bin/uvicorn" ]; then
  echo "✋ Falta el entorno del backend. Créalo una vez con:"
  echo "     cd \"$ROOT/backend\" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "✋ Faltan dependencias del frontend. Instálalas una vez con:"
  echo "     cd \"$ROOT/frontend\" && npm install"
  exit 1
fi

# 3) Arrancar backend (MPS con fallback a CPU — nunca CUDA)
echo "▶  Arrancando backend  → http://localhost:${BACK_PORT}"
(
  cd "$ROOT/backend"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  nohup .venv/bin/uvicorn app.main:app --port "$BACK_PORT" > "$LOG_DIR/backend.log" 2>&1 &
  echo $! > "$LOG_DIR/backend.pid"
)

# 4) Arrancar frontend (Vite)
echo "▶  Arrancando frontend → http://localhost:${FRONT_PORT}"
(
  cd "$ROOT/frontend"
  nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$LOG_DIR/frontend.pid"
)

# 5) Esperar a que respondan
printf "⏳ Esperando al backend"
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:${BACK_PORT}/api/health" >/dev/null 2>&1; then break; fi
  printf "."; sleep 0.5
done
printf "\n"

if curl -sf "http://localhost:${BACK_PORT}/api/health" >/dev/null 2>&1; then
  dev="$(curl -sf "http://localhost:${BACK_PORT}/api/health" | sed -E 's/.*"device":"([^"]+)".*/\1/')"
  echo "✅ Backend OK (device: ${dev})"
else
  echo "⚠️  El backend aún no responde. Mira el log: $LOG_DIR/backend.log"
fi

printf "⏳ Esperando al frontend"
for _ in $(seq 1 40); do
  if curl -sf "http://localhost:${FRONT_PORT}" >/dev/null 2>&1; then break; fi
  printf "."; sleep 0.5
done
printf "\n"
if curl -sf "http://localhost:${FRONT_PORT}" >/dev/null 2>&1; then
  echo "✅ Frontend OK"
else
  echo "⚠️  El frontend aún no responde (Vite tarda un poco la 1ª vez). Log: $LOG_DIR/frontend.log"
fi

cat <<EOF

  Abre:   http://localhost:${FRONT_PORT}
  Logs:   tail -f "$LOG_DIR/backend.log"
          tail -f "$LOG_DIR/frontend.log"
  Parar:  lsof -ti tcp:${BACK_PORT} | xargs kill ; lsof -ti tcp:${FRONT_PORT} | xargs kill
          (o vuelve a ejecutar ./arrancar.sh para reiniciar)
EOF
