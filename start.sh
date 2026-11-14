#!/bin/bash
cd ~/felly/generate-t2i-model || exit 1
source .venv/bin/activate && source env.sh
F=$HOME/felly
if [ -f uvicorn.pid ] && kill -0 $(cat uvicorn.pid) 2>/dev/null; then
  echo "Server masih jalan. Jalankan ./stop.sh dulu."; exit 1
fi
setsid bash -c 'echo $$ > ollama.pid; exec env HOME='$F'/tools/ollama-home '$F'/tools/ollama/bin/ollama serve > ollama.log 2>&1' < /dev/null &
sleep 3
setsid bash -c 'echo $$ > uvicorn.pid; exec env DATA_DIR=$PWD/data_studi uvicorn app.main:app --host 127.0.0.1 --port 8010 > uvicorn.log 2>&1' < /dev/null &
echo "Menunggu warmup model..."
sleep 2
until grep -q "Uvicorn running" uvicorn.log 2>/dev/null; do
  kill -0 $(cat uvicorn.pid) 2>/dev/null || { echo "uvicorn gagal:"; tail -20 uvicorn.log; exit 1; }
  sleep 2
done
grep "\[warmup\]" uvicorn.log
setsid bash -c 'echo $$ > tunnel.pid; exec env HOME='$F'/tools/cloudflared-home '$F'/tools/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8010 > tunnel.log 2>&1' < /dev/null &
for i in $(seq 1 15); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log 2>/dev/null | head -1)
  [ -n "$URL" ] && { echo "Tautan publik: $URL"; exit 0; }
  sleep 2
done
echo "Tunnel belum memberi tautan, cek: tail tunnel.log"
