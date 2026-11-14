#!/bin/bash
cd ~/felly/generate-t2i-model || exit 1
for p in tunnel uvicorn ollama; do
  if [ -f $p.pid ]; then kill $(cat $p.pid) 2>/dev/null && echo "$p dimatikan"; rm -f $p.pid; fi
done
