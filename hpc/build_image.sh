#!/bin/bash
# Run on <login-node> from a dir holding nav116.def + image_requirements.txt (hpc/ of a synced clone).
#   nohup bash build_image.sh > $PROJ/logs/build_image.log 2>&1 &     Out: $PROJ/images/nav116.{sif,sif.sha256,versions.txt}
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
P=${PROJECT_DIR:?set PROJECT_DIR}
mkdir -p "$P"/{images,logs,cache/apptainer,cache/tmp}
export APPTAINER_CACHEDIR=$P/cache/apptainer APPTAINER_TMPDIR=$P/cache/tmp
SIF=$P/images/nav116.sif
echo "build start $(date -Is)"
rm -f "$SIF.part"
apptainer build --fakeroot --force "$SIF.part" nav116.def
mv "$SIF.part" "$SIF"
sha256sum "$SIF" > "$SIF.sha256"
echo "build end $(date -Is)"
{ echo "# $(cat "$SIF.sha256")  built $(date -Is -r "$SIF")"
  apptainer exec "$SIF" python -VV
  apptainer exec "$SIF" ffmpeg -version | head -1
  echo "# pip freeze (/opt/venv)"
  apptainer exec "$SIF" /usr/local/bin/uv pip freeze --python /opt/venv/bin/python
} > "$P/images/nav116.versions.txt"
ls -lh "$SIF"; cat "$SIF.sha256"
echo BUILD_OK
