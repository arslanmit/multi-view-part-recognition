#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-checkpoints/run_real}"
OUT_DIR="${2:-results/run_real}"

mkdir -p "${OUT_DIR}"

cat "${SRC_DIR}"/run_real_best.ckpt.part-* > "${OUT_DIR}"/run_real_best.ckpt
cat "${SRC_DIR}"/run_real_current.ckpt.part-* > "${OUT_DIR}"/run_real_current.ckpt

if [[ -f "${SRC_DIR}/run_real_logs.log" ]]; then
  cp "${SRC_DIR}/run_real_logs.log" "${OUT_DIR}/run_real_logs.log"
fi

echo "Reconstructed checkpoints in ${OUT_DIR}"
