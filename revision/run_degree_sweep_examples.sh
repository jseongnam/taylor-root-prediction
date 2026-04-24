#!/usr/bin/env bash
set -e

DEGREES=(5 10 15 20 25 30 35 40 45 50)

INTERVAL_SCRIPT="scripts/data/generate_interval_dataset_physchem_v4.py"
INTERVAL_CONFIG="configs/dataset_interval_physchem_v4_deg25.yaml"

ROOT_SCRIPT="scripts/data/generate_dataset_physchem_v4.py"
ROOT_CONFIG="configs/dataset_physchem_v4_deg25.yaml"

for DEG in "${DEGREES[@]}"; do
  echo "==============================================="
  echo "[INTERVAL] degree=${DEG}"
  python generate_interval_dataset_physchem_v4_cli.py \
    --script "${INTERVAL_SCRIPT}" \
    --config "${INTERVAL_CONFIG}" \
    --degree "${DEG}"
done

for DEG in "${DEGREES[@]}"; do
  echo "==============================================="
  echo "[ROOT] degree=${DEG}"
  python generate_dataset_physchem_v4_cli.py \
    --script "${ROOT_SCRIPT}" \
    --config "${ROOT_CONFIG}" \
    --degree "${DEG}"
done

echo "[DONE] all degree sweeps finished."
