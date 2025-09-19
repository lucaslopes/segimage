#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/verify_graph_view.sh [INPUT_IMG]
#
# Builds four graph_view outputs with distinct parameters and verifies they differ.
# Requires uv and segimage CLI to be available in the active environment.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMG_PATH="${1:-$ROOT_DIR/data/color.png}"

out_dir="$ROOT_DIR/out"
mkdir -p "$out_dir"

uv run segimage process "$IMG_PATH" "$out_dir/grid_gray08"          -t graph_view -f png --graph-method grid       --edge-filter gray --edge-similarity 0.8 -v
uv run segimage process "$IMG_PATH" "$out_dir/prob4_s5_em60"        -t graph_view -f png --graph-method prob4      --sigma-i 5.0 --edge-min 0.6 -v
uv run segimage process "$IMG_PATH" "$out_dir/contrast4_a6_em60"    -t graph_view -f png --graph-method contrast4  --alpha 6.0 --edge-min 0.6 -v
uv run segimage process "$IMG_PATH" "$out_dir/affinity_r3_sI8_sX2_em90" -t graph_view -f png --graph-method affinity --radius 3 --sigma-i 8.0 --sigma-x 2.0 --edge-min 0.9 -v

echo "\nComparing images..."
uv run python "$ROOT_DIR/tests/verify_images_diff.py" "$out_dir/grid_gray08/color_processed.png" \
    "$out_dir/prob4_s5_em60/color_processed.png" \
    "$out_dir/contrast4_a6_em60/color_processed.png" \
    "$out_dir/affinity_r3_sI8_sX2_em90/color_processed.png" --assert-different

echo "Done. Images differ."


