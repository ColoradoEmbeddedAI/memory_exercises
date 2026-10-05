#!/usr/bin/env bash
# Export one build, compile, flash, and capture the board's UART report
# (Part 5, optional).
#
# Usage (from anywhere, ExecuTorch venv active):
#   scripts/run_board.sh 3c int8                 # UART adapter on /dev/ttyUSB0
#   scripts/run_board.sh teacher int8
#   UART=/dev/ttyUSB1 scripts/run_board.sh 3a int8
#
# The report is printed and saved to results/<model>-<precision>.txt, and
# then compared with the host's predictions by python/compare_preds.py.
# By hand, the steps are:
#   python3 python/export_mlp.py --model 3c --precision int8
#   cmake --build --preset stm32 --target flash
#   picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
set -euo pipefail

model="${1:-}"
precision="${2:-}"
case "$model/$precision" in
    teacher/fp32|teacher/int8|3[abcd]/fp32|3[abcd]/int8) ;;
    *) echo "usage: $0 {teacher|3a|3b|3c|3d} {fp32|int8}" >&2; exit 2 ;;
esac

proj="$(cd "$(dirname "$0")/.." && pwd)"
uart="${UART:-/dev/ttyUSB0}"
[ -e "$uart" ] || { echo "no UART adapter at $uart (set UART=...)" >&2; exit 1; }
build="$model-$precision"
mkdir -p "$proj/results"
out="$proj/results/$build.txt"

cd "$proj"
python3 python/export_mlp.py --model "$model" --precision "$precision" 2>&1 | grep '^\[' | grep -v 'ops:'
[ -f build/stm32/build.ninja ] || cmake --preset stm32 > /dev/null
cmake --build --preset stm32 | tail -3

# Start listening before flashing: the firmware runs once, right after
# the reset that ends programming, and prints everything in one go.
stty -F "$uart" 115200 cs8 -cstopb -parenb raw -echo
tmp="$(mktemp)"
trap 'kill "$catpid" 2>/dev/null || true; rm -f "$tmp"' EXIT
cat "$uart" > "$tmp" &
catpid=$!

openocd -f interface/stlink.cfg -f target/stm32f4x.cfg -c "transport select hla_swd" \
    -c "program build/stm32/firmware.bin 0x08000000 verify reset exit" > /dev/null 2>&1 \
    || { echo "openocd failed -- is the board connected?" >&2; exit 1; }

for _ in $(seq 60); do
    grep -q -e "--- Done ---" -e "ERROR" "$tmp" && break
    sleep 1
done
tr -d '\r' < "$tmp" | sed -n '/--- Distillation lab/,$p' > "$out"
cat "$out"
grep -q -e "--- Done ---" "$tmp" || { echo "(no '--- Done ---' within 60 s)" >&2; exit 1; }
echo
python3 python/compare_preds.py "$build"
