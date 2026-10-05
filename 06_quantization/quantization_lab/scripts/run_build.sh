#!/usr/bin/env bash
# Export one build, compile, flash, and capture the board's UART report.
#
# Usage (from anywhere, ExecuTorch venv active):
#   scripts/run_build.sh A int8                    # UART adapter on /dev/ttyUSB0
#   scripts/run_build.sh A int8 --order pool-relu  # extra args go to export_model.py
#   UART=/dev/ttyUSB1 scripts/run_build.sh C fp32
#
# The report is printed and saved to results/<variant>-<precision>.txt, and
# then compared with the host's predictions by python/compare_preds.py.
# This is the same as running the steps in README.md by hand:
#   python3 python/export_model.py --variant A --precision int8
#   cmake --build --preset stm32 --target flash
#   picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
set -euo pipefail

variant="${1:-}"
precision="${2:-}"
case "$variant/$precision" in
    A/fp32|A/int8|C/fp32|C/int8) ;;
    *) echo "usage: $0 {A|C} {fp32|int8} [export_model.py options]" >&2; exit 2 ;;
esac
shift 2

proj="$(cd "$(dirname "$0")/.." && pwd)"
uart="${UART:-/dev/ttyUSB0}"
[ -e "$uart" ] || { echo "no UART adapter at $uart (set UART=...)" >&2; exit 1; }
build="$variant-$precision"
mkdir -p "$proj/results"
out="$proj/results/$build.txt"

cd "$proj"
python3 python/export_model.py --variant "$variant" --precision "$precision" "$@" 2>&1 \
    | grep '^\[' | grep -v 'wrote\|ops:'
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

# A-fp32 takes about 50 s: 20 + 11 + 1 inferences at 1.4 s each, plus
# the slower stepped one. Every other build finishes in seconds.
for _ in $(seq 150); do
    grep -q -e "--- Done ---" -e "ERROR" "$tmp" && break
    sleep 1
done
tr -d '\r' < "$tmp" | sed -n '/--- Quantization lab/,$p' > "$out"
cat "$out"
grep -q -e "--- Done ---" "$tmp" || { echo "(no '--- Done ---' within 150 s)" >&2; exit 1; }
echo
python3 python/compare_preds.py "$build"
