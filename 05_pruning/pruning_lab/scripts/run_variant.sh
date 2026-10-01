#!/usr/bin/env bash
# Export one variant, build, flash, and capture the board's UART report.
#
# Usage (from anywhere, ExecuTorch venv active):
#   scripts/run_variant.sh B                 # UART adapter on /dev/ttyUSB0
#   UART=/dev/ttyUSB1 scripts/run_variant.sh B
#
# The report is printed and also saved to results/variant_B.txt.
# This is the same as running the steps in README.md by hand:
#   python3 python/export_variants.py --variant B
#   cmake --build --preset stm32 --target flash
#   picocom -b 115200 --imap lfcrlf /dev/ttyUSB0
set -euo pipefail

variant="${1:-}"
case "$variant" in
    A|B|C) ;;
    *) echo "usage: $0 {A|B|C}" >&2; exit 2 ;;
esac

proj="$(cd "$(dirname "$0")/.." && pwd)"
uart="${UART:-/dev/ttyUSB0}"
[ -e "$uart" ] || { echo "no UART adapter at $uart (set UART=...)" >&2; exit 1; }
mkdir -p "$proj/results"
out="$proj/results/variant_$variant.txt"

cd "$proj"
python3 python/export_variants.py --variant "$variant" 2>&1 | grep -v -i warning
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

# A full-width variant takes about 30 s: 21 inferences at 1.4 s each.
for _ in $(seq 120); do
    grep -q -e "--- Done ---" -e "ERROR" "$tmp" && break
    sleep 1
done
tr -d '\r' < "$tmp" | sed -n '/--- Pruning lab/,$p' | tee "$out"
grep -q -e "--- Done ---" "$tmp" || { echo "(no '--- Done ---' within 120 s)" >&2; exit 1; }
