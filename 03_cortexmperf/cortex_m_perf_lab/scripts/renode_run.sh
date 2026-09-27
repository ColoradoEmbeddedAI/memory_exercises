#!/usr/bin/env bash
# Runs one build of the lab under Renode and reports its exact
# INSTRUCTIONS PER MAC -- see ../README.md ("Running under Renode").
#
# Usage (from anywhere, after building that preset):
#   scripts/renode_run.sh f32-hard      # or f32-soft, q15-scalar, q15-simd
#
# How it works: Renode's DWT model derives CYCCNT from simulated time. With
# the simulated CPU set to 168 MIPS (one instruction per 168 MHz tick), the
# cycle counter advances by exactly one per executed instruction, so every
# "cycles" figure the firmware prints under Renode is an instruction count.
# Renode does not model pipeline timing, so it can't give real cycles --
# that's what the board is for.
#
# Assumes Renode's portable release at ~/renode_portable with its venv at
# ~/renode_portable/renode-venv (override the directory with RENODE_HOME).
set -euo pipefail

preset="${1:-}"
case "$preset" in
    f32-hard|f32-soft|q15-scalar|q15-simd) ;;
    *) echo "usage: $0 {f32-hard|f32-soft|q15-scalar|q15-simd}" >&2; exit 2 ;;
esac

proj="$(cd "$(dirname "$0")/.." && pwd)"
elf="$proj/build/$preset/firmware.elf"
[ -f "$elf" ] || {
    echo "no $elf -- build it first: cmake --preset $preset && cmake --build --preset $preset" >&2
    exit 1
}

renode_home="${RENODE_HOME:-$HOME/renode_portable}"
export PATH="$renode_home/renode-venv/bin:$renode_home:$PATH"
command -v renode >/dev/null || { echo "renode not found in $renode_home" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# Generated with absolute paths: Renode resolves a relative path for a file
# it *creates* (the UART capture) against its own install directory.
cat > "$tmp/run.resc" <<RESC
mach create
machine LoadPlatformDescription @$proj/platforms/stm32f4_discovery_full.repl
cpu PerformanceInMips 168
sysbus LoadELF @$elf
usart2 CreateFileBackend @$tmp/uart.txt true
# 2 simulated seconds covers the slowest build (f32-soft, ~0.3 s of
# instructions) with room to spare; the firmware then idles.
emulation RunFor "2"
quit
RESC

echo "Running $preset under Renode (takes about 10 seconds)..."

# stdin from /dev/null: with --console, Renode runs an interactive monitor
# on whatever terminal stdin is attached to, and when that's a real
# terminal it never gets to 'quit' by itself. Without a terminal it runs
# the script to the end and exits.
# timeout: a normal run takes ~6 s. If a command above fails, Renode drops
# to its monitor instead of reaching 'quit' -- don't hang if it does.
if ! timeout 60 renode --disable-gui --console "$tmp/run.resc" \
        < /dev/null > "$tmp/renode.log" 2>&1; then
    echo "Renode failed or timed out; its log:" >&2
    cat "$tmp/renode.log" >&2
    exit 1
fi

echo "=== Renode run of $preset (simulated: 'cycles' below are INSTRUCTION COUNTS) ==="
tr -d '\r' < "$tmp/uart.txt"

ipm="$(tr -d '\r' < "$tmp/uart.txt" | sed -n 's/.*| *\([0-9.]*\) cycles\/MAC.*/\1/p' | head -1)"
echo
if [ -n "$ipm" ]; then
    echo ">>> Instructions per MAC (Renode, exact): $ipm"
else
    echo ">>> Could not find a result line in the Renode output." >&2
    exit 1
fi
