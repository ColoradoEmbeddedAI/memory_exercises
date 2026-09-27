# Activity 1: Reading Your `.map` File
### Embedded AI — Memory Hierarchy & Types
### Board: STM32F4DISCOVERY (STM32F407VG)

Goal: prove, from build artifacts rather than assumption, where a model's weights, code, and buffers actually live in memory.

## 0. Prerequisites

None. This activity hands you `firmware.elf` and `firmware.map` — that's the *only* thing you need — and asks you to reverse-engineer everything else about the firmware's memory layout from them. No source code, no ExecuTorch toolchain, no build step.

## 1. Open the `.map` file and find the "Memory Configuration" section

Open `firmware.map` in a text editor (it's over 6,000 lines — use search, don't read linearly). Search for `Memory Configuration` — it's on **line 1421**.

You'll find a table with four columns: **Name**, **Origin**, **Length**, **Attributes** — one row each for `FLASH`, `RAM`, and `CCMRAM`.

**Task:** Find it, then write down the Origin and Length of each region:

| Region | Origin | Length |
|---|---|---|
| FLASH  |  |  |
| RAM    |  |  |
| CCMRAM |  |  |

Confirm these match the STM32F407VG datasheet values covered in the lecture. (The lecture's datasheet table lists five regions, not three — see if you can figure out why before asking.)

## 2. Find the "Linker script and memory map" section

Search for `Linker script and memory map` — it's on **line 1429**. This marks the start of a section: everything from line 1429 through **line 4822** (where you'll see `OUTPUT(firmware.elf elf32-littlearm)`) belongs to it — about 3,400 lines. This section lists, in order, every input section from every object file and the address it was placed at. It's long — you're going to search within it for specific things, not read the whole thing.

**Task A — Find the weight data.**
Within this section (lines 1429–4822), search for `sine_model_pte`. You'll land on a line that looks like:
```
 .rodata._ZL14sine_model_pte
```
(The name looks mangled because this is C++, where a top-level `const` array gets internal linkage by default — unlike C — but the plain-text substring `sine_model_pte` still matches your search.)

The next line lists two numbers: an address, then a size.

**Task:** Find that line, and write both down:

Address: ______________________

Size: ______________________

**Task:** Which memory region does this address fall in? (Check your Section 1 table.)

Region: ______________________

**Task:** That size, in decimal, should look like a familiar kind of number if you know what file this project embeds. What is it the size of?

______________________

**Task B — Find the memory arena(s).**
Search for `method_pool`, then search for `planned_pool` (these are the two buffers the firmware's `MemoryAllocator`/`HierarchicalAllocator` use for the model's method state and activation buffers, respectively — see the lecture's discussion of where ExecuTorch's memory arena lives).

**Neither search finds a named line. Why not?** Look instead for a block like this:
```
.ccmram         0x10000000     0xf800
                0x10000000                        _sccmram = .
 *(.ccmram)
 .ccmram        0x10000000     0xf800 CMakeFiles/firmware.elf.dir/src/main.cpp.obj
```
This is the entire `.ccmram` custom section from `main.cpp.obj` as *one* merged block. Write down your explanation for why `method_pool` and `planned_pool` don't show up individually here (hint: think about what granularity this listing works at, and how this custom section differs from `.bss`/`.data`, which *do* show individual symbols like `.bss._ZL7g_ticks` a few pages earlier):

______________________________________________________________
______________________________________________________________

To find each pool's individual address and size, you need a different tool:
```
arm-none-eabi-nm -C -S firmware.elf | grep pool
```
`nm` lists every symbol in an object file or executable. With `-S` (print size) and `-C` (demangle C++ names), each output line has four columns: **address**, **size**, **type** (a one-letter code — lowercase means local/internal linkage), and **name**.

**Task:** Run it, and record both addresses and sizes:

| Symbol | Address | Size |
|---|---|---|
| `method_pool` |  |  |
| `planned_pool` |  |  |

Confirm both fall in the `0x1000xxxx` range (CCM RAM). Do the two sizes add up to the merged block's `0xf800` from the `.map` file? (They should — that's a good self-check that you found the right two symbols.)

Sum: ______________________ (matches `0xf800`? Y / N)

**Task C — Find `.data` and note its addresses.**
Search for the `.data` section header line — it's on **line 4729**. You'll see three numbers on it:
- a runtime address (where `.data` lives in SRAM once the program is executing)
- a size (the section's length in bytes)
- a second address, labeled "load address"

**Task:** Find that line, and write down all three:

Runtime address: ______________________

Size: ______________________

Load address: ______________________

**Task:** In your own words (one or two sentences), why does `.data` need *two* addresses instead of one? (Hint: think about what's volatile and what isn't, and what has to survive a reset.)

______________________________________________________________
______________________________________________________________
______________________________________________________________

**Task:** Within `.data`, find this project's own initialized global variables — the ones that come from `main.cpp`, not the toolchain or the ExecuTorch runtime. (Hint: there are three of them, and they're all small — tensor shape metadata.) List each one's name and size, then add them up:

| Name | Size (bytes) |
|---|---|
|  |  |
|  |  |
|  |  |
| **Total** |  |

## 3. Cross-check with `size`

Recall from **Section 2, Task B** you already found `.ccmram`'s size, and from **Task C** you found `.data`'s size — you'll be able to cross-check both against what `size` reports below.

From a terminal, run:
```
arm-none-eabi-size -A firmware.elf
```
(the `-A` gives a section-by-section breakdown rather than just totals). Compare `.text`, `.data`, `.bss`, and `.ccmram` here against what you found manually in Tasks A–C. They should be consistent.

**Task:** Using these numbers, compute:

Total Flash consumed = `.isr_vector` + `.text` + `.ARM.extab` + `.ARM.exidx` + `.init_array` + `.data` (the `.data` load copy counts against Flash too)

= ______________________ bytes of 1,048,576 (1 MB) → ______________________ % used

Total main-SRAM consumed at runtime = `.data` + `.bss`

= ______________________ bytes of 131,072 (128 KB) → ______________________ % used

(Your expected stack/heap usage won't show up here — why not?)

______________________________________________________________

Total CCM RAM consumed = `.ccmram`

= ______________________ bytes of 65,536 (64 KB) → ______________________ % used

**Then answer:** which of the three regions is closest to full? Is that the one you'd have guessed?

______________________________________________________________

## 4. Find the largest symbols

Run:
```
arm-none-eabi-nm --size-sort -C firmware.elf | tail -30
```
This lists the 30 largest symbols in the binary, smallest of the 30 first (`-C` demangles C++ names).

**Task:** Identify which of these are:
- NN weight/constant data
- ExecuTorch runtime code or operator kernels (e.g. `Method::init`, `addmm_out`, `permute_copy_out`)
- This project's own application code (there's very little of it — see if you can even find `main` in the list, and note how small it is relative to everything else)
- Statically allocated buffers (the memory arena(s), kernel registration tables, etc.)

Write your categorization below (there will be some symbols left over that don't fit any of the four — note those too, and what you think they are):

NN weight/constant data: ______________________________________________________________

ExecuTorch runtime code/kernels: ______________________________________________________________

This project's own code: ______________________________________________________________

Statically allocated buffers: ______________________________________________________________

Everything else (what is it?): ______________________________________________________________
