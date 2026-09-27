/*
 * stm32f407.ld — Linker script for STM32F407VG (Cortex-M4)
 *
 * Memory map:
 *   FLASH:  1 MB  at 0x08000000  (code, rodata)
 *   RAM:  128 KB  at 0x20000000  (stack, .data, .bss -- fa/fb and qa/qb,
 *                                 the benchmark arrays, see src/main.c)
 *   CCMRAM: 64 KB at 0x10000000  (CPU-only fast RAM -- unused by this lab;
 *                                 the section is kept so the file stays
 *                                 identical in structure to the Topic 1
 *                                 and Topic 4 labs' linker scripts)
 *
 * Copied from ../../04_memmovement/dma_bandwidth_lab/stm32f407.ld (itself
 * from ../../01_memtypes/memory_benchmark/stm32f407.ld).
 */

MEMORY
{
  FLASH  (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM    (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
  CCMRAM (rwx) : ORIGIN = 0x10000000, LENGTH = 64K
}

ENTRY(Reset_Handler)

SECTIONS
{
  /* Vector table must be first in flash at 0x08000000 */
  .isr_vector :
  {
    KEEP(*(.isr_vector))
  } > FLASH

  /* Code and read-only data */
  .text :
  {
    *(.text*)
    *(.rodata*)
  } > FLASH

  /*
   * ARM exception unwinding tables — required by libgcc even with
   * -fno-exceptions; omitting these causes undefined __exidx_start/end.
   */
  .ARM.extab :
  {
    *(.ARM.extab*)
    *(.gnu.linkonce.armextab.*)
  } > FLASH

  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx*)
    *(.gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > FLASH

  /* C constructor/destructor tables — iterated by __libc_init_array */
  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > FLASH

  .fini_array :
  {
    PROVIDE_HIDDEN(__fini_array_start = .);
    KEEP(*(.fini_array*))
    PROVIDE_HIDDEN(__fini_array_end = .);
  } > FLASH

  /*
   * Initialized data — stored in FLASH (LMA), copied to RAM (VMA) by
   * Reset_Handler before main(). .got MUST come before .data* so the
   * Global Offset Table is covered by the startup copy loop.
   */
  .data :
  {
    _sdata = .;
    *(.got)
    *(.got.plt)
    *(.data*)
    _edata = .;
  } > RAM AT > FLASH

  /* Flash address of .data load image — used by Reset_Handler copy loop */
  _sidata = LOADADDR(.data);

  /* Zero-initialized data — zeroed by Reset_Handler before main().
   * The benchmark arrays fa/fb/qa/qb (see src/main.c) land here, in main
   * SRAM -- zero wait states, so Topic 1's Flash-latency effects don't
   * apply to the operands (instruction fetch is still from Flash). */
  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  /*
   * CCM RAM — 64KB at 0x10000000, CPU-only (no DMA access). NOLOAD: it
   * has no flash image, so Reset_Handler zeroes it the same way as .bss.
   * Nothing in this lab is placed here (empty section, zero-length loop
   * in Reset_Handler); try moving fa/fb or qa/qb here with
   * __attribute__((section(".ccmram"))) as an extension -- for a pure
   * CPU workload with no DMA traffic, Topic 4 predicts no difference.
   */
  .ccmram (NOLOAD) :
  {
    _sccmram = .;
    *(.ccmram)
    *(.ccmram*)
    _eccmram = .;
  } > CCMRAM

  /* Stack grows downward from top of RAM */
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
