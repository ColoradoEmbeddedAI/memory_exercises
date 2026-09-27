/*
 * stm32f407.ld — Linker script for STM32F407VG (Cortex-M4)
 *
 * Memory map:
 *   FLASH:  1 MB  at 0x08000000  (code, rodata)
 *   RAM:  128 KB  at 0x20000000  (stack, .data, .bss -- src_buf/dst_buf,
 *                                 compute_buf, bg_src/bg_dst, workload_sram,
 *                                 see src/main.c)
 *   CCMRAM: 64 KB at 0x10000000  (CPU-only fast RAM, no DMA path -- holds
 *                                 workload_ccm, see src/main.c)
 *
 * Copied from ../../01_memtypes/memory_benchmark/stm32f407.ld -- same
 * CCM RAM section setup (see that project's README.md, "CCM RAM: already
 * wired up here", for why a default project might be missing it).
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
   * src_buf/dst_buf, compute_buf, bg_src/bg_dst, workload_sram (see
   * src/main.c) all land here, in main SRAM (DMA-reachable). */
  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  /*
   * CCM RAM — 64KB at 0x10000000, CPU-only (no DMA access). NOLOAD: it
   * has no flash image, so Reset_Handler zeroes it the same way as .bss
   * rather than copying from flash. workload_ccm (see src/main.c) is
   * placed here via __attribute__((section(".ccmram"))) -- this is the
   * buffer Part 2 uses to test whether CCM's DMA restriction actually
   * insulates it from bus contention.
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
