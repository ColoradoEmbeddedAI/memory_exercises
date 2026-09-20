/*
 * main.c — Activity 2 companion firmware ("Measuring Memory Speed")
 * Embedded AI course — Memory Hierarchy & Types
 * Board: STM32F4DISCOVERY (STM32F407VG)
 *
 * Measures read/write latency (in CPU cycles, via the DWT cycle counter)
 * for three memory regions:
 *   - SRAM    (main on-chip RAM, 0x2000xxxx, DMA-accessible)
 *   - CCM RAM (core-coupled RAM, 0x1000xxxx, CPU-only, no DMA)
 *   - Flash   (0x0800xxxx, read-only in this activity)
 *
 * This is a from-scratch, bare-metal port of the original
 * lecture1-activity-memory-benchmark-handout.md companion sketch: this repo's
 * other projects use direct register access (no HAL/CMSIS), so
 * HAL_Init()/SystemClock_Config()/CoreDebug/DWT-via-CMSIS and
 * __disable_irq()/__enable_irq() intrinsics are all replaced with the
 * same register-map style as lab0_hello_world_soln and wakeword_et_dev
 * (see stm32f407.h).
 *
 * Clock: HSE -> PLL -> 168 MHz (see clock_init() below), not the 16 MHz
 * HSI-only config most of this repo's simpler demos use. At 16 MHz the
 * STM32F407 needs zero flash wait states, which would make Flash reads
 * look identical to SRAM and defeat the point of this activity (see
 * Slide 14 in the lecture deck on the ART Accelerator). At 168 MHz,
 * Flash needs 5 wait states (RM0090 Table 10) and the ART Accelerator's
 * prefetch/cache actually has something to hide -- that's what Section 6,
 * Question 3's sequential-vs-strided extension is probing.
 *
 * No printf: newlib-nano's snprintf/vsnprintf need FILE* infrastructure
 * this bare-metal build doesn't have (see syscalls.c), and
 * we don't want floating point in the report path either. report() below
 * does its own fixed-point (2 decimal place) integer formatting instead.
 */

#include <stdint.h>
#include "stm32f407.h"

/* ---- Configuration ------------------------------------------------- */

#define CLOCK_HZ  168000000UL

#define N_WORDS   1024u   /* words per buffer (4 KB) — fits comfortably
                              in all three regions on the F407VG */
#define N_REPS    2000u   /* repetitions per measurement, to amortize
                              loop-overhead and get a stable reading */

/* ---- Buffers, one per memory region --------------------------------- */

/* Main SRAM: no special placement needed, this is the default region
 * for globals. */
static volatile uint32_t sram_buf[N_WORDS];

/* CCM RAM: requires the .ccmram section in the linker script (present in
 * this project's stm32f407.ld -- see README.md, "CCM RAM: already wired
 * up here", for what that section looks like and why a default project
 * might be missing it). */
__attribute__((section(".ccmram")))
static volatile uint32_t ccm_buf[N_WORDS];

/* Flash: a const array is placed in .rodata, which lives in Flash and
 * is read via XIP — no copy to RAM happens for a const like this. */
static const uint32_t flash_buf[N_WORDS] = {
    /* Content doesn't matter for a latency measurement; a real pattern
     * (not just zero-init) so the compiler can't reason the array away. */
    [0 ... N_WORDS - 1] = 0xA5A5A5A5u
};

/* ---- Clock / peripheral init ----------------------------------------- */

static void clock_init(void) {
    /* 1. Enable HSE (8 MHz, from the ST-LINK's MCO on this board -- same
     * source wakeword_et_dev/src/hal.c uses) and wait for it to be
     * stable. */
    RCC->CR |= RCC_CR_HSEON;
    while (!(RCC->CR & RCC_CR_HSERDY));

    /* 2. Configure Flash: 5 wait states required at 168 MHz / 3.3V
     * (RM0090 Table 10). Also enable the ART Accelerator's prefetch and
     * instruction/data caches -- this is the thing Activity 2 Question 3
     * and the strided-access extension are about. */
    FLASH->ACR = FLASH_ACR_LATENCY(5) | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;

    /* 3. Configure main PLL: HSE -> /8 (PLLM) -> x336 (PLLN) -> /2 (PLLP)
     * = 168 MHz. */
    RCC->PLLCFGR = (8U)             /* PLLM */
                 | (336U << 6)      /* PLLN */
                 | (0U << 16)       /* PLLP: 0 = /2 */
                 | RCC_PLLCFGR_PLLSRC_HSE;

    /* 4. Enable PLL and wait for lock. */
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY));

    /* 5. AHB = /1 (168 MHz), APB1 = /4 (42 MHz), APB2 = /2 (84 MHz),
     * switch SYSCLK to PLL. */
    RCC->CFGR = RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS_MASK) != RCC_CFGR_SWS_PLL);

    /* 6. GPIOA (USART2 TX pin) + USART2 (PA2 = TX, AF7), 115200 8N1 at
     * 42 MHz APB1: BRR = 42000000 / (16*115200) = 22.786 ->
     * mantissa=22=0x16, fraction=round(0.786*16)=13=0xD -> BRR=0x16D
     * (same derivation as wakeword_et_dev/src/hal.c). */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER  &= ~(0x3U << (2 * 2));
    GPIOA->MODER  |=  (GPIO_MODER_AF << (2 * 2));
    GPIOA->AFR[0] &= ~(0xFU << (2 * 4));
    GPIOA->AFR[0] |=  (7U << (2 * 4));   /* AF7 = USART2 */

    RCC_APB1ENR |= RCC_APB1ENR_USART2EN;
    USART2->BRR = 0x16DU;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE;
}

/* ---- UART output (no printf -- see file header) ----------------------- */

static void uart_putchar(char c) {
    while (!(USART2->SR & USART_SR_TXE));
    USART2->DR = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) uart_putchar(*s++);
    while (!(USART2->SR & USART_SR_TC));
}

static void uart_put_uint(uint32_t v) {
    char tmp[10];
    int n = 0;
    if (v == 0) {
        uart_putchar('0');
        return;
    }
    while (v) {
        tmp[n++] = (char)('0' + (v % 10U));
        v /= 10U;
    }
    while (n) uart_putchar(tmp[--n]);
}

/* Prints v as a fixed-point value with 2 implied decimal places, e.g.
 * uart_put_fixed2(12345) prints "123.45". */
static void uart_put_fixed2(uint32_t v) {
    uart_put_uint(v / 100U);
    uart_putchar('.');
    uint32_t frac = v % 100U;
    if (frac < 10U) uart_putchar('0');
    uart_put_uint(frac);
}

/* ---- DWT cycle counter ------------------------------------------------ */

static void DWT_Init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA;
}

static inline uint32_t DWT_Cycles(void) {
    return DWT->CYCCNT;
}

/* No CMSIS on this build, so disable/enable IRQ by hand instead of
 * __disable_irq()/__enable_irq(). */
static inline void disable_irq(void) { __asm volatile ("cpsid i" ::: "memory"); }
static inline void enable_irq(void)  { __asm volatile ("cpsie i" ::: "memory"); }

/* ---- Benchmark primitives --------------------------------------------
 *
 * Each function does N_REPS full passes over an N_WORDS buffer and
 * returns total elapsed cycles. Interrupts are disabled around the
 * timed region so a preempting ISR can't inflate the count.
 *
 * The inner loop is manually unrolled 10-wide: with 1 memory op per
 * iteration, the loop's own bookkeeping (increment/compare/branch, ~3
 * more instructions -- see README.md "What the inner loop actually
 * costs") is a sizeable fraction of every measured access. Unrolling
 * amortizes that fixed overhead over 10 memory ops instead of 1, so
 * cycles/access converges closer to the true per-access memory cost. A
 * short tail loop handles n_words not being a multiple of 10 (it isn't,
 * for N_WORDS=1024) without changing how many words get touched per rep.
 * ---------------------------------------------------------------------- */

static uint32_t bench_write(volatile uint32_t *buf, uint32_t n_words, uint32_t reps) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    for (uint32_t r = 0; r < reps; r++) {
        uint32_t i = 0;
        for (; i + 10 <= n_words; i += 10) {
            buf[i + 0] = i + 0;
            buf[i + 1] = i + 1;
            buf[i + 2] = i + 2;
            buf[i + 3] = i + 3;
            buf[i + 4] = i + 4;
            buf[i + 5] = i + 5;
            buf[i + 6] = i + 6;
            buf[i + 7] = i + 7;
            buf[i + 8] = i + 8;
            buf[i + 9] = i + 9;
        }
        for (; i < n_words; i++) {
            buf[i] = i;
        }
    }

    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

static uint32_t bench_read(volatile const uint32_t *buf, uint32_t n_words, uint32_t reps) {
    /* Deliberately NOT volatile: a volatile sink forces a stack
     * spill+reload (both to main SRAM) on every single iteration, on
     * top of the buf[i] read under test -- which would mean 2 of every
     * 3 memory accesses in this loop always hit SRAM regardless of
     * which region is being benchmarked, diluting exactly the
     * SRAM/CCM/Flash difference this function exists to measure. Keep
     * sink in a register across the whole loop instead, and use a
     * compiler barrier after the loop (not volatile) to stop the loop
     * itself from being optimized away. */
    uint32_t sink = 0;

    disable_irq();
    uint32_t start = DWT_Cycles();

    for (uint32_t r = 0; r < reps; r++) {
        uint32_t i = 0;
        for (; i + 10 <= n_words; i += 10) {
            sink += buf[i + 0];
            sink += buf[i + 1];
            sink += buf[i + 2];
            sink += buf[i + 3];
            sink += buf[i + 4];
            sink += buf[i + 5];
            sink += buf[i + 6];
            sink += buf[i + 7];
            sink += buf[i + 8];
            sink += buf[i + 9];
        }
        for (; i < n_words; i++) {
            sink += buf[i];
        }
    }

    uint32_t end = DWT_Cycles();
    enable_irq();

    /* No-op asm that declares sink as both input and output: tells GCC
     * the accumulated value escapes, so the read loop above can't be
     * proven dead and removed, without forcing any real memory traffic. */
    __asm volatile ("" : "+r" (sink));

    return end - start;
}

/* ---- Reporting -------------------------------------------------------- */

static void report(const char *label, uint32_t cycles, uint32_t n_words, uint32_t reps) {
    uint32_t total_accesses = n_words * reps;
    uint32_t us_x100  = (uint32_t)(((uint64_t)cycles * 100U) / (CLOCK_HZ / 1000000UL));
    uint32_t cpa_x100 = (uint32_t)(((uint64_t)cycles * 100U) / total_accesses);

    uart_puts(label);
    uart_puts(": ");
    uart_put_uint(cycles);
    uart_puts(" cycles total | ");
    uart_put_fixed2(us_x100);
    uart_puts(" us total | ");
    uart_put_fixed2(cpa_x100);
    uart_puts(" cycles/access\r\n");
}

/* ---- Main -------------------------------------------------------------- */

int main(void) {
    clock_init();
    DWT_Init();

    uart_puts("\r\n--- Memory Benchmark: STM32F407VG ---\r\n");
    uart_puts("Clock = 168000000 Hz (HSE -> PLL)\r\n");
    uart_puts("Buffer size = 1024 words (4096 bytes), 2000 reps\r\n\r\n");

    /* Warm-up pass (not timed) — lets the ART Accelerator's prefetch
     * buffer settle before the first real Flash measurement, so the
     * first result isn't penalized by a one-time cold-start cost. */
    (void)bench_read(flash_buf, N_WORDS, 1);

    uint32_t c;

    c = bench_write(sram_buf, N_WORDS, N_REPS);
    report("SRAM write    ", c, N_WORDS, N_REPS);

    c = bench_read(sram_buf, N_WORDS, N_REPS);
    report("SRAM read     ", c, N_WORDS, N_REPS);

    c = bench_write(ccm_buf, N_WORDS, N_REPS);
    report("CCM RAM write ", c, N_WORDS, N_REPS);

    c = bench_read(ccm_buf, N_WORDS, N_REPS);
    report("CCM RAM read  ", c, N_WORDS, N_REPS);

    c = bench_read(flash_buf, N_WORDS, N_REPS);
    report("Flash read    ", c, N_WORDS, N_REPS);

    uart_puts("\r\n--- Done ---\r\n");

    /* Heartbeat -- confirms the board is still alive after printing,
     * for anyone without a UART adapter handy. */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
    GPIOD->MODER &= ~(0x3U << (12 * 2));
    GPIOD->MODER |=  (GPIO_MODER_OUTPUT << (12 * 2));
    while (1) {
        GPIOD->ODR ^= (1U << 12);
        for (volatile uint32_t i = 0; i < 4000000U; i++) { }
    }
}
