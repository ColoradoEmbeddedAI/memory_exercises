/*
 * main.c — DMA Overlap & Bus Contention lab
 * Embedded AI course — Data Movement & Bandwidth Bottlenecks (Topic 4)
 * Board: STM32F4DISCOVERY (STM32F407VG)
 *
 * Part 1: Does DMA overlap actually deliver on its promise?
 *   Compares a blocking CPU copy-then-compute sequence against a
 *   DMA-driven copy running concurrently with the same compute work.
 *
 * Part 2: Bus contention, made visible.
 *   Runs the same CPU read-modify-write workload with and without a
 *   background DMA transfer active, once with the workload buffer in
 *   main SRAM (DMA-reachable, contends for the same AHB port) and once
 *   in CCM RAM (CPU-only port, structurally insulated from DMA traffic).
 *
 * This is a bare-metal, register-level port of the project's original
 * HAL-based sketch (the old top-level dma_bandwidth_lab.c): like every
 * other project in this repo, there is no HAL/CMSIS here, only direct
 * register access via stm32f407.h -- same style as
 * ../../01_memtypes/memory_benchmark/src/main.c, which this file borrows
 * clock_init()/DWT/UART-output plumbing from almost verbatim. See
 * README.md ("What changed from the HAL sketch") for the specific
 * HAL-call -> register mapping, especially for DMA.
 *
 * No printf, for the same reason as memory_benchmark: newlib-nano's
 * snprintf/vsnprintf need FILE* infrastructure this bare-metal build
 * doesn't provide (see syscalls.c), and pulling in float formatting for
 * one lab isn't worth it. report()/uart_put_signed_pct1() below do their
 * own fixed-point integer formatting instead.
 *
 * Clock: HSE -> PLL -> 168 MHz, same as memory_benchmark and for the same
 * reason (this course's default 16 MHz HSI-only config needs zero flash
 * wait states, which isn't representative of a real deployment clock).
 *
 * IMPORTANT — sizing: N_BG in Part 2 must be large enough that the
 * background DMA transfer is STILL RUNNING when the CPU workload
 * finishes, or you're not actually measuring contention. main() below
 * prints a [warning] if this happens; see the companion activity
 * handout (lecture4-activity-dma-bandwidth-handout.md) for tuning
 * guidance.
 */

#include <stdint.h>
#include "stm32f407.h"

#define CLOCK_HZ  168000000UL

/* ---- Clock / peripheral init (identical to memory_benchmark) --------- */

static void clock_init(void) {
    /* 1. Enable HSE and wait for it to be stable. */
    RCC->CR |= RCC_CR_HSEON;
    while (!(RCC->CR & RCC_CR_HSERDY));

    /* 2. Flash: 5 wait states at 168 MHz / 3.3V (RM0090 Table 10), plus
     * the ART Accelerator's prefetch and instruction/data caches. */
    FLASH->ACR = FLASH_ACR_LATENCY(5) | FLASH_ACR_PRFTEN | FLASH_ACR_ICEN | FLASH_ACR_DCEN;

    /* 3. Main PLL: HSE -> /8 (PLLM) -> x336 (PLLN) -> /2 (PLLP) = 168 MHz. */
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
     * mantissa=22=0x16, fraction=round(0.786*16)=13=0xD -> BRR=0x16D. */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    GPIOA->MODER  &= ~(0x3U << (2 * 2));
    GPIOA->MODER  |=  (GPIO_MODER_AF << (2 * 2));
    GPIOA->AFR[0] &= ~(0xFU << (2 * 4));
    GPIOA->AFR[0] |=  (7U << (2 * 4));   /* AF7 = USART2 */

    RCC_APB1ENR |= RCC_APB1ENR_USART2EN;
    USART2->BRR = 0x16DU;
    USART2->CR1 = USART_CR1_UE | USART_CR1_TE;
}

/* ---- UART output (no printf -- see file header) ------------------------ */

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

/* Prints a signed percentage with 1 implied decimal place, e.g.
 * pct_x10 = -37 prints "-3.7", pct_x10 = 412 prints "+41.2". Unlike the
 * original HAL sketch (which always printed a literal '+'), this derives
 * the sign from the actual value -- a small or noisy CCM "penalty" can
 * legitimately come out negative (see the activity handout's discussion
 * questions), and the output should say so rather than lie about it. */
static void uart_put_signed_pct1(int32_t pct_x10) {
    if (pct_x10 < 0) {
        uart_putchar('-');
        pct_x10 = -pct_x10;
    } else {
        uart_putchar('+');
    }
    uart_put_uint((uint32_t)(pct_x10 / 10));
    uart_putchar('.');
    uart_put_uint((uint32_t)(pct_x10 % 10));
}

/* ---- DWT cycle counter -------------------------------------------------- */

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

/* ---- DMA2 Stream0, memory-to-memory -------------------------------------
 *
 * Register-level equivalent of the HAL sketch's DMA_M2M_Init() +
 * HAL_DMA_Start() + HAL_DMA_PollForTransfer(). Memory-to-memory transfers
 * on the F4 must use DMA2 (DMA1 doesn't support M2M -- RM0090 §9.3.3),
 * and M2M mode disallows circular mode, which is why this lab uses
 * one-shot transfers and Part 2 needs the N_BG tuning step.
 *
 * For M2M, HAL's (SrcAddress, DstAddress) map to (PAR, M0AR) respectively
 * -- that's not a "peripheral" register in the usual sense here, just the
 * name the DMA engine's address-generation logic uses internally.
 */

static void dma2_clock_init(void) {
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
}

static void dma_m2m_start(const volatile void *src, volatile void *dst, uint32_t n_words) {
    DMA_Stream_t *s = &DMA2->STREAM[0];

    /* Clear stale event flags from the previous transfer on this stream
     * before reconfiguring -- not required for EN-based completion
     * detection to work, but cheap and avoids stale flags confusing a
     * future extension that does use interrupts. */
    DMA2->LIFCR = DMA_LIFCR_CFEIF0 | DMA_LIFCR_CDMEIF0 | DMA_LIFCR_CTEIF0
                | DMA_LIFCR_CHTIF0 | DMA_LIFCR_CTCIF0;

    s->CR = 0;                         /* most CR fields are only writable
                                           while EN = 0 */
    while (s->CR & DMA_SxCR_EN) { }    /* should already be clear here --
                                           the caller is expected to have
                                           waited for the previous transfer */

    s->PAR  = (uint32_t)src;
    s->M0AR = (uint32_t)dst;
    s->NDTR = n_words;
    s->FCR  = DMA_SxFCR_DMDIS | DMA_SxFCR_FTH_FULL;   /* FIFO required for M2M */
    s->CR   = DMA_SxCR_DIR_M2M | DMA_SxCR_PINC | DMA_SxCR_MINC
            | DMA_SxCR_PSIZE_WORD | DMA_SxCR_MSIZE_WORD | DMA_SxCR_PL_HIGH;
    s->CR  |= DMA_SxCR_EN;
}

/* Hardware auto-clears EN when a non-circular stream's transfer completes
 * (RM0090 §9.3.15) -- checking EN is a simpler completion test here than
 * polling/clearing TCIF0, and it's exactly what a non-blocking
 * "already done?" check (Part 2's N_BG tuning warning) needs. */
static inline int dma_m2m_done(void) {
    return (DMA2->STREAM[0].CR & DMA_SxCR_EN) == 0;
}

static void dma_m2m_wait(void) {
    while (!dma_m2m_done()) { }
}

/* ---- Generic CPU workload: read-modify-write over a buffer ------------- */

static uint32_t run_workload(volatile uint32_t *buf, uint32_t n_words, uint32_t reps) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    for (uint32_t r = 0; r < reps; r++) {
        for (uint32_t i = 0; i < n_words; i++) {
            buf[i] = buf[i] + 1u;
        }
    }

    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

/* =========================================================================
 * Part 1: DMA overlap
 * ========================================================================= */

#define N_XFER          2048u   /* words moved by the DMA copy (8 KB) */
#define N_COMPUTE        512u   /* words in the "compute" workload buffer */
#define N_COMPUTE_REPS   400u

static uint32_t src_buf[N_XFER];
static uint32_t dst_buf[N_XFER];
static volatile uint32_t compute_buf[N_COMPUTE];

/* The "compute" work, factored into one function that every Part 1
 * measurement calls, rather than each measurement writing out its own
 * copy of the same two `for` loops.
 *
 * This isn't just style: an earlier version of this file (like the
 * original HAL sketch it replaces) duplicated this loop inline inside
 * blocking_copy_then_compute() and dma_overlap_copy_and_compute(),
 * textually identical to the loop inside run_workload(). At -O2, GCC
 * compiled the two copies differently anyway -- run_workload()'s
 * argument-passed buffer pointer got a tight post-increment addressing
 * mode (ldr/str/cmp/bne, no separate index increment), while the
 * duplicated loop, inlined directly into main() alongside the memcpy
 * call and IRQ-disable bookkeeping, kept an explicit array-index
 * register and compiled one instruction longer per iteration. That ~2
 * cycles/iteration difference, times 204,800 iterations per
 * measurement, added ~400,000 cycles of pure measurement artifact to
 * `blocking` and `overlap` relative to the `compute_alone` reference
 * they're supposed to be compared against -- found by actually flashing
 * this and reading the numbers back, not by inspection. Routing all
 * three measurements through this one function keeps them comparable:
 * same source, same call site pattern, same generated code. */
static void compute_workload(void) {
    for (uint32_t r = 0; r < N_COMPUTE_REPS; r++) {
        for (uint32_t i = 0; i < N_COMPUTE; i++) {
            compute_buf[i] = compute_buf[i] + 1u;
        }
    }
}

/* Reference measurement: DMA copy alone, nothing else happening. */
static uint32_t copy_alone(void) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    dma_m2m_start(src_buf, dst_buf, N_XFER);
    dma_m2m_wait();

    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

/* Reference measurement: compute alone, nothing else happening. */
static uint32_t compute_alone(void) {
    disable_irq();
    uint32_t start = DWT_Cycles();
    compute_workload();
    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

/* Method 1: blocking -- CPU copies the buffer itself, then computes. */
static uint32_t blocking_copy_then_compute(void) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    for (uint32_t i = 0; i < N_XFER; i++) {
        dst_buf[i] = src_buf[i];
    }
    compute_workload();

    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

/* Method 2: DMA does the copy while the CPU computes concurrently.
 * IRQs are only disabled around the compute loop (matching the original
 * sketch) -- disabling them around dma_m2m_start()/dma_m2m_wait() too
 * would also block the DMA engine's own completion signaling path in a
 * more elaborate (interrupt-driven) version of this lab, even though
 * this particular version only polls. */
static uint32_t dma_overlap_copy_and_compute(void) {
    uint32_t start = DWT_Cycles();

    dma_m2m_start(src_buf, dst_buf, N_XFER);

    disable_irq();
    compute_workload();
    enable_irq();

    /* In case compute finished before the DMA copy did, wait for it. */
    dma_m2m_wait();

    uint32_t end = DWT_Cycles();
    return end - start;
}

/* =========================================================================
 * Part 2: Bus contention
 * ========================================================================= */

#define N_BG            12000u  /* background DMA transfer size (~46.9 KB
                                    per buffer) -- tune so it outlasts the
                                    workload below; near the practical max
                                    for this SRAM budget (see README.md) */
#define N_WORKLOAD       512u
#define N_WORKLOAD_REPS  8u

static uint32_t bg_src[N_BG];
static uint32_t bg_dst[N_BG];

/* Diagnostic for the activity handout's Part 2 tuning check: how long the
 * background transfer alone takes, so N_BG/N_WORKLOAD_REPS can be tuned
 * against the "DMA idle" workload baseline printed right after it,
 * instead of by trial and error against the [warning] alone. */
static uint32_t bg_copy_alone(void) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    dma_m2m_start(bg_src, bg_dst, N_BG);
    dma_m2m_wait();

    uint32_t end = DWT_Cycles();
    enable_irq();
    return end - start;
}

static volatile uint32_t workload_sram[N_WORKLOAD];   /* default: main SRAM */

__attribute__((section(".ccmram")))
static volatile uint32_t workload_ccm[N_WORKLOAD];    /* CCM RAM, CPU-only port */

static uint32_t measure_workload(volatile uint32_t *buf, int with_background_dma) {
    if (with_background_dma) {
        dma_m2m_start(bg_src, bg_dst, N_BG);
    }

    uint32_t cycles = run_workload(buf, N_WORKLOAD, N_WORKLOAD_REPS);

    if (with_background_dma) {
        /* If the transfer is already done, N_BG needs to be bigger for a
         * valid contention measurement (see the activity handout). */
        if (dma_m2m_done()) {
            uart_puts("  [warning] background DMA finished before the workload did -- "
                      "increase N_BG\r\n");
        }
        /* Let it fully finish before starting the next test. */
        dma_m2m_wait();
    }

    return cycles;
}

/* ---- Reporting ----------------------------------------------------------
 *
 * pct_x10 = (with - idle) * 1000 / idle, i.e. the percentage change with
 * one implied decimal digit -- int64_t throughout so the intermediate
 * *1000 can't overflow for any cycle count this lab produces.
 */
static void report_contention(const char *label, uint32_t idle, uint32_t with_dma) {
    int64_t delta = (int64_t)with_dma - (int64_t)idle;
    int32_t pct_x10 = (int32_t)((delta * 1000) / (int64_t)idle);

    uart_puts(label);
    uart_puts(", DMA idle    : ");
    uart_put_uint(idle);
    uart_puts(" cycles\r\n");

    uart_puts(label);
    uart_puts(", DMA active  : ");
    uart_put_uint(with_dma);
    uart_puts(" cycles  (");
    uart_put_signed_pct1(pct_x10);
    uart_puts("%)\r\n");
}

/* ---- Main ---------------------------------------------------------------- */

int main(void) {
    clock_init();
    dma2_clock_init();
    DWT_Init();

    uart_puts("\r\n--- Data Movement & Bandwidth Lab: STM32F407VG ---\r\n");
    uart_puts("Clock = 168000000 Hz (HSE -> PLL)\r\n\r\n");

    uart_puts("=== Part 1: DMA Overlap ===\r\n");
    uint32_t t_copy     = copy_alone();
    uint32_t t_compute  = compute_alone();
    uint32_t t_blocking = blocking_copy_then_compute();
    uint32_t t_overlap  = dma_overlap_copy_and_compute();

    uart_puts("Copy alone                          : ");
    uart_put_uint(t_copy);
    uart_puts(" cycles\r\n");

    uart_puts("Compute alone                       : ");
    uart_put_uint(t_compute);
    uart_puts(" cycles\r\n");

    uart_puts("Naive sum prediction (copy+compute) : ");
    uart_put_uint(t_copy + t_compute);
    uart_puts(" cycles\r\n");

    uart_puts("Blocking, sequential                : ");
    uart_put_uint(t_blocking);
    uart_puts(" cycles\r\n");

    uart_puts("Overlapped (DMA copy || compute)    : ");
    uart_put_uint(t_overlap);
    uart_puts(" cycles\r\n");

    uart_puts("\r\n=== Part 2: Bus Contention ===\r\n");
    uint32_t t_bg = bg_copy_alone();
    uart_puts("Background DMA duration (alone)     : ");
    uart_put_uint(t_bg);
    uart_puts(" cycles\r\n\r\n");

    uint32_t sram_no_dma   = measure_workload(workload_sram, 0);
    uint32_t sram_with_dma = measure_workload(workload_sram, 1);
    uint32_t ccm_no_dma    = measure_workload(workload_ccm, 0);
    uint32_t ccm_with_dma  = measure_workload(workload_ccm, 1);

    report_contention("SRAM workload", sram_no_dma, sram_with_dma);
    report_contention("CCM  workload", ccm_no_dma, ccm_with_dma);

    uart_puts("\r\n--- Done ---\r\n");

    /* Heartbeat -- confirms the board is still alive after printing, for
     * anyone without a UART adapter handy. */
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIODEN;
    GPIOD->MODER &= ~(0x3U << (12 * 2));
    GPIOD->MODER |=  (GPIO_MODER_OUTPUT << (12 * 2));
    while (1) {
        GPIOD->ODR ^= (1U << 12);
        for (volatile uint32_t i = 0; i < 4000000U; i++) { }
    }
}
