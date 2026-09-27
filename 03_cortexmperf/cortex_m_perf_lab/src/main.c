/*
 * main.c — FPU and SIMD Performance lab
 * Embedded AI course — Cortex-M Performance Characteristics (Topic 3)
 * Board: STM32F4DISCOVERY (STM32F407VG, Cortex-M4F)
 *
 * Three kernels; each build times exactly ONE of them, chosen at compile
 * time by KERNEL_F32 / KERNEL_Q15_SCALAR / KERNEL_Q15_SIMD (set by the
 * CMake preset, see README.md), so every run produces one number:
 *   - float32 dot product: built with the hard float ABI (preset f32-hard)
 *     and the soft one (f32-soft). The ABI choice is compile-time, not
 *     something a running binary can toggle.
 *   - q15 (int16) dot product, scalar (preset q15-scalar)
 *   - q15 dot product using SMLAD, two MACs per instruction (q15-simd)
 *
 * This is a bare-metal, register-level port of the project's original
 * HAL-based sketch (../cortex_m_perf_lab.c): like every other project in
 * this repo, there is no HAL/CMSIS here, only direct register access via
 * stm32f407.h. Clock/UART/DWT plumbing is copied from
 * ../../04_memmovement/dma_bandwidth_lab/src/main.c (itself from
 * ../../01_memtypes/memory_benchmark). See README.md ("What changed from
 * the HAL sketch") for the call-by-call mapping.
 *
 * No printf, and -- deliberately -- no float arithmetic anywhere in the
 * reporting path: report() below formats MACs/cycle etc. with integer
 * fixed-point math. In the soft-float build every float operation is a
 * library call, and it would be a poor float benchmark if the thing printing the
 * results were itself one of the things being emulated.
 *
 * Compiled at -O2 (CMAKE_BUILD_TYPE=RelWithDebInfo). Source arrays are
 * declared volatile so the compiler can't hoist the (data-invariant) inner
 * computation out of the repetition loop -- same technique used in the
 * Topic 1 memory benchmark. A side effect worth knowing about for the
 * discussion: volatile also stops GCC from unrolling the inner loops or
 * merging adjacent loads, so every iteration pays full loop overhead.
 *
 * Clock: HSE -> PLL -> 168 MHz, same as memory_benchmark and
 * dma_bandwidth_lab, so cycle counts are comparable across Topics 1, 3, 4.
 */

#include <stdint.h>
#include <arm_acle.h>     /* __smlad -- GCC's ACLE intrinsic, standing in
                              for CMSIS's __SMLAD */
#include "stm32f407.h"

#define CLOCK_HZ  168000000UL

/* Which kernel this build times -- defined by the CMake preset (KERNEL
 * cache variable, see CMakeLists.txt). Exactly one must be set. */
#if (defined(KERNEL_F32) + defined(KERNEL_Q15_SCALAR) + defined(KERNEL_Q15_SIMD)) != 1
#error "Define exactly one of KERNEL_F32, KERNEL_Q15_SCALAR, KERNEL_Q15_SIMD -- build with a CMake preset"
#endif

/* ---- Clock / peripheral init (identical to dma_bandwidth_lab) ---------- */

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

static void uart_put_int(int32_t v) {
    if (v < 0) {
        uart_putchar('-');
        uart_put_uint((uint32_t)(-(int64_t)v));
    } else {
        uart_put_uint((uint32_t)v);
    }
}

/* Right-aligns uart_put_uint() output in a field of `width` characters,
 * standing in for printf's "%10lu". */
static void uart_put_uint_w(uint32_t v, int width) {
    int digits = 1;
    for (uint32_t t = v; t >= 10U; t /= 10U) digits++;
    while (width-- > digits) uart_putchar(' ');
    uart_put_uint(v);
}

/* Prints value / 10^decimals with exactly `decimals` fractional digits,
 * e.g. uart_put_fixed(1234, 3) prints "1.234", uart_put_fixed(7, 3)
 * prints "0.007". Stands in for printf's "%.3f" without touching float. */
static void uart_put_fixed(uint32_t value, int decimals) {
    uint32_t scale = 1;
    for (int i = 0; i < decimals; i++) scale *= 10U;
    uart_put_uint(value / scale);
    if (decimals == 0) return;
    uart_putchar('.');
    uint32_t frac = value % scale;
    for (uint32_t s = scale / 10U; s > 0; s /= 10U) {
        uart_putchar((char)('0' + (frac / s) % 10U));
    }
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

/* =====================================================================
 * float32 dot product (hard vs soft float ABI -- via build preset)
 *
 * Each bench_*() function is noinline/noclone so it is a single function
 * in the disassembly under its own name (`cmake --build <dir> --target
 * disasm-kernel`), not folded into main() (noinline) or renamed
 * bench_*.constprop.0 (noclone).
 * GCC still propagates the constant n/reps into the loop bounds, which is
 * why the disassembly compares against fixed addresses/immediates. See dma_bandwidth_lab/README.md ("A bug this port found") for
 * what inlining into surrounding context can do to a benchmark loop.
 * ===================================================================== */

#if defined(KERNEL_F32)

#define N_FLOAT       1024u   /* same length and reps as the q15 kernels, */
#define N_FLOAT_REPS   500u   /* so every build times 512,000 MACs         */

__attribute__((aligned(4)))
static volatile float fa[N_FLOAT];
__attribute__((aligned(4)))
static volatile float fb[N_FLOAT];

__attribute__((noinline, noclone))
static uint32_t bench_dot_f32(uint32_t n, uint32_t reps, float *result) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    volatile float acc = 0.0f;
    for (uint32_t r = 0; r < reps; r++) {
        float local_acc = 0.0f;
        for (uint32_t i = 0; i < n; i++) {
            local_acc += fa[i] * fb[i];
        }
        acc = local_acc; /* volatile store: prevents the whole rep loop
                             from being recognized as redundant work */
    }

    uint32_t end = DWT_Cycles();
    enable_irq();
    *result = acc;
    return end - start;
}

#endif /* KERNEL_F32 */

/* =====================================================================
 * int16 (q15-style) dot product, scalar vs SIMD (SMLAD)
 * ===================================================================== */

#if defined(KERNEL_Q15_SCALAR) || defined(KERNEL_Q15_SIMD)

#define N_Q15        1024u   /* must be even -- SIMD version processes
                                 two elements per iteration */
#define N_Q15_REPS    500u

__attribute__((aligned(4)))
static volatile int16_t qa[N_Q15];
__attribute__((aligned(4)))
static volatile int16_t qb[N_Q15];

/* Untimed reference result, so each q15 build can check its own kernel
 * computed the right answer (the SIMD kernel's packing in particular). */
__attribute__((noinline, noclone))
static int32_t q15_reference(void) {
    int32_t acc = 0;
    for (uint32_t i = 0; i < N_Q15; i++) {
        acc += (int32_t)qa[i] * (int32_t)qb[i];
    }
    return acc;
}

#endif /* KERNEL_Q15_* */

#if defined(KERNEL_Q15_SCALAR)
__attribute__((noinline, noclone))
static uint32_t bench_dot_q15_scalar(uint32_t n, uint32_t reps, int32_t *result) {
    disable_irq();
    uint32_t start = DWT_Cycles();

    volatile int32_t acc = 0;
    for (uint32_t r = 0; r < reps; r++) {
        int32_t local_acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            local_acc += (int32_t)qa[i] * (int32_t)qb[i];
        }
        acc = local_acc;
    }

    uint32_t end = DWT_Cycles();
    enable_irq();
    *result = acc;
    return end - start;
}

#endif /* KERNEL_Q15_SCALAR */

#if defined(KERNEL_Q15_SIMD)
__attribute__((noinline, noclone))
static uint32_t bench_dot_q15_simd(uint32_t n, uint32_t reps, int32_t *result) {
    /* Reinterpret two consecutive int16 elements as one packed 32-bit
     * word -- exactly the layout SMLAD expects (element i in the low
     * halfword, element i+1 in the high halfword, since this core is
     * little-endian). Arrays are 4-byte aligned above, so this is safe. */
    volatile uint32_t *pa = (volatile uint32_t *)qa;
    volatile uint32_t *pb = (volatile uint32_t *)qb;
    uint32_t n2 = n / 2u;

    disable_irq();
    uint32_t start = DWT_Cycles();

    volatile int32_t acc = 0;
    for (uint32_t r = 0; r < reps; r++) {
        int32_t local_acc = 0;
        for (uint32_t i = 0; i < n2; i++) {
            uint32_t va = pa[i]; /* read through volatile into a plain
                                     local so it can be passed to the
                                     intrinsic */
            uint32_t vb = pb[i];
            /* acc += lo(va)*lo(vb) + hi(va)*hi(vb) -- one instruction */
            local_acc = __smlad((int16x2_t)va, (int16x2_t)vb, local_acc);
        }
        acc = local_acc;
    }

    uint32_t end = DWT_Cycles();
    enable_irq();
    *result = acc;
    return end - start;
}

#endif /* KERNEL_Q15_SIMD */

/* ---- Reporting ----------------------------------------------------------
 *
 * All integer math (see file header). uint64_t intermediates so the *1000
 * scalings can't overflow for any cycle count this lab produces.
 *
 *   us         = cycles / 168          (1 decimal place)
 *   MACs/cycle = macs / cycles         (3 decimal places)
 *   cycles/MAC = cycles / macs         (2 decimal places) -- the inverse,
 *                printed too because the soft-float result is ~0.01
 *                MACs/cycle, where 3 decimals of MACs/cycle hide a lot
 */
static void report(const char *label, uint32_t cycles, uint32_t macs_per_rep, uint32_t reps) {
    uint64_t total_macs = (uint64_t)macs_per_rep * (uint64_t)reps;
    uint32_t us_x10          = (uint32_t)(((uint64_t)cycles * 10U) / (CLOCK_HZ / 1000000U));
    uint32_t macs_per_cyc_x1000 = (uint32_t)((total_macs * 1000U) / cycles);
    uint32_t cyc_per_mac_x100   = (uint32_t)(((uint64_t)cycles * 100U) / total_macs);

    uart_puts(label);
    uart_puts(": ");
    uart_put_uint_w(cycles, 10);
    uart_puts(" cycles | ");
    uart_put_uint_w(us_x10 / 10U, 7);
    uart_putchar('.');
    uart_put_uint(us_x10 % 10U);
    uart_puts(" us | ");
    uart_put_fixed(macs_per_cyc_x1000, 3);
    uart_puts(" MACs/cycle | ");
    uart_put_fixed(cyc_per_mac_x100, 2);
    uart_puts(" cycles/MAC\r\n");
}

/* ---- Main ---------------------------------------------------------------- */

int main(void) {
    clock_init();
    DWT_Init();

    /* Give the volatile arrays some non-zero content -- values don't
     * matter for the q15 kernels' timing (integer multiply is data-independent on
     * this core). They DO matter a little for the soft-float build, whose
     * emulation routines take data-dependent branches (e.g. normalization
     * after an add) -- ordinary, normal-range values like these exercise
     * the common path. */
#if defined(KERNEL_F32)
    for (uint32_t i = 0; i < N_FLOAT; i++) {
        fa[i] = 1.0001f;
        fb[i] = 0.9999f;
    }
#else
    for (uint32_t i = 0; i < N_Q15; i++) {
        qa[i] = (int16_t)(i % 100);
        qb[i] = (int16_t)((N_Q15 - i) % 100);
    }
#endif

    uart_puts("\r\n--- Cortex-M Performance Lab: STM32F407VG ---\r\n");
    uart_puts("Clock = 168000000 Hz (HSE -> PLL)\r\n\r\n");

    /* Confirm you built what you think you built before trusting the float32
     * number. __ARM_PCS_VFP is only defined for the hard ABI; __ARM_FP is
     * defined whenever FPU instructions are allowed (hard or softfp). */
#if defined(__ARM_PCS_VFP)
    uart_puts("Build config: FLOAT_ABI=hard   -- FPU instructions ENABLED, "
              "float args in FPU registers\r\n");
#elif defined(__ARM_FP) && (__ARM_FP != 0)
    uart_puts("Build config: FLOAT_ABI=softfp -- FPU instructions ENABLED, "
              "float args in core registers\r\n");
#else
    uart_puts("Build config: FLOAT_ABI=soft   -- FPU instructions DISABLED, "
              "float ops emulated in software\r\n");
#endif

#if defined(KERNEL_F32)
    uart_puts("Kernel:       float32 dot product (1024 elements x 500 reps)\r\n\r\n");
    float f_result;
    uint32_t cycles = bench_dot_f32(N_FLOAT, N_FLOAT_REPS, &f_result);
    report("float32 dot product  ", cycles, N_FLOAT, N_FLOAT_REPS);
    /* Sanity check, printed as an integer (x1000) to keep float out of
     * the formatting path: 1024 * 1.0001 * 0.9999 = 1023.99999..., so expect
     * ~1023999 or ~1024000 depending on rounding. The conversion is the one
     * float operation outside the timed region. */
    uart_puts("  (result x1000 = ");
    uart_put_int((int32_t)(f_result * 1000.0f));
    uart_puts(", expected ~1024000)\r\n");
#else
    int32_t r_kernel;
#  if defined(KERNEL_Q15_SCALAR)
    uart_puts("Kernel:       q15 scalar dot product (1024 elements x 500 reps)\r\n\r\n");
    uint32_t cycles = bench_dot_q15_scalar(N_Q15, N_Q15_REPS, &r_kernel);
    report("q15 scalar           ", cycles, N_Q15, N_Q15_REPS);
#  else
    uart_puts("Kernel:       q15 SIMD (SMLAD) dot product (1024 elements x 500 reps)\r\n\r\n");
    uint32_t cycles = bench_dot_q15_simd(N_Q15, N_Q15_REPS, &r_kernel);
    report("q15 SIMD (SMLAD)     ", cycles, N_Q15, N_Q15_REPS);
#  endif
    /* The timed kernel must compute the same dot product as the plain
     * reference loop -- if not (e.g. wrong SIMD packing), its timing is
     * meaningless. */
    int32_t r_ref = q15_reference();
    uart_puts("  (result = ");
    uart_put_int(r_kernel);
    uart_puts(", reference = ");
    uart_put_int(r_ref);
    uart_puts(r_kernel == r_ref ? "  -- match)\r\n" : "  -- MISMATCH!)\r\n");
#endif

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
