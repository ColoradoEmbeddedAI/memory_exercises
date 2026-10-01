/*
 * main.cpp — Pruning lab: what does pruning change on the MCU?
 * Embedded AI course — Pruning (Topic 5)
 * Board: STM32F4DISCOVERY (STM32F407VG)
 *
 * Runs ONE variant of Model B2 (whichever python/export_variants.py
 * --variant X last generated into generated/model_pte.h) and reports:
 *
 *   1. the .pte size embedded in Flash, and the planned activation arena
 *      size the runtime reads out of the program at load time;
 *   2. an export sanity check: predictions on 10 embedded Fashion-MNIST
 *      test images (generated/test_images.h), compared with the laptop's
 *      PyTorch predictions for the same checkpoint. This checks that the
 *      export is correct. It is NOT an accuracy measurement (10 images);
 *      accuracy comes from python/measure_host.py;
 *   3. latency: method.execute() timed with the DWT cycle counter,
 *      N_WARMUP untimed runs then N_TIMED timed runs, reported as
 *      min / median / max cycles and median milliseconds.
 *
 * Plumbing: clock_init()/UART/DWT are copied from
 * ../../04_memmovement/dma_bandwidth_lab/src/main.c (HSE -> PLL ->
 * 168 MHz, USART2 115200 8N1 on PA2, no printf). The ExecuTorch bring-up
 * follows ../../../lab0_hello_world_soln/src/main.cpp.
 *
 * Memory placement (see README.md "Memory budget"):
 *   planned_pool  main SRAM   the activation arena; up to 112 KB for B2
 *   method_pool   CCM RAM     ExecuTorch's per-method bookkeeping
 *   model_pte     Flash       read in place, never copied to RAM
 */

#include <cstdint>
#include <cstring>

#include "stm32f407.h"

#include "executorch/runtime/executor/program.h"
#include "executorch/runtime/executor/method.h"
#include "executorch/runtime/core/exec_aten/exec_aten.h"
#include "executorch/runtime/core/portable_type/tensor_impl.h"
#include "executorch/runtime/core/hierarchical_allocator.h"
#include "executorch/runtime/platform/runtime.h"
#include "executorch/extension/data_loader/buffer_data_loader.h"

#include "model_pte.h"      /* generated: model_pte[], host_predictions[] */
#include "test_images.h"    /* generated: test_images[][], test_labels[] */

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

#define CLOCK_HZ  168000000UL
/* Portable-kernel inference takes ~1.4 s for the full-width variants, and
 * run-to-run spread is a handful of cycles out of ~235 million (bare
 * metal, no interrupts, no cache misses to vary), so a few runs are
 * plenty. More runs only make students wait. */
#define N_WARMUP  1
#define N_TIMED   10

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

/* ---- UART output (no printf -- see dma_bandwidth_lab's file header) ---- */

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

/* Right-aligns v in a field of `width` characters. */
static void uart_put_uint_w(uint32_t v, int width) {
    int digits = 1;
    for (uint32_t t = v; t >= 10U; t /= 10U) digits++;
    while (width-- > digits) uart_putchar(' ');
    uart_put_uint(v);
}

static void uart_put_hex(uint32_t v) {
    uart_puts("0x");
    for (int i = 28; i >= 0; i -= 4) uart_putchar("0123456789abcdef"[(v >> i) & 0xFU]);
}

/* Prints a label, a value, and a newline. */
static void report(const char *label, uint32_t v, const char *unit) {
    uart_puts(label);
    uart_put_uint(v);
    uart_puts(unit);
    uart_puts("\r\n");
}

/* Stops with the error on the UART. ExecuTorch error codes are listed in
 * executorch/runtime/core/error.h: 0x14 = OperatorMissing,
 * 0x21 = MemoryAllocationFailed, 0x12 = InvalidArgument. */
static void fail(const char *what, uint32_t err) {
    uart_puts("ERROR: ");
    uart_puts(what);
    uart_puts(" failed, error ");
    uart_put_hex(err);
    uart_puts("\r\n");
    while (1);
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

/* ---- ExecuTorch memory -------------------------------------------------- */

/* The activation arena. Its size must be >= the planned size the runtime
 * reports (printed at boot, and by python/measure_host.py): 114,688 bytes
 * for the full-width variants A and B, 57,344 for C. It's sized for the worst
 * case so one build fits every variant -- which also means SRAM *usage*
 * as reported by arm-none-eabi-size doesn't change between variants; the
 * "planned arena" line on the UART is the number that does. Too big for
 * the 64 KB CCM, so it lives in main SRAM. */
#define PLANNED_POOL_BYTES  (112 * 1024)
static uint8_t planned_pool[PLANNED_POOL_BYTES] __attribute__((aligned(16)));

__attribute__((section(".ccmram"), aligned(16)))
static uint8_t method_pool[16 * 1024];

/* Model input staging buffer: set_input() copies from here into the
 * input's slot in planned_pool. */
__attribute__((section(".ccmram"), aligned(16)))
static float input_data[32 * 32];
static TensorImpl::SizesType    sizes[4]     = {1, 1, 32, 32};
static TensorImpl::DimOrderType dim_order[4] = {0, 1, 2, 3};
static TensorImpl::StridesType  strides[4]   = {32 * 32, 32 * 32, 32, 1};

static uint32_t timings[N_TIMED];

static void sort_u32(uint32_t *a, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
}

static int argmax10(const float *p) {
    int best = 0;
    for (int i = 1; i < 10; i++)
        if (p[i] > p[best]) best = i;
    return best;
}

int main(void) {
    clock_init();
    DWT_Init();
    runtime_init();

    uart_puts("\r\n--- Pruning lab: STM32F407VG ---\r\n");
    uart_puts("Variant ");
    uart_puts(MODEL_VARIANT);
    uart_puts(": ");
    uart_puts(MODEL_DESCRIPTION);
    uart_puts("\r\n");
    report("Clock                  : ", CLOCK_HZ, " Hz (HSE -> PLL)");

    /* ---- Load ---- */
    BufferDataLoader loader(model_pte, model_pte_len);
    Result<Program> program_result = Program::load(&loader, Program::Verification::Minimal);
    if (!program_result.ok()) fail("Program::load", (uint32_t)program_result.error());
    Program program = std::move(program_result.get());

    Result<MethodMeta> meta = program.method_meta("forward");
    if (!meta.ok()) fail("method_meta", (uint32_t)meta.error());
    uint32_t planned_bytes = 0;
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); i++)
        planned_bytes += (uint32_t)meta->memory_planned_buffer_size(i).get();

    report(".pte size (Flash)      : ", model_pte_len, " bytes");
    report("Planned arena (SRAM)   : ", planned_bytes, " bytes");
    if (meta->num_memory_planned_buffers() != 1 || planned_bytes > sizeof(planned_pool)) {
        report("planned_pool is only   : ", sizeof(planned_pool), " bytes");
        fail("planned arena fit", planned_bytes);
    }

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    Span<uint8_t> planned_span(planned_pool, planned_bytes);
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) fail("load_method", (uint32_t)method_result.error());
    Method method = std::move(method_result.get());
    /* MemoryAllocator has no "bytes used" accessor; a 1-byte, unaligned
     * allocation returns the current bump pointer, which is the same thing. */
    uint8_t *method_pool_end = (uint8_t *)method_allocator.allocate(1, 1);
    report("method_pool used (CCM) : ", (uint32_t)(method_pool_end - method_pool), " bytes");

    TensorImpl input_impl(ScalarType::Float, 4, sizes, input_data, dim_order, strides);
    Tensor input_tensor(&input_impl);

    /* ---- Export sanity check ---- */
    uart_puts("\r\n=== Sanity check: 10 test images ===\r\n");
    int match_host = 0;
    int match_label = 0;
    for (int i = 0; i < N_TEST_IMAGES; i++) {
        memcpy(input_data, test_images[i], sizeof(input_data));
        Error err = method.set_input(EValue(input_tensor), 0);
        if (err != Error::Ok) fail("set_input", (uint32_t)err);
        err = method.execute();
        if (err != Error::Ok) fail("execute", (uint32_t)err);
        int pred = argmax10(method.get_output(0).toTensor().const_data_ptr<float>());

        match_host += (pred == host_predictions[i]);
        match_label += (pred == test_labels[i]);
        uart_puts("  image ");
        uart_put_uint((uint32_t)i);
        uart_puts(": label ");
        uart_puts(class_names[test_labels[i]]);
        uart_puts(", board ");
        uart_puts(class_names[pred]);
        uart_puts(pred == host_predictions[i] ? " (= host)\r\n" : " (HOST DIFFERS)\r\n");
    }
    report("Board matches host     : ", (uint32_t)match_host, "/10  (export check)");
    report("Board matches label    : ", (uint32_t)match_label, "/10  (NOT an accuracy measurement)");

    /* ---- Latency ----
     * Input stays at image 0 for every run. Timing depends on data only
     * through a few compare branches (ReLU, max-pool): a few hundred to a
     * couple of thousand cycles out of ~235 million. No kernel skips a
     * multiply because an operand is zero. */
    memcpy(input_data, test_images[0], sizeof(input_data));
    if (method.set_input(EValue(input_tensor), 0) != Error::Ok) fail("set_input", 0);

    for (int i = 0; i < N_WARMUP; i++) (void)method.execute();
    for (int i = 0; i < N_TIMED; i++) {
        uint32_t t0 = DWT_Cycles();
        Error err = method.execute();
        timings[i] = DWT_Cycles() - t0;
        if (err != Error::Ok) fail("execute", (uint32_t)err);
    }
    sort_u32(timings, N_TIMED);
    uint32_t median = timings[N_TIMED / 2];
    /* cycles -> tenths of a millisecond */
    uint32_t median_ms_x10 = median / (CLOCK_HZ / 10000U);

    uart_puts("\r\n=== Latency: method.execute() ===\r\n");
    report("Runs                   : ", N_WARMUP, " warm-up");
    report("                         ", N_TIMED, " timed");
    uart_puts("Cycles min/median/max  : ");
    uart_put_uint_w(timings[0], 10);
    uart_puts(" /");
    uart_put_uint_w(median, 10);
    uart_puts(" /");
    uart_put_uint_w(timings[N_TIMED - 1], 10);
    uart_puts("\r\n");
    uart_puts("Median latency         : ");
    uart_put_uint(median_ms_x10 / 10U);
    uart_putchar('.');
    uart_put_uint(median_ms_x10 % 10U);
    uart_puts(" ms\r\n");

    uart_puts("\r\n--- Done ---\r\n");
    while (1);
}
