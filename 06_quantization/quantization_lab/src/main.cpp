/*
 * main.cpp — Quantization lab: what does int8 change on the MCU?
 * Embedded AI course — Quantization (Topic 6)
 * Board: STM32F4DISCOVERY (STM32F407VG)
 *
 * Runs ONE build of Model B2 (whichever python/export_model.py last
 * generated into generated/model_pte.h: A or C, fp32 or int8) and reports:
 *
 *   1. memory: the .pte size embedded in Flash, the planned activation
 *      arena the runtime reads out of the program, and the peak scratch
 *      memory the kernels asked for while running;
 *   2. predictions on the embedded Fashion-MNIST test images
 *      (generated/test_images.h, uint8): accuracy against the labels, and
 *      agreement with the laptop's predictions for the same build. For the
 *      first few disagreements it prints the board's logits so they can be
 *      compared with the host's (python/compare_preds.py does that);
 *   3. latency: method.execute() timed with the DWT cycle counter,
 *      N_WARMUP untimed runs then N_TIMED timed runs;
 *   4. a per-op profile: one more inference run one instruction at a time
 *      with Method::step(), each instruction timed separately.
 *
 * Plumbing (clock, UART, DWT, ExecuTorch bring-up) is the pruning lab's
 * ../../05_pruning/pruning_lab/src/main.cpp. New here: the temp allocator
 * (CMSIS-NN kernels need scratch memory), the input tensor's dtype and dim
 * order come from the program instead of being hard-coded, and the
 * per-instruction profile.
 *
 * Memory placement (see README.md "Memory budget"):
 *   planned_pool  main SRAM   the activation arena; up to 112 KB (A-fp32)
 *   method_pool   CCM RAM     ExecuTorch's per-method bookkeeping
 *   temp_pool     CCM RAM     kernel scratch, reset after every kernel
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

#include "model_pte.h"      /* generated: model_pte[], host_predictions[], instruction_names[] */
#include "test_images.h"    /* generated: test_images[][], test_labels[], pixel_lut[] */

using namespace executorch::runtime;
using namespace executorch::extension;
using namespace executorch::runtime::etensor;

#define CLOCK_HZ  168000000UL
/* Run-to-run spread is a few dozen cycles (bare metal, no interrupts), so
 * a few runs are plenty -- the same choice as the pruning lab. */
#define N_WARMUP  1
#define N_TIMED   10
/* Logits are printed for at most this many host/board disagreements. */
#define N_SHOW_DISAGREE  3

/* ---- Clock / peripheral init (identical to the pruning lab) ------------ */

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

/* ---- UART output (no printf) ------------------------------------------- */

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

/* Fixed-point with 3 decimals, enough to compare logits by eye. */
static void uart_put_float3(float f) {
    if (f < 0.0f) {
        uart_putchar('-');
        f = -f;
    }
    uint32_t milli = (uint32_t)(f * 1000.0f + 0.5f);
    uart_put_uint(milli / 1000U);
    uart_putchar('.');
    uint32_t frac = milli % 1000U;
    uart_putchar((char)('0' + frac / 100U));
    uart_putchar((char)('0' + (frac / 10U) % 10U));
    uart_putchar((char)('0' + frac % 10U));
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

/* The activation arena, sized for the largest build (A-fp32, 114,688 B) so
 * one firmware runs every build. arm-none-eabi-size therefore reports the
 * same SRAM for every build; the "Planned arena" line on the UART is the
 * number that changes. Too big for the 64 KB CCM, so it lives in main SRAM. */
#define PLANNED_POOL_BYTES  (112 * 1024)
static uint8_t planned_pool[PLANNED_POOL_BYTES] __attribute__((aligned(16)));

__attribute__((section(".ccmram"), aligned(16)))
static uint8_t method_pool[16 * 1024];

/* Kernel scratch space. CMSIS-NN's conv kernels unpack each input patch
 * into an int16 buffer (im2col) before the SMLAD loop; ExecuTorch hands
 * them this allocator and resets it after every kernel call. The portable
 * fp32 kernels never use it. */
__attribute__((section(".ccmram"), aligned(16)))
static uint8_t temp_pool[8 * 1024];

/* A MemoryAllocator that remembers the most it ever had allocated, so the
 * firmware can report the scratch memory the kernels really used. */
class PeakAllocator : public MemoryAllocator {
public:
    PeakAllocator(uint32_t size, uint8_t *base) : MemoryAllocator(size, base) {}
    void *allocate(size_t size, size_t alignment = kDefaultAlignment) override {
        uint8_t *p = (uint8_t *)MemoryAllocator::allocate(size, alignment);
        if (p != nullptr && (uint32_t)(p + size - base_address()) > peak) {
            peak = (uint32_t)(p + size - base_address());
        }
        return p;
    }
    uint32_t peak = 0;
};

/* Model input staging buffer: set_input() copies from here into the
 * input's slot in planned_pool. */
__attribute__((section(".ccmram"), aligned(16)))
static float input_data[32 * 32];

static uint8_t board_preds[N_EVAL_RUN];
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

/* execute() failed: run the method again one instruction at a time to find
 * out which one fails, report it, and stop. */
static void fail_execute(Method &method, Error err) {
    uart_puts("ERROR: execute failed, error ");
    uart_put_hex((uint32_t)err);
    uart_puts("\r\n");
    for (int k = 0; k < N_INSTRUCTIONS; k++) {
        Error e = method.step();
        if (e != Error::Ok) {
            uart_puts("  at instruction ");
            uart_put_uint((uint32_t)k);
            uart_puts(": ");
            uart_puts(instruction_names[k]);
            uart_puts(", error ");
            uart_put_hex((uint32_t)e);
            uart_puts("\r\n");
            break;
        }
    }
    while (1);
}

static int argmax10(const float *p) {
    int best = 0;
    for (int i = 1; i < 10; i++)
        if (p[i] > p[best]) best = i;
    return best;
}

static void load_image(int i) {
    for (int k = 0; k < 32 * 32; k++) input_data[k] = pixel_lut[test_images[i][k]];
}

int main(void) {
    clock_init();
    DWT_Init();
    runtime_init();

    uart_puts("\r\n--- Quantization lab: STM32F407VG ---\r\n");
    uart_puts("Build ");
    uart_puts(MODEL_BUILD);
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

    /* The input is a 1x1x32x32 float tensor in every build. int8 builds
     * quantize it inside the graph (their first op), and were exported with
     * a channels-last example input, so their dim order is NHWC. With one
     * channel, NCHW and NHWC are the same bytes; only the recorded dim
     * order differs, and set_input() checks it. */
    Result<TensorInfo> in_meta = meta->input_tensor_meta(0);
    if (!in_meta.ok()) fail("input_tensor_meta", (uint32_t)in_meta.error());
    if (in_meta->scalar_type() != ScalarType::Float || in_meta->sizes().size() != 4)
        fail("input is not a 4-D float tensor", (uint32_t)in_meta->scalar_type());
    static TensorImpl::SizesType    sizes[4]     = {1, 1, 32, 32};
    static TensorImpl::DimOrderType dim_order[4];
    static TensorImpl::StridesType  strides[4];
    for (int d = 0; d < 4; d++) dim_order[d] = in_meta->dim_order()[d];
    TensorImpl::StridesType stride = 1;
    for (int k = 3; k >= 0; k--) {
        strides[dim_order[k]] = stride;
        stride *= sizes[dim_order[k]];
    }
    uart_puts("Input                  : float32 1x1x32x32, ");
    uart_puts(dim_order[1] == 1 ? "NCHW (contiguous)\r\n" : "NHWC (channels-last)\r\n");

    MemoryAllocator method_allocator(sizeof(method_pool), method_pool);
    PeakAllocator temp_allocator(sizeof(temp_pool), temp_pool);
    Span<uint8_t> planned_span(planned_pool, planned_bytes);
    HierarchicalAllocator planned_allocator({&planned_span, 1});
    MemoryManager memory_manager(&method_allocator, &planned_allocator, &temp_allocator);

    Result<Method> method_result = program.load_method("forward", &memory_manager);
    if (!method_result.ok()) fail("load_method", (uint32_t)method_result.error());
    Method method = std::move(method_result.get());
    /* MemoryAllocator has no "bytes used" accessor; a 1-byte, unaligned
     * allocation returns the current bump pointer, which is the same thing. */
    uint8_t *method_pool_end = (uint8_t *)method_allocator.allocate(1, 1);
    report("method_pool used (CCM) : ", (uint32_t)(method_pool_end - method_pool), " bytes");

    TensorImpl input_impl(ScalarType::Float, 4, sizes, input_data, dim_order, strides);
    Tensor input_tensor(&input_impl);

    /* ---- Predictions ---- */
    uart_puts("\r\n=== Predictions: ");
    uart_put_uint(N_EVAL_RUN);
    uart_puts(" test images ===\r\n");
    int match_host = 0;
    int match_label = 0;
    for (int i = 0; i < N_EVAL_RUN; i++) {
        load_image(i);
        Error err = method.set_input(EValue(input_tensor), 0);
        if (err != Error::Ok) fail("set_input", (uint32_t)err);
        err = method.execute();
        if (err != Error::Ok) fail_execute(method, err);
        board_preds[i] = (uint8_t)argmax10(method.get_output(0).toTensor().const_data_ptr<float>());
        match_host += (board_preds[i] == host_predictions[i]);
        match_label += (board_preds[i] == test_labels[i]);
    }
    /* Machine-readable for python/compare_preds.py: 50 digits per line. */
    for (int i = 0; i < N_EVAL_RUN; i += 50) {
        uart_puts("preds ");
        uart_put_uint_w((uint32_t)i, 3);
        uart_puts(": ");
        for (int k = i; k < i + 50 && k < N_EVAL_RUN; k++) uart_putchar((char)('0' + board_preds[k]));
        uart_puts("\r\n");
    }
    uart_puts("Board accuracy         : ");
    uart_put_uint((uint32_t)match_label);
    uart_putchar('/');
    uart_put_uint(N_EVAL_RUN);
    uart_puts(" correct\r\n");
    uart_puts("Board matches host     : ");
    uart_put_uint((uint32_t)match_host);
    uart_putchar('/');
    uart_put_uint(N_EVAL_RUN);
    uart_puts("\r\n");

    /* Logits for the first few disagreements, to compare with the host's. */
    int shown = 0;
    for (int i = 0; i < N_EVAL_RUN && shown < N_SHOW_DISAGREE; i++) {
        if (board_preds[i] == host_predictions[i]) continue;
        load_image(i);
        if (method.set_input(EValue(input_tensor), 0) != Error::Ok) fail("set_input", 0);
        if (method.execute() != Error::Ok) fail("execute", 0);
        const float *logits = method.get_output(0).toTensor().const_data_ptr<float>();
        uart_puts("  image ");
        uart_put_uint((uint32_t)i);
        uart_puts(": board ");
        uart_puts(class_names[board_preds[i]]);
        uart_puts(", host ");
        uart_puts(class_names[host_predictions[i]]);
        uart_puts("\r\n    logits:");
        for (int c = 0; c < 10; c++) {
            uart_putchar(' ');
            uart_put_float3(logits[c]);
        }
        uart_puts("\r\n");
        shown++;
    }

    /* ---- Latency (input stays at image 0) ---- */
    load_image(0);
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
    /* cycles -> hundredths of a millisecond */
    uint32_t median_ms_x100 = median / (CLOCK_HZ / 100000U);

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
    uart_put_uint(median_ms_x100 / 100U);
    uart_putchar('.');
    uart_putchar((char)('0' + (median_ms_x100 / 10U) % 10U));
    uart_putchar((char)('0' + median_ms_x100 % 10U));
    uart_puts(" ms\r\n");

    /* ---- Per-op profile: one inference, one instruction at a time ----
     * Each step() is one kernel call (or a move/free instruction). The sum
     * is a little above the execute() median: step() adds a few hundred
     * cycles of bookkeeping per instruction. */
    uart_puts("\r\n=== Per-op profile: Method::step() ===\r\n");
    uint32_t total = 0;
    for (int k = 0;; k++) {
        uint32_t t0 = DWT_Cycles();
        Error err = method.step();
        uint32_t dt = DWT_Cycles() - t0;
        if (err == Error::EndOfMethod) break;
        if (err != Error::Ok) fail("step", (uint32_t)err);
        total += dt;
        uart_puts("  ");
        uart_put_uint_w((uint32_t)k, 2);
        uart_puts("  ");
        uart_put_uint_w(dt, 10);
        uart_puts("  ");
        uart_puts(k < N_INSTRUCTIONS ? instruction_names[k] : "?");
        uart_puts("\r\n");
    }
    if (method.reset_execution() != Error::Ok) fail("reset_execution", 0);
    uart_puts("  total ");
    uart_put_uint_w(total, 10);
    uart_puts("\r\n");
    report("Kernel scratch (peak)  : ", temp_allocator.peak, " bytes (temp_pool, CCM)");

    uart_puts("\r\n--- Done ---\r\n");
    while (1);
}
