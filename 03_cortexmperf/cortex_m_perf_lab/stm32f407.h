/* stm32f407.h — Peripheral register map for the cortex_m_perf_lab project.
 *
 * Copied from ../../04_memmovement/dma_bandwidth_lab/stm32f407.h with its
 * DMA2 section removed (this lab doesn't use DMA). RCC/FLASH/GPIO/USART2/
 * DWT/CoreDebug are the same as there and in
 * ../../01_memtypes/memory_benchmark/stm32f407.h: same HSE->PLL->168MHz
 * clock config, same USART2 setup, same DWT cycle counter.
 *
 * All addresses from STM32F407 reference manual RM0090 and the Cortex-M4
 * Technical Reference Manual (DWT/CoreDebug are core peripherals, not
 * ST-specific).
 */

#ifndef STM32F407_H
#define STM32F407_H

#include <stdint.h>

/* ── Base addresses ──────────────────────────────────────────────────────── */
#define PERIPH_BASE     0x40000000UL
#define APB1_BASE       (PERIPH_BASE + 0x00000000UL)
#define AHB1_BASE       (PERIPH_BASE + 0x00020000UL)

#define GPIOA_BASE      (AHB1_BASE + 0x0000UL)
#define GPIOD_BASE      (AHB1_BASE + 0x0C00UL)
#define RCC_BASE        (AHB1_BASE + 0x3800UL)

#define USART2_BASE     (APB1_BASE + 0x4400UL)
#define FLASH_BASE      0x40023C00UL

#define DWT_BASE        0xE0001000UL
#define CoreDebug_BASE  0xE000EDF0UL

/* ── GPIO ────────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
    volatile uint32_t PUPDR;
    volatile uint32_t IDR;
    volatile uint32_t ODR;
    volatile uint32_t BSRR;
    volatile uint32_t LCKR;
    volatile uint32_t AFR[2];
} GPIO_t;

#define GPIOA   ((GPIO_t *)GPIOA_BASE)
#define GPIOD   ((GPIO_t *)GPIOD_BASE)

#define GPIO_MODER_OUTPUT   0x1U
#define GPIO_MODER_AF       0x2U

/* ── RCC ─────────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t CR;           /* 0x00 */
    volatile uint32_t PLLCFGR;      /* 0x04 */
    volatile uint32_t CFGR;         /* 0x08 */
    volatile uint32_t CIR;          /* 0x0C */
    volatile uint32_t AHB1RSTR;     /* 0x10 */
    volatile uint32_t AHB2RSTR;     /* 0x14 */
    volatile uint32_t AHB3RSTR;     /* 0x18 */
    uint32_t          RESERVED0;    /* 0x1C */
    volatile uint32_t APB1RSTR;     /* 0x20 */
    volatile uint32_t APB2RSTR;     /* 0x24 */
    uint32_t          RESERVED1[2]; /* 0x28-0x2C */
    volatile uint32_t AHB1ENR;      /* 0x30 */
} RCC_t;

#define RCC     ((RCC_t *)RCC_BASE)

/* CR bits */
#define RCC_CR_HSEON        (1U << 16)
#define RCC_CR_HSERDY       (1U << 17)
#define RCC_CR_PLLON        (1U << 24)
#define RCC_CR_PLLRDY       (1U << 25)

/* PLLCFGR bits */
#define RCC_PLLCFGR_PLLSRC_HSE  (1U << 22)

/* CFGR bits */
#define RCC_CFGR_SW_PLL     (0x2U << 0)
#define RCC_CFGR_SWS_PLL    (0x2U << 2)
#define RCC_CFGR_SWS_MASK   (0x3U << 2)
#define RCC_CFGR_PPRE1_DIV4 (0x5U << 10)  /* APB1 = AHB/4 */
#define RCC_CFGR_PPRE2_DIV2 (0x4U << 13)  /* APB2 = AHB/2 */

/* AHB1ENR bits */
#define RCC_AHB1ENR_GPIOAEN (1U << 0)
#define RCC_AHB1ENR_GPIODEN (1U << 3)

/* APB1ENR is at offset 0x40, past the fields declared above -- accessed
 * via a raw pointer since we don't need the registers in between. */
#define RCC_APB1ENR   (*(volatile uint32_t *)(RCC_BASE + 0x40UL))
#define RCC_APB1ENR_USART2EN  (1U << 17)

/* ── Flash interface (for wait-state config) ─────────────────────────────── */
typedef struct {
    volatile uint32_t ACR;
} FLASH_t;

#define FLASH   ((FLASH_t *)FLASH_BASE)
#define FLASH_ACR_LATENCY(n)    ((n) & 0x7U)
#define FLASH_ACR_PRFTEN        (1U << 8)
#define FLASH_ACR_ICEN          (1U << 9)
#define FLASH_ACR_DCEN          (1U << 10)

/* ── USART ───────────────────────────────────────────────────────────────── */
typedef struct {
    volatile uint32_t SR;
    volatile uint32_t DR;
    volatile uint32_t BRR;
    volatile uint32_t CR1;
    volatile uint32_t CR2;
    volatile uint32_t CR3;
    volatile uint32_t GTPR;
} USART_t;

#define USART2  ((USART_t *)USART2_BASE)

#define USART_SR_TXE    (1U << 7)
#define USART_SR_TC     (1U << 6)
#define USART_CR1_UE    (1U << 13)
#define USART_CR1_TE    (1U << 3)

/* ── DWT (Data Watchpoint & Trace) cycle counter ─────────────────────────── */
/* Core peripheral, not ST-specific -- same on every Cortex-M4. Used here
 * as the micro-benchmark's clock: it counts CPU cycles directly with no
 * peripheral setup overhead beyond the two-register init below. */
typedef struct {
    volatile uint32_t CTRL;    /* 0x00 */
    volatile uint32_t CYCCNT;  /* 0x04 */
} DWT_t;

#define DWT     ((DWT_t *)DWT_BASE)
#define DWT_CTRL_CYCCNTENA  (1U << 0)

/* CoreDebug->DEMCR must have TRCENA set before DWT->CTRL has any effect. */
typedef struct {
    uint32_t          RESERVED0[3]; /* DHCSR, DCRSR, DCRDR -- unused here */
    volatile uint32_t DEMCR;        /* 0xE000EDFC */
} CoreDebug_t;

#define CoreDebug   ((CoreDebug_t *)CoreDebug_BASE)
#define CoreDebug_DEMCR_TRCENA  (1U << 24)

#endif /* STM32F407_H */
