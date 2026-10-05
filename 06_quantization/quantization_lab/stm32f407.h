/* stm32f407.h — Peripheral register map for the dma_bandwidth_lab project.
 *
 * RCC/FLASH/GPIO/USART2/DWT/CoreDebug fields copied from
 * ../../01_memtypes/memory_benchmark/stm32f407.h (same HSE->PLL->168MHz
 * clock config, same USART2 setup, same DWT cycle counter). DMA2 is new
 * here -- no other bare-metal (non-HAL) project in this repo drives a DMA
 * controller by hand yet.
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
#define DMA2_BASE       0x40026400UL   /* DMA1 is 0x40026000; DMA2 is the
                                           only one of the two that can do
                                           memory-to-memory on the F4. */

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
#define RCC_AHB1ENR_DMA2EN  (1U << 22)

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

/* ── DMA (memory-to-memory) ───────────────────────────────────────────────
 *
 * Only DMA2 is used (DMA1's peripheral port isn't wired to the memories,
 * so it cannot do memory-to-memory transfers on the F4 -- RM0090 §9.3.3).
 * Each stream is 6 words (24 bytes): SxCR, SxNDTR, SxPAR, SxM0AR, SxM1AR,
 * SxFCR, immediately following the 4 status/clear registers common to all
 * 8 streams -- so STREAM[n] indexes correctly with no padding needed.
 * This lab only drives STREAM[0] (DMA2_Stream0).
 */
typedef struct {
    volatile uint32_t CR;      /* 0x00 configuration                      */
    volatile uint32_t NDTR;    /* 0x04 number of data items to transfer   */
    volatile uint32_t PAR;     /* 0x08 peripheral address (= src for M2M) */
    volatile uint32_t M0AR;    /* 0x0C memory 0 address (= dst for M2M)   */
    volatile uint32_t M1AR;    /* 0x10 memory 1 address (double-buffer,   *
                                 *      unused here)                       */
    volatile uint32_t FCR;     /* 0x14 FIFO control                       */
} DMA_Stream_t;

typedef struct {
    volatile uint32_t LISR;    /* 0x00 low interrupt status  (streams 0-3) */
    volatile uint32_t HISR;    /* 0x04 high interrupt status (streams 4-7) */
    volatile uint32_t LIFCR;   /* 0x08 low interrupt flag clear            */
    volatile uint32_t HIFCR;   /* 0x0C high interrupt flag clear           */
    DMA_Stream_t      STREAM[8];
} DMA_t;

#define DMA2    ((DMA_t *)DMA2_BASE)

/* SxCR bits */
#define DMA_SxCR_EN             (1U << 0)
#define DMA_SxCR_CIRC           (1U << 8)   /* not permitted for M2M       */
#define DMA_SxCR_PINC           (1U << 9)
#define DMA_SxCR_MINC           (1U << 10)
#define DMA_SxCR_DIR_M2M        (2U << 6)   /* 10 = memory-to-memory       */
#define DMA_SxCR_PSIZE_WORD     (2U << 11)
#define DMA_SxCR_MSIZE_WORD     (2U << 13)
#define DMA_SxCR_PL_HIGH        (2U << 16)  /* priority level: high        */

/* SxFCR bits -- M2M mode requires the FIFO (direct mode is not permitted
 * when DIR = memory-to-memory), so DMDIS must be set. */
#define DMA_SxFCR_FTH_FULL      (3U << 0)   /* FIFO threshold = full       */
#define DMA_SxFCR_DMDIS         (1U << 2)   /* direct mode disable         */

/* LIFCR/LISR flag bits for stream 0 (the only stream this lab uses) */
#define DMA_LIFCR_CFEIF0        (1U << 0)
#define DMA_LIFCR_CDMEIF0       (1U << 2)
#define DMA_LIFCR_CTEIF0        (1U << 3)
#define DMA_LIFCR_CHTIF0        (1U << 4)
#define DMA_LIFCR_CTCIF0        (1U << 5)

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
