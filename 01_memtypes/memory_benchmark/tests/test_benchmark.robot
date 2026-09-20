*** Settings ***
Documentation     Boots memory_benchmark under Renode and captures each of the
...               five report lines it prints over USART2. Run with:
...                   renode-test tests/test_benchmark.robot
...               The captured lines are echoed to the console (Log To
...               Console) and also land in log.html for later reference.

*** Variables ***
${UART}     sysbus.usart2
${ELF}      @${CURDIR}/../build/firmware.elf
${REPL}     @${CURDIR}/../platforms/stm32f4_discovery_full.repl

*** Test Cases ***
Capture Benchmark Output
    Execute Command          mach create "bench"
    Execute Command          machine LoadPlatformDescription ${REPL}
    Execute Command          sysbus LoadELF ${ELF}
    Create Terminal Tester   ${UART}    timeout=60
    Start Emulation

    Wait For Line On Uart    --- Memory Benchmark: STM32F407VG ---
    ${clock}=       Wait For Line On Uart    Clock =
    ${bufsize}=      Wait For Line On Uart    Buffer size =
    ${sram_w}=       Wait For Line On Uart    SRAM write
    ${sram_r}=       Wait For Line On Uart    SRAM read
    ${ccm_w}=        Wait For Line On Uart    CCM RAM write
    ${ccm_r}=        Wait For Line On Uart    CCM RAM read
    ${flash_r}=      Wait For Line On Uart    Flash read
    Wait For Line On Uart    --- Done ---

    Log To Console    ${\n}${clock}[Line]
    Log To Console    ${bufsize}[Line]
    Log To Console    ${\n}${sram_w}[Line]
    Log To Console    ${sram_r}[Line]
    Log To Console    ${ccm_w}[Line]
    Log To Console    ${ccm_r}[Line]
    Log To Console    ${flash_r}[Line]
