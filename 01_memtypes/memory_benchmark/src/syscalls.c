/* syscalls.c — Minimal newlib platform stubs for bare-metal STM32F407
 *
 * No ExecuTorch here, so no snprintf/vsnprintf overrides or PAL logging
 * stub are needed (compare lab0_hello_world_soln/src/syscalls.c, which
 * does need them). This project formats its own output by hand in
 * main.c instead of using printf -- see the comment there for why.
 */

#include <sys/stat.h>
#include <sys/time.h>
#include <stdint.h>
#include <stddef.h>
#include <reent.h>

/* ── Heap allocator ──────────────────────────────────────────────────────── */
extern uint32_t _ebss;
static uint8_t *heap_end = NULL;

void *_sbrk(int incr) {
    if (heap_end == NULL)
        heap_end = (uint8_t *)&_ebss;
    uint8_t *prev = heap_end;
    heap_end += incr;
    return (void *)prev;
}

/* ── Newlib reentrant structure init ─────────────────────────────────────── */
/* Called from Reset_Handler before __libc_init_array. */
void init_newlib(void) {
    _REENT_INIT_PTR(_impure_ptr);
}

/* ── POSIX syscall stubs ─────────────────────────────────────────────────── */
int _write(int fd, char *buf, int len)      { (void)fd; (void)buf; return len; }
int _read(int fd, char *buf, int len)       { (void)fd; (void)buf; (void)len; return 0; }
int _close(int fd)                          { (void)fd; return -1; }
int _isatty(int fd)                         { (void)fd; return 1; }
int _lseek(int fd, int off, int w)          { (void)fd; (void)off; (void)w; return 0; }
void _exit(int s)                           { (void)s; while (1); }
int _kill(int p, int s)                     { (void)p; (void)s; return -1; }
int _getpid(void)                           { return 1; }

int _fstat(int fd, struct stat *st) {
    (void)fd;
    st->st_mode = S_IFCHR;
    return 0;
}

int _gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (tv) { tv->tv_sec = 0; tv->tv_usec = 0; }
    return 0;
}
