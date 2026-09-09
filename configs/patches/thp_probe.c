/* Prove GLIBC_TUNABLES=glibc.malloc.hugetlb=1 is in effect, in THIS container.
 * ===========================================================================
 * Mooncake's store segment is NOT shared memory. The path is
 *
 *   MooncakeDistributedStore.setup() -> real_client.cpp
 *     -> client_buffer_allocation.cpp: allocate_buffer_allocator_memory()
 *       -> aligned_alloc(alignment, total_size)
 *
 * i.e. ordinary anonymous private memory from glibc malloc. (SharedSegment /
 * shm_open in engine.so is a separate opt-in API that the vLLM
 * MooncakeStoreConnector never calls -- a symbol being present is not the same
 * as it being used, which is a mistake this file exists to stop repeating.)
 *
 * glibc malloc reaches the kernel through its INTERNAL __mmap, not the PLT
 * symbol, so no LD_PRELOAD interposer can see the segment mapping. glibc's own
 * tunable is the switch: it madvises MADV_HUGEPAGE on every mmapped chunk.
 * That matters because EFA counts its registration budget in 4 KiB pages --
 * 190 GB x 8 ranks does not fit, the same memory on 2 MiB pages does.
 *
 * Read AnonHugePages for THIS range from /proc/self/smaps, not the global
 * counter in /proc/meminfo: the global number moves for unrelated reasons and
 * would make a negative result unfalsifiable. (Checking the global counter is
 * exactly how this tunable was first, wrongly, written off.)
 *
 * Build: gcc -O2 -o thp_probe thp_probe.c
 * Exit:  0 if the range is huge-page backed, 1 if not.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TWO_MIB (2UL << 20)

static long smaps_thp_kb(uintptr_t lo, uintptr_t hi) {
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    char line[512];
    int in = 0;
    long total = 0, v;
    uintptr_t a, b;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%lx-%lx", &a, &b) == 2 && strchr(line, ' '))
            in = (a < hi && b > lo);
        if (in && sscanf(line, "AnonHugePages: %ld kB", &v) == 1) total += v;
    }
    fclose(f);
    return total;
}

int main(int argc, char **argv) {
    size_t gib = (argc > 1) ? strtoul(argv[1], NULL, 10) : 2;
    size_t n = gib << 30;
    void *p = aligned_alloc(TWO_MIB, n);
    if (!p) { fprintf(stderr, "thp_probe: aligned_alloc(%zu GiB) failed\n", gib); return 2; }
    memset(p, 1, n);
    long thp = smaps_thp_kb((uintptr_t)p, (uintptr_t)p + n);
    double frac = (double)thp * 1024.0 / (double)n;
    printf("    thp_probe: aligned_alloc %zu GiB -> AnonHugePages %ld kB (%.0f%% of the range)\n",
           gib, thp, frac * 100.0);
    if (frac < 0.9) {
        fprintf(stderr, "thp_probe: FAILED -- the range is not huge-page backed.\n"
                        "  GLIBC_TUNABLES=%s\n"
                        "  Expected glibc.malloc.hugetlb=1 to be set in the worker environment.\n",
                getenv("GLIBC_TUNABLES") ? getenv("GLIBC_TUNABLES") : "(unset)");
        return 1;
    }
    return 0;
}
