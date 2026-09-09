/* Make Mooncake's host segment huge-page backed, so EFA will register it.
 * ===========================================================================
 * WHY THIS EXISTS. On aws-pdx an EFA device refuses ibv_reg_mr past a
 * cumulative 383 GiB (measured; equals its reported max_mr_size of 384 GiB).
 * The store mounts one 190 GiB segment per rank and the generic RDMA transport
 * registers it on EVERY named device, so 8 ranks ask each NIC for 1,552 GiB and
 * seven of them die with
 *
 *     rdma_context.cpp:602] Failed to register memory ...: Cannot allocate memory
 *     real_client.cpp:1078] Failed to mount segment: INVALID_PARAMS
 *
 * The budget is not bytes. The kernel builds it as
 *
 *     max_mr_size = max_mr_pages * PAGE_SIZE            (efa_verbs.c)
 *
 * and registration goes through ib_umem_find_best_pgsz(), which uses the
 * LARGEST page size the mapping actually supports. Memory backed by 2 MiB pages
 * therefore costs 1/512 of the budget. Measured on a pool0 node, same device,
 * same process shape:
 *
 *     4 KiB pages : ibv_reg_mr fails at   383 GiB
 *     2 MiB THP   : 1,700 GiB registered, no failure (that was the probe's cap)
 *
 * so the ceiling moves by at least 4.4x and in principle by 512x. 1,552 GiB
 * fits with room to spare.
 *
 * WHY A SHIM AND NOT A KNOB. There is no knob. /sys/module/efa/parameters is
 * empty -- the driver exposes nothing. Mooncake's MC_STORE_USE_HUGEPAGE takes
 * the hugetlbfs route (MAP_HUGETLB) and the node's explicit pool is ~43 GB, so
 * that mmap fails outright. Its ordinary path is an anonymous mmap with no
 * madvise, and /sys/kernel/mm/transparent_hugepage/enabled reads
 * `always [madvise] never` on these hosts -- madvise is selected, so THP simply
 * never engages for a caller that does not ask.
 *
 * This interposes mmap() and asks, for anonymous mappings at or above
 * THP_SHIM_MIN_BYTES (default 2 GiB):
 *   - MADV_HUGEPAGE       opt in to THP for this range
 *   - MADV_POPULATE_WRITE fault it in now, as huge pages; an unpopulated range
 *                         would still be registered as 4 KiB pages later
 *
 * Only large anonymous maps are touched. File mappings, stacks, and small
 * allocations go through untouched, so this cannot quietly change how the
 * allocator or the loader behaves.
 *
 * Build:  gcc -O2 -fPIC -shared -o thp_mmap_shim.so thp_mmap_shim.c -ldl
 * Use:    LD_PRELOAD=/configs/patches/thp_mmap_shim.so
 * Tune:   THP_SHIM_MIN_BYTES=<n>   THP_SHIM_QUIET=1
 * ===========================================================================
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

#define TWO_MIB (2UL * 1024UL * 1024UL)
#define DEFAULT_MIN (2UL * 1024UL * 1024UL * 1024UL)   /* 2 GiB */

static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static size_t min_bytes;
static int quiet;
static int ready;

static void init_once(void) {
    if (ready) return;
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    const char *m = getenv("THP_SHIM_MIN_BYTES");
    min_bytes = m ? strtoul(m, NULL, 10) : DEFAULT_MIN;
    if (min_bytes < TWO_MIB) min_bytes = TWO_MIB;
    quiet = getenv("THP_SHIM_QUIET") != NULL;
    ready = 1;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    init_once();

    /* Decide BEFORE calling through, because of MAP_POPULATE. */
    int eligible = (flags & MAP_ANONYMOUS) && (prot & PROT_WRITE)
                   && len >= min_bytes && !(flags & MAP_HUGETLB);

    /* THE ORDERING BUG THIS AVOIDS. Mooncake maps its segment with
     * MAP_POPULATE, which faults every page in DURING the mmap call. By the
     * time an madvise(MADV_HUGEPAGE) afterwards could run, the range is already
     * backed by 4 KiB pages, and madvise does NOT retroactively collapse them
     * -- khugepaged might, eventually, but not before ibv_reg_mr runs. The
     * first version of this shim advised after the fact and changed nothing:
     * the run failed with the same rdma_context.cpp:602 ENOMEM as without it.
     *
     * So drop MAP_POPULATE here and do the population ourselves, after the
     * hugepage hint. The caller still gets a fully resident mapping, which is
     * what MAP_POPULATE promised. */
    int passthru_flags = eligible ? (flags & ~MAP_POPULATE) : flags;

    void *p = real_mmap(addr, len, prot, passthru_flags, fd, off);
    if (p == MAP_FAILED || !eligible) return p;

    /* THP only backs a 2 MiB-aligned subrange, so advise the aligned interior
     * rather than the raw bounds -- an unaligned call silently no-ops the head
     * and tail and would leave those on 4 KiB pages. */
    unsigned long start = ((unsigned long)p + TWO_MIB - 1) & ~(TWO_MIB - 1);
    unsigned long end = ((unsigned long)p + len) & ~(TWO_MIB - 1);
    if (end <= start) return p;
    size_t span = end - start;

    int rc_h = madvise((void *)start, span, MADV_HUGEPAGE);
    int rc_p = madvise((void *)start, span, MADV_POPULATE_WRITE);
    if (rc_p != 0) {
        /* Older kernels lack MADV_POPULATE_WRITE. Touch one byte per huge page
         * instead; the write fault is what assembles the THP. */
        for (unsigned long o = 0; o < span; o += TWO_MIB)
            ((volatile char *)start)[o] = ((volatile char *)start)[o];
        rc_p = 0;
    }
    /* Log every mapping we touch, not only under a verbose flag: there are a
     * handful of them per rank, and their absence is the only way to tell "the
     * shim did not apply" from "the shim applied and did not help". */
    if (!quiet)
        fprintf(stderr, "[thp-shim] %.2f GiB at %p: hugepage=%d populate=%d%s\n",
                (double)span / (1024.0 * 1024.0 * 1024.0), (void *)start,
                rc_h, rc_p, (flags & MAP_POPULATE) ? " (MAP_POPULATE deferred)" : "");
    return p;
}
