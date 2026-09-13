/* Count and locate every cudaLaunchHostFunc the process makes.
 *
 * WHY. The Kimi-K3 stall leaves one rank's GPU at 0% utilisation with the step
 * already submitted, no kernel resident, no Xid -- and the store threads idle by
 * the time it is visible. cudaLaunchHostFunc is the only CUDA API the Mooncake
 * transfer engine imports that can park a stream in exactly that way: the stream
 * does not advance until the host callback returns, and nothing is resident
 * while it waits.
 *
 * But "engine.so imports it" is not "our path calls it". Disassembly could not
 * attribute the call site -- the library is stripped, and the nearest-preceding-
 * symbol heuristic returned a std::vector method, which is nonsense. The
 * symbols that DID attribute (cudaHostRegister) land in IbgdaDeviceTransportImpl
 * and mlx5gda_create_rc_qp, i.e. the Mellanox GPUDirect-Async path, which is not
 * what we run on EFA. So the honest state is: unknown.
 *
 * This settles it by counting. Zero calls kills the hypothesis for the price of
 * one run; a non-zero count gives the return addresses to resolve.
 *
 * LD_PRELOAD works here where it did not for the segment allocation: engine.so
 * reaches cudaLaunchHostFunc through the PLT into libcudart, which is
 * interposable, unlike glibc's internal __mmap.
 *
 *   gcc -O2 -fPIC -shared -o cuda_hostfunc_probe.so cuda_hostfunc_probe.c -ldl
 *   LD_PRELOAD=/path/cuda_hostfunc_probe.so <cmd>
 *
 * Diagnostics only: it forwards every call unchanged and never fails one.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

typedef int (*launch_host_func_t)(void *stream, void *fn, void *userData);
static launch_host_func_t real_launch;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned long calls;
static int banner_done;

static void banner(void)
{
    if (banner_done) return;
    banner_done = 1;
    fprintf(stderr, "[hostfunc-probe] loaded in pid %d\n", getpid());
    fflush(stderr);
}

int cudaLaunchHostFunc(void *stream, void *fn, void *userData)
{
    if (!real_launch) {
        real_launch = (launch_host_func_t)dlsym(RTLD_NEXT, "cudaLaunchHostFunc");
        banner();
    }

    pthread_mutex_lock(&lock);
    unsigned long n = ++calls;
    pthread_mutex_unlock(&lock);

    /* The first few carry the information; after that just count, so a hot path
     * does not drown the worker log. */
    if (n <= 20 || n % 1000 == 0) {
        Dl_info info;
        const char *who = "?";
        const char *cb = "?";
        if (dladdr(__builtin_return_address(0), &info) && info.dli_fname)
            who = info.dli_fname;
        if (dladdr(fn, &info) && info.dli_sname)
            cb = info.dli_sname;
        fprintf(stderr,
                "[hostfunc-probe] call #%lu stream=%p callback=%p (%s) from %s\n",
                n, stream, fn, cb, who);
        fflush(stderr);
    }

    if (!real_launch) return 0;   /* never fail the caller */
    return real_launch(stream, fn, userData);
}

__attribute__((destructor))
static void report(void)
{
    fprintf(stderr, "[hostfunc-probe] pid %d total cudaLaunchHostFunc calls: %lu\n",
            getpid(), calls);
    fflush(stderr);
}
