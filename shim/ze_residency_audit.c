// SPDX-License-Identifier: Apache-2.0
/* ze_residency_audit: LD_AUDIT version of the intel/compute-runtime#968 workaround.
 * Redirects zeMemAllocDevice / zeMemFree / zeContextMakeMemoryResident at symbol-binding time (la_symbind64),
 * which covers dlsym() lookups by UMF and the UR v2 adapter without interposing dlsym itself.
 * Interposing dlsym broke IGC: a wrapper changes the caller that RTLD_NEXT is resolved against.
 * Skips MakeMemoryResident on a device that does not own the allocation.
 * 2026-10-02: the UR v2 adapter calls most L0 entry points through DDI tables (zeGet*ProcAddrTable), which
 * symbol binding never sees, so its peer MakeMemoryResident went unhooked and peer allocations were
 * mirrored into system RAM (OOM at 32 GB). Tables requested BY THE v2 ADAPTER are now patched as well. Only the
 * adapter's: patching the loader's own internal table made the wrappers re-enter themselves (segfault, first try).
 * Symbol-bound and table-bound calls keep separate forwarding targets for the same reason.
 * Build: gcc -shared -fPIC -O2 -o ze_residency_audit.so ze_residency_audit.c -lpthread
 * Use:   LD_AUDIT=/path/ze_residency_audit.so  (ZE_RESIDENCY_SHIM_LOG=1 for stderr counters) */
#define _GNU_SOURCE
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef int ze_result_t;
typedef void *ze_handle;
#define NBUCKET 65536
struct ent { void *ptr; ze_handle dev; struct ent *next; };
static struct ent *tab[NBUCKET];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long skipped, passed, allocs;
static int logon = -1;
typedef ze_result_t (*alloc_fn)(ze_handle, const void *, size_t, size_t, ze_handle, void **);
typedef ze_result_t (*free_fn)(ze_handle, void *);
typedef ze_result_t (*res_fn)(ze_handle, ze_handle, void *, size_t);
static alloc_fn real_alloc; static free_fn real_free; static res_fn real_res;

static unsigned h(void *p) { uintptr_t x = (uintptr_t)p; x ^= x >> 21; x *= 0x9E3779B97F4A7C15ull; return (unsigned)(x >> 48) % NBUCKET; }
static void put(void *p, ze_handle d) {
    struct ent *e = malloc(sizeof *e); if (!e) return; e->ptr = p; e->dev = d;
    pthread_mutex_lock(&mu); unsigned b = h(p); e->next = tab[b]; tab[b] = e; allocs++; pthread_mutex_unlock(&mu);
}
static int owner(void *p, ze_handle *d) {
    int f = 0; pthread_mutex_lock(&mu);
    for (struct ent *e = tab[h(p)]; e; e = e->next) if (e->ptr == p) { *d = e->dev; f = 1; break; }
    pthread_mutex_unlock(&mu); return f;
}
static void del(void *p) {
    pthread_mutex_lock(&mu); struct ent **pp = &tab[h(p)];
    while (*pp) { if ((*pp)->ptr == p) { struct ent *e = *pp; *pp = e->next; free(e); break; } pp = &(*pp)->next; }
    pthread_mutex_unlock(&mu);
}
static ze_result_t w_alloc(ze_handle c, const void *d, size_t s, size_t a, ze_handle dev, void **p) {
    ze_result_t r = real_alloc(c, d, s, a, dev, p); if (r == 0 && p && *p) put(*p, dev); return r;
}
static ze_result_t w_free(ze_handle c, void *p) { del(p); return real_free(c, p); }
static ze_result_t w_res(ze_handle c, ze_handle dev, void *p, size_t s) {
    if (logon < 0) { const char *v = getenv("ZE_RESIDENCY_SHIM_LOG"); logon = v && *v == '1'; }
    ze_handle own;
    if (owner(p, &own) && own != dev) {
        unsigned long n = __atomic_add_fetch(&skipped, 1, __ATOMIC_RELAXED);
        if (logon && (n == 1 || n % 1024 == 0))
            fprintf(stderr, "[ze_residency_audit] skipped %lu peer MakeMemoryResident (passed %lu, tracked allocs %lu)\n", n, passed, allocs);
        return 0;
    }
    __atomic_add_fetch(&passed, 1, __ATOMIC_RELAXED);
    return real_res(c, dev, p, s);
}



/* ---- diagnostics: bytes requested as host / shared USM (system-RAM backed) ---- */
typedef ze_result_t (*shared_fn)(ze_handle, const void *, const void *, size_t, size_t, ze_handle, void **);
typedef ze_result_t (*host_fn)(ze_handle, const void *, size_t, size_t, void **);
static shared_fn s_shared, t_shared; static host_fn s_host, t_host;
static unsigned long long b_shared, b_host, n_shared, n_host, next_report = 1ull << 30;
static void usm_note(int sh, size_t s) {
    if (sh) { __atomic_add_fetch(&b_shared, s, __ATOMIC_RELAXED); __atomic_add_fetch(&n_shared, 1, __ATOMIC_RELAXED); }
    else    { __atomic_add_fetch(&b_host, s, __ATOMIC_RELAXED); __atomic_add_fetch(&n_host, 1, __ATOMIC_RELAXED); }
    unsigned long long tot = b_shared + b_host;
    if (logon == 1 && tot >= next_report) { next_report = tot + (1ull << 30);
        fprintf(stderr, "[ze_residency_audit] USM pid %d: shared %llu MiB in %llu allocs, host %llu MiB in %llu allocs\n",
                (int)getpid(), b_shared >> 20, n_shared, b_host >> 20, n_host); }
}
static ze_result_t s_w_shared(ze_handle c, const void *d, const void *h, size_t s, size_t a, ze_handle dev, void **p) { ze_result_t r = s_shared(c, d, h, s, a, dev, p); if (!r) usm_note(1, s); return r; }
static ze_result_t t_w_shared(ze_handle c, const void *d, const void *h, size_t s, size_t a, ze_handle dev, void **p) { ze_result_t r = t_shared(c, d, h, s, a, dev, p); if (!r) usm_note(1, s); return r; }
static ze_result_t s_w_host(ze_handle c, const void *d, size_t s, size_t a, void **p) { ze_result_t r = s_host(c, d, s, a, p); if (!r) usm_note(0, s); return r; }
static ze_result_t t_w_host(ze_handle c, const void *d, size_t s, size_t a, void **p) { ze_result_t r = t_host(c, d, s, a, p); if (!r) usm_note(0, s); return r; }
/* ---- DDI-table path (v2 adapter only) ---- */
typedef ze_result_t (*freeext_fn)(ze_handle, const void *, void *);
typedef ze_result_t (*gettbl_fn)(int, void **);
static alloc_fn t_alloc; static free_fn t_free; static freeext_fn t_freeext; static res_fn t_res;
static gettbl_fn real_getctx, real_getmem;
static struct link_map *adapter_map;
static void lg(const char *m) {
    if (logon < 0) { const char *v = getenv("ZE_RESIDENCY_SHIM_LOG"); logon = v && *v == '1'; }
    if (logon) fprintf(stderr, "[ze_residency_audit] %s pid %d\n", m, (int)getpid());
}
static ze_result_t t_w_alloc(ze_handle c, const void *d, size_t s, size_t a, ze_handle dev, void **p) {
    ze_result_t r = t_alloc(c, d, s, a, dev, p); if (r == 0 && p && *p) put(*p, dev); return r;
}
static ze_result_t t_w_free(ze_handle c, void *p) { del(p); return t_free(c, p); }
static ze_result_t t_w_freeext(ze_handle c, const void *d, void *p) { del(p); return t_freeext(c, d, p); }
static ze_result_t t_w_res(ze_handle c, ze_handle dev, void *p, size_t s) {
    if (logon < 0) { const char *v = getenv("ZE_RESIDENCY_SHIM_LOG"); logon = v && *v == '1'; }
    ze_handle own;
    if (owner(p, &own) && own != dev) {
        unsigned long n = __atomic_add_fetch(&skipped, 1, __ATOMIC_RELAXED);
        if (logon && (n == 1 || n % 1024 == 0))
            fprintf(stderr, "[ze_residency_audit] (table) skipped %lu peer MakeMemoryResident (passed %lu, tracked allocs %lu)\n", n, passed, allocs);
        return 0;
    }
    __atomic_add_fetch(&passed, 1, __ATOMIC_RELAXED);
    return t_res(c, dev, p, s);
}
/* ze_context_dditable_t: pfnCreate, pfnDestroy, pfnGetStatus, pfnSystemBarrier, pfnMakeMemoryResident (4), ... */
static ze_result_t w_getctx(int ver, void **t) {
    ze_result_t r = real_getctx(ver, t);
    if (r == 0 && t && t[4] && t[4] != (void *)t_w_res) { if (!t_res) t_res = (res_fn)t[4]; t[4] = (void *)t_w_res; lg("patched v2-adapter context table MakeMemoryResident"); }
    return r;
}
/* ze_mem_dditable_t: pfnAllocShared, pfnAllocDevice (1), pfnAllocHost, pfnFree (3), ..., pfnFreeExt (9), ... */
static ze_result_t w_getmem(int ver, void **t) {
    ze_result_t r = real_getmem(ver, t);
    if (r == 0 && t) {
        if (t[0] && t[0] != (void *)t_w_shared) { if (!t_shared) t_shared = (shared_fn)t[0]; t[0] = (void *)t_w_shared; }
        if (t[2] && t[2] != (void *)t_w_host) { if (!t_host) t_host = (host_fn)t[2]; t[2] = (void *)t_w_host; }
        if (t[1] && t[1] != (void *)t_w_alloc) { if (!t_alloc) t_alloc = (alloc_fn)t[1]; t[1] = (void *)t_w_alloc; }
        if (t[3] && t[3] != (void *)t_w_free) { if (!t_free) t_free = (free_fn)t[3]; t[3] = (void *)t_w_free; }
        if (t[9] && t[9] != (void *)t_w_freeext) { if (!t_freeext) t_freeext = (freeext_fn)t[9]; t[9] = (void *)t_w_freeext; }
        lg("patched v2-adapter mem table AllocDevice/Free/FreeExt");
    }
    return r;
}
__attribute__((destructor)) static void fini(void) {
    if (logon == 1) fprintf(stderr, "[ze_residency_audit] exit pid %d: skipped %lu passed %lu tracked allocs %lu; USM shared %llu MiB host %llu MiB\n", (int)getpid(), skipped, passed, allocs, b_shared >> 20, b_host >> 20);
}

unsigned int la_version(unsigned int v) { return v < LAV_CURRENT ? v : LAV_CURRENT; }
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    *cookie = (uintptr_t)map;
    if (map && map->l_name && strstr(map->l_name, "libur_adapter_level_zero_v2")) adapter_map = map;
    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}
uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook, uintptr_t *defcook, unsigned int *flags, const char *name) {
    if (name && name[0] == 'z' && name[1] == 'e') {
        if (!strcmp(name, "zeContextMakeMemoryResident")) {
            if (!real_res) real_res = (res_fn)sym->st_value;
            if (logon < 0) { const char *v = getenv("ZE_RESIDENCY_SHIM_LOG"); logon = v && *v == '1'; }
            if (logon) fprintf(stderr, "[ze_residency_audit] bound zeContextMakeMemoryResident pid %d\n", (int)getpid());
            return (uintptr_t)w_res;
        }
        if (!strcmp(name, "zeMemAllocDevice")) { if (!real_alloc) real_alloc = (alloc_fn)sym->st_value; return (uintptr_t)w_alloc; }
        if (!strcmp(name, "zeMemFree")) { if (!real_free) real_free = (free_fn)sym->st_value; return (uintptr_t)w_free; }
        if (!strcmp(name, "zeMemAllocShared")) { if (!s_shared) s_shared = (shared_fn)sym->st_value; return (uintptr_t)s_w_shared; }
        if (!strcmp(name, "zeMemAllocHost")) { if (!s_host) s_host = (host_fn)sym->st_value; return (uintptr_t)s_w_host; }
        /* only the v2 adapter's own table requests (never the loader's internal dispatch) */
        if (adapter_map && refcook && *refcook == (uintptr_t)adapter_map) {
            if (!strcmp(name, "zeGetContextProcAddrTable")) { real_getctx = (gettbl_fn)sym->st_value; return (uintptr_t)w_getctx; }
            if (!strcmp(name, "zeGetMemProcAddrTable")) { real_getmem = (gettbl_fn)sym->st_value; return (uintptr_t)w_getmem; }
        }
    }
    return sym->st_value;
}
