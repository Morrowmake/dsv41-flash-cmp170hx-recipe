// CPU-only runtime substitute for content verification tests.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorPeerAccessAlreadyEnabled = 704;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
constexpr int cudaIpcMemLazyEnablePeerAccess = 1;
struct cudaIpcMemHandle_t { char path[64]; };
struct cudaDeviceProp { struct { char bytes[16]; } uuid; };
struct Allocation { size_t n; int owner; std::string path; bool imported; int fd; };
static std::map<void*, Allocation> allocations;
static int device = 0;
inline const char* cudaGetErrorString(int) { return "mock CUDA error"; }
inline int cudaGetLastError() { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaGetDeviceCount(int* out) {
    *out = std::getenv("MOCK_COUNT") ? std::atoi(std::getenv("MOCK_COUNT")) : 4; return 0;
}
inline int cudaDriverGetVersion(int* out) { *out = 13030; return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp* out, int index) {
    std::memset(out, 0, sizeof(*out)); out->uuid.bytes[15] = index + 1; return 0;
}
inline int cudaSetDevice(int index) { device = index; return 0; }
inline int cudaDeviceCanAccessPeer(int* out, int actor, int owner) {
    *out = !(std::getenv("MOCK_NO_PEER") && actor == 3 && owner == 0); return 0;
}
inline int cudaDeviceEnablePeerAccess(int, int) { return 0; }
// Device memory is anonymous shared memory (memfd): no file name exists, so a probe that fails a check, throws past
// its cudaFree calls or is killed leaves nothing behind; the kernel frees it with the last mapping and descriptor.
template <class T> int cudaMalloc(T** out, size_t n) {
    int fd = memfd_create("p2p-cpu", MFD_CLOEXEC); if (fd < 0) return 1;
    if (ftruncate(fd, n)) { close(fd); return 1; }
    void* ptr = mmap(nullptr, n, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { close(fd); return 1; }
    // The IPC child opens the region through the owner's descriptor while the owner waits for it.
    std::string path = "/proc/" + std::to_string(getpid()) + "/fd/" + std::to_string(fd);
    allocations[ptr] = Allocation{n, device, path, false, fd};
    *out = static_cast<T*>(ptr); return 0;
}
inline int cudaFree(void* ptr) {
    auto found = allocations.find(ptr);
    if (found == allocations.end() || found->second.owner != device) return 1;
    munmap(ptr, found->second.n); close(found->second.fd);
    allocations.erase(found); return 0;
}
inline int cudaMemcpy(void* out, const void* in, size_t n, int kind) {
    void* ptr = kind == cudaMemcpyHostToDevice ? out : const_cast<void*>(in);
    auto found = allocations.find(ptr);
    if (found == allocations.end() || found->second.owner != device || n > found->second.n) return 1;
    std::memcpy(out, in, n); return 0;
}
inline int cudaMemset(void* ptr, int value, size_t n) {
    auto found = allocations.find(ptr);
    if (found == allocations.end() || found->second.owner != device) return 1;
    std::memset(ptr, value, n); return 0;
}
inline bool corrupt(const char* method) {
    const char* value = std::getenv("MOCK_CORRUPT");
    return value && std::string(value) == method;
}
inline int cudaMemcpyPeer(void* out, int owner, const void* in, int actor, size_t n) {
    if (allocations.at(out).owner != owner || allocations.at(const_cast<void*>(in)).owner != actor) return 1;
    std::memcpy(out, in, n);
    if (corrupt("memcpy")) static_cast<unsigned char*>(out)[n-1] ^= 1;
    return 0;
}
inline void mock_peer_copy(unsigned char* out, const unsigned char* in, size_t n) {
    auto dst = allocations.at(out), src = allocations.at(const_cast<unsigned char*>(in));
    if (dst.imported && corrupt("ipc")) { std::memcpy(out, in, n); out[n-1] ^= 1; }
    else if (dst.owner != device && corrupt("write")) {
        // A remote write lands in unrelated memory; the owner retains its sentinel.
    } else {
        std::memcpy(out, in, n);
        if (src.owner != device && corrupt("read")) out[n-1] ^= 1;
    }
}
inline int cudaIpcGetMemHandle(cudaIpcMemHandle_t* out, void* ptr) {
    auto a = allocations.at(ptr);
    if (a.owner != device) return 1;
    std::snprintf(out->path, sizeof(out->path), "%s", a.path.c_str()); return 0;
}
inline int cudaIpcOpenMemHandle(void** out, cudaIpcMemHandle_t handle, int) {
    int fd = open(handle.path, O_RDWR); if (fd < 0) return 1;
    size_t n = lseek(fd, 0, SEEK_END);
    void* ptr = mmap(nullptr, n, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0); close(fd);
    if (ptr == MAP_FAILED) return 1;
    allocations[ptr] = Allocation{n, -1, handle.path, true, -1}; *out = ptr; return 0;
}
inline int cudaIpcCloseMemHandle(void* ptr) {
    auto a = allocations.at(ptr); munmap(ptr, a.n); allocations.erase(ptr); return 0;
}
