// Validate peer transfers before enabling server collectives.
// Pageable H2D uploads can return before DMA completes: synchronize the upload
// device before peer access. Finish every writer before switching devices or
// processes for a read/compare; each kernel and IPC writer has an explicit fence.
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

static void ck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__global__ void peer_copy(unsigned char* out, const unsigned char* in, size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) out[i] = in[i];
}
static void exact(int fd, void* p, size_t n, bool writing) {
    auto* b = static_cast<unsigned char*>(p);
    while (n) {
        ssize_t k = writing ? write(fd, b, n) : read(fd, b, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) throw std::runtime_error("IPC pipe failed");
        b += k; n -= k;
    }
}
static void compare(int owner, unsigned char* ptr,
                    const std::vector<unsigned char>& expected, const char* method) {
    ck(cudaSetDevice(owner));
    ck(cudaDeviceSynchronize());
    std::vector<unsigned char> actual(expected.size());
    ck(cudaMemcpy(actual.data(), ptr, actual.size(), cudaMemcpyDeviceToHost));
    if (std::memcmp(actual.data(), expected.data(), actual.size()) == 0) return;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) throw std::runtime_error(
            std::string(method) + " mismatch on owner " + std::to_string(owner) +
            " at byte " + std::to_string(i) + " size " + std::to_string(actual.size()));
    }
}
static void randomize(std::vector<unsigned char>& data, std::mt19937& rng) {
    for (size_t i = 0; i < data.size();) {
        auto value = rng();
        size_t bytes = std::min(sizeof(value), data.size() - i);
        std::memcpy(data.data() + i, &value, bytes); i += bytes;
    }
}
static int ipc_child(int actor) {
    ck(cudaSetDevice(actor));
    cudaIpcMemHandle_t handle;
    size_t n;
    exact(STDIN_FILENO, &handle, sizeof(handle), false);
    exact(STDIN_FILENO, &n, sizeof(n), false);
    if (n != 32 * 1024 * 1024) throw std::runtime_error("invalid IPC size");
    std::vector<unsigned char> data(n);
    exact(STDIN_FILENO, data.data(), n, false);
    unsigned char *remote, *local;
    ck(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&remote), handle,
                           cudaIpcMemLazyEnablePeerAccess));
    ck(cudaMalloc(&local, n));
    ck(cudaMemcpy(local, data.data(), n, cudaMemcpyHostToDevice));
    ck(cudaDeviceSynchronize());
    peer_copy<<<256, 256>>>(remote, local, n);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    ck(cudaFree(local)); ck(cudaIpcCloseMemHandle(remote));
    return 0;
}
static void ipc_write(int actor, int owner, unsigned char* target,
                      std::vector<unsigned char>& data) {
    ck(cudaSetDevice(owner));
    ck(cudaMemset(target, 0, data.size())); ck(cudaDeviceSynchronize());
    cudaIpcMemHandle_t handle; ck(cudaIpcGetMemHandle(&handle, target));
    int fds[2]; if (pipe(fds)) throw std::runtime_error("IPC pipe unavailable");
    std::string ordinal = std::to_string(actor);
    pid_t pid = fork();
    if (pid == 0) {
        close(fds[1]);
        if (dup2(fds[0], STDIN_FILENO) < 0) _exit(126);
        close(fds[0]);
        execl("/proc/self/exe", "p2p-probe", "--ipc", ordinal.c_str(), nullptr);
        _exit(127);
    }
    close(fds[0]);
    if (pid < 0) { close(fds[1]); throw std::runtime_error("IPC fork failed"); }
    size_t n = data.size();
    exact(fds[1], &handle, sizeof(handle), true);
    exact(fds[1], &n, sizeof(n), true);
    exact(fds[1], data.data(), n, true); close(fds[1]);
    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw std::runtime_error("IPC wait failed");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error("IPC child failed");
    compare(owner, target, data, "IPC write");
}
static void identity() {
    int count, driver;
    ck(cudaGetDeviceCount(&count)); ck(cudaDriverGetVersion(&driver));
    std::printf("{\"driver\":%d,\"uuids\":[", driver);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop; ck(cudaGetDeviceProperties(&prop, i));
        std::printf("%s\"GPU-", i ? "," : "");
        for (int j = 0; j < 16; ++j) {
            if (j == 4 || j == 6 || j == 8 || j == 10) std::printf("-");
            std::printf("%02x", static_cast<unsigned char>(prop.uuid.bytes[j]));
        }
        std::printf("\"");
    }
    bool advertised = count > 1;
    for (int actor = 0; actor < count; ++actor)
        for (int owner = 0; owner < count; ++owner) if (actor != owner) {
            int available; ck(cudaDeviceCanAccessPeer(&available, actor, owner));
            advertised = advertised && available;
        }
    std::printf("],\"advertised\":%s}\n", advertised ? "true" : "false");
}
static double elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
static void check() {
    const size_t sizes[] = {128*1024-1, 128*1024, 128*1024+1,
        512*1024-1, 512*1024, 512*1024+1, 1024*1024, 8*1024*1024, 32*1024*1024};
    int count; ck(cudaGetDeviceCount(&count));
    if (count < 2) throw std::runtime_error("not advertised: fewer than two GPUs");
    std::random_device entropy; std::mt19937 rng(entropy());
    int pairs = 0;
    double setup_s = 0, random_s = 0, memcpy_s = 0, write_s = 0, read_s = 0, ipc_s = 0, cleanup_s = 0;
    double sizes_s[9] = {};
    std::vector<double> pairs_s;
    for (int actor = 0; actor < count; ++actor)
        for (int owner = 0; owner < count; ++owner) if (actor != owner) {
            auto pair_start = std::chrono::steady_clock::now();
            auto phase = pair_start;
            int available; ck(cudaDeviceCanAccessPeer(&available, actor, owner));
            if (!available) throw std::runtime_error("not advertised");
            ck(cudaSetDevice(actor));
            auto enabled = cudaDeviceEnablePeerAccess(owner, 0);
            if (enabled == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else ck(enabled);
            unsigned char *local, *readback, *remote;
            ck(cudaMalloc(&local, sizes[8])); ck(cudaMalloc(&readback, sizes[8]));
            ck(cudaSetDevice(owner)); ck(cudaMalloc(&remote, sizes[8]));
            setup_s += elapsed(phase);
            int size_index = 0;
            for (size_t n : sizes) {
                auto size_start = std::chrono::steady_clock::now();
                phase = size_start;
                std::vector<unsigned char> data(n); randomize(data, rng);
                random_s += elapsed(phase); phase = std::chrono::steady_clock::now();
                ck(cudaSetDevice(actor));
                ck(cudaMemcpy(local, data.data(), n, cudaMemcpyHostToDevice));
                ck(cudaDeviceSynchronize());
                ck(cudaSetDevice(owner)); ck(cudaMemset(remote, 0, n));
                ck(cudaDeviceSynchronize());
                ck(cudaMemcpyPeer(remote, owner, local, actor, n));
                ck(cudaDeviceSynchronize()); compare(owner, remote, data, "cudaMemcpyPeer");
                memcpy_s += elapsed(phase); phase = std::chrono::steady_clock::now();
                randomize(data, rng);
                random_s += elapsed(phase); phase = std::chrono::steady_clock::now();
                ck(cudaSetDevice(actor)); ck(cudaMemcpy(local, data.data(), n, cudaMemcpyHostToDevice));
                ck(cudaDeviceSynchronize());
                ck(cudaSetDevice(owner)); ck(cudaMemset(remote, 0, n)); ck(cudaDeviceSynchronize());
                ck(cudaSetDevice(actor)); peer_copy<<<256, 256>>>(remote, local, n);
                ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
                compare(owner, remote, data, "SM peer write");
                write_s += elapsed(phase); phase = std::chrono::steady_clock::now();
                // Independent owner data prevents a wrong-address write/read from agreeing.
                randomize(data, rng);
                random_s += elapsed(phase); phase = std::chrono::steady_clock::now();
                ck(cudaSetDevice(owner)); ck(cudaMemcpy(remote, data.data(), n, cudaMemcpyHostToDevice));
                ck(cudaDeviceSynchronize());
                ck(cudaSetDevice(actor)); ck(cudaMemset(readback, 0, n));
                peer_copy<<<256, 256>>>(readback, remote, n);
                ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
                compare(actor, readback, data, "SM peer read");
                read_s += elapsed(phase);
                sizes_s[size_index++] += elapsed(size_start);
            }
            phase = std::chrono::steady_clock::now();
            std::vector<unsigned char> ipc_data(sizes[8]); randomize(ipc_data, rng);
            random_s += elapsed(phase); phase = std::chrono::steady_clock::now();
            ipc_write(actor, owner, remote, ipc_data);
            ipc_s += elapsed(phase); phase = std::chrono::steady_clock::now();
            ck(cudaSetDevice(owner)); ck(cudaFree(remote));
            ck(cudaSetDevice(actor)); ck(cudaFree(local)); ck(cudaFree(readback));
            cleanup_s += elapsed(phase);
            pairs_s.push_back(elapsed(pair_start));
            ++pairs;
        }
    std::printf("{\"passed\":true,\"pairs\":%d,\"sizes\":9,\"max_bytes\":33554432,"
        "\"timings_s\":{\"setup\":%.6f,\"randomize\":%.6f,\"memcpy\":%.6f,\"write\":%.6f,"
        "\"read\":%.6f,\"ipc\":%.6f,\"cleanup\":%.6f},\"pairs_s\":[",
        pairs, setup_s, random_s, memcpy_s, write_s, read_s, ipc_s, cleanup_s);
    for (size_t i = 0; i < pairs_s.size(); ++i) std::printf("%s%.6f", i ? "," : "", pairs_s[i]);
    std::printf("],\"sizes_s\":[");
    for (size_t i = 0; i < 9; ++i) std::printf("%s%.6f", i ? "," : "", sizes_s[i]);
    std::printf("]}\n");
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--ipc") return ipc_child(std::stoi(argv[2]));
        if (argc == 2 && std::string(argv[1]) == "--identity") identity();
        else if (argc == 2 && std::string(argv[1]) == "--check") check();
        else throw std::runtime_error("expected --identity or --check");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "p2p-probe: %s\n", error.what()); return 1;
    }
}
