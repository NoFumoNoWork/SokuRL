#include <cstddef>
#include <cstdint>
#include <cstring>

static_assert(sizeof(void *) == 8, "Host helper is x64, never inject into TH123");

extern "C" __declspec(dllexport) unsigned host_hash_abi() { return 1; }

extern "C" __declspec(dllexport) std::uint64_t fnv64(const unsigned char *bytes, std::size_t count)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0; i < count; ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

extern "C" __declspec(dllexport) void copy_bytes(void *target, const void *source, std::size_t count)
{
    std::memcpy(target, source, count);
}
