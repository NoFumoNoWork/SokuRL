#include "AudioMute.hpp"
#include <Windows.h>
#include <cstring>

namespace SokuRLBridge {
bool installAudioMute()
{
    // th123 1.10a music and sound-effect master setters. Replace only the
    // floating-point argument load with zero; keep buffer and timing logic.
    // BGM setter 0x403D10 is also identified by upstream ReplayDnD.
    // SFX setter 0x401C20 is called by 0x43E230 and writes TextureManager+0x450.
    const DWORD addresses[] = {0x00403D16, 0x00401C27};
    const unsigned char original[] = {0xD9, 0x45, 0x08}; // fld dword ptr [ebp+8]
    const unsigned char muted[] = {0xD9, 0xEE, 0x90};    // fldz; nop
    for (const auto address : addresses)
        if (std::memcmp(reinterpret_cast<void *>(address), original, sizeof(original)))
            return false;
    for (const auto address : addresses) {
        auto *target = reinterpret_cast<void *>(address);
        DWORD protection = 0;
        if (!VirtualProtect(target, sizeof(muted), PAGE_EXECUTE_WRITECOPY, &protection))
            return false;
        std::memcpy(target, muted, sizeof(muted));
        DWORD ignored = 0;
        if (!VirtualProtect(target, sizeof(muted), protection, &ignored))
            return false;
    }
    return FlushInstructionCache(GetCurrentProcess(), nullptr, 0) != FALSE;
}
}
