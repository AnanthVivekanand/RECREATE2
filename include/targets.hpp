#pragma once
#include <cstdint>
#include <cstring>

struct salt_result {
    uint64_t salt_lo;
    uint64_t salt_hi;
    uint32_t score;
};

enum TargetType : uint8_t {
    TGT_PREFIX          = 0,
    TGT_LEADING_ZEROS   = 1,
    TGT_PREFIX_PLUS_ZEROS = 2
};

static constexpr int MAX_TARGETS = 8;

// Pre-swizzled target for branchless kernel checking.
// Masks/values match LE Keccak lane byte order:
//   w0 = (uint32_t)(res[1] >> 32)  — addr bytes 0-3
//   w1 = res[2]                     — addr bytes 4-11
//   w2 = res[3]                     — addr bytes 12-19
struct TargetSpec {
    TargetType type;
    uint8_t    id;
    uint8_t    prefix_nibbles;
    uint8_t    _pad;
    uint32_t   threshold;

    uint32_t   mask0, val0;
    uint64_t   mask1, val1;
    uint64_t   mask2, val2;

    char       name[32];
};

struct MultiResult {
    salt_result results[MAX_TARGETS];
    uint32_t    found_mask;
    uint32_t    num_targets;
};

// Convert big-endian prefix bytes to LE-lane-order mask/value pairs.
inline void precompute_target_masks(TargetSpec& tgt,
                                    const uint8_t* prefix_bytes,
                                    int prefix_nibbles) {
    tgt.mask0 = 0; tgt.val0 = 0;
    tgt.mask1 = 0; tgt.val1 = 0;
    tgt.mask2 = 0; tgt.val2 = 0;
    tgt.prefix_nibbles = static_cast<uint8_t>(prefix_nibbles);

    int full_bytes = prefix_nibbles / 2;
    bool half = (prefix_nibbles & 1);
    int total_bytes = full_bytes + (half ? 1 : 0);

    for (int i = 0; i < total_bytes; i++) {
        uint8_t mask_byte = (i < full_bytes) ? 0xFF : 0xF0;
        uint8_t val_byte = prefix_bytes[i] & mask_byte;

        if (i < 4) {
            tgt.mask0 |= (uint32_t)mask_byte << (i * 8);
            tgt.val0  |= (uint32_t)val_byte  << (i * 8);
        } else if (i < 12) {
            int j = i - 4;
            tgt.mask1 |= (uint64_t)mask_byte << (j * 8);
            tgt.val1  |= (uint64_t)val_byte  << (j * 8);
        } else {
            int j = i - 12;
            tgt.mask2 |= (uint64_t)mask_byte << (j * 8);
            tgt.val2  |= (uint64_t)val_byte  << (j * 8);
        }
    }
}
