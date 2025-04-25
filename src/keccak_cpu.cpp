#include "keccak.hpp"
#include <cstring>

// very minimal CPU keccak256 (one-block, sponge padding)
std::array<uint8_t,32> keccak256_cpu(const uint8_t* data, size_t len){
    State A{}; // zero
    // absorb up to 136 B
    uint8_t buf[136] = {0};
    memcpy(buf, data, len);
    buf[len] = 0x01;
    buf[135] |= 0x80;
    // XOR into state lanes
    memcpy(A.data(), buf, 136);
    keccak_f1600_unrolled(A, A);
    // squeeze
    std::array<uint8_t,32> out;
    memcpy(out.data(), reinterpret_cast<uint8_t*>(A.data())+0, 32);
    return out;
}

std::array<uint8_t,20> create2_address_cpu(const uint8_t d[20],
                                           const uint8_t s[32],
                                           const uint8_t h[32]){
    uint8_t buf[1+20+32+32];
    buf[0]=0xff;
    memcpy(buf+1, d,20);
    memcpy(buf+1+20,s,32);
    memcpy(buf+1+20+32,h,32);
    auto hash = keccak256_cpu(buf,sizeof(buf));
    std::array<uint8_t,20> out;
    memcpy(out.data(), hash.data()+12,20);
    return out;
}