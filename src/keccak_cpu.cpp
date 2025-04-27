#include "keccak.hpp"
#include <cstring>
#include <cstdio>

std::array<uint8_t,20> create2_address_cpu(const uint8_t deployer[20],
                                           const uint8_t salt[32],
                                           const uint8_t initHash[32]){
    
    // print salt
    printf("salt: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", salt[i]);
    }
    printf("\n");


    State s{};

    uint8_t buf[136] = {0};
    buf[0] = 0xff;
    memcpy(buf + 1, deployer, 20);
    memcpy(buf + 21, salt,    32);
    memcpy(buf + 53, initHash,32);
    buf[85]      = 0x01;
    buf[135]    |= 0x80;

    uint64_t* s_ptr   = reinterpret_cast<uint64_t*>(s.data());
    const uint64_t* b = reinterpret_cast<const uint64_t*>(buf);
    for (int i = 0; i < 17; i++) {
        s_ptr[i] = b[i];
    }

    keccak_f1600_cpu(s, s);

    std::array<uint8_t,20> out;
    uint8_t* p = reinterpret_cast<uint8_t*>(s.data());
    memcpy(out.data(), p + 12, 20);
    return out;
}