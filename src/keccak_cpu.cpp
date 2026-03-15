#include "keccak.hpp"
#include "miner.hpp"
#include <cstring>
#include <cstdio>

std::array<uint8_t,20> create2_address_cpu(const uint8_t deployer[20],
                                           const uint8_t salt[32],
                                           const uint8_t initHash[32]){
    printf("salt: ");
    for (int i = 0; i < 32; i++) printf("%02x", salt[i]);
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
    for (int i = 0; i < 17; i++) s_ptr[i] = b[i];

    keccak_f1600_cpu(s, s);

    std::array<uint8_t,20> out;
    uint8_t* p = reinterpret_cast<uint8_t*>(s.data());
    memcpy(out.data(), p + 12, 20);
    return out;
}

std::array<uint8_t,20> create3_address_cpu(const uint8_t factory[20],
                                           const uint8_t salt[32]) {
    printf("salt: ");
    for (int i = 0; i < 32; i++) printf("%02x", salt[i]);
    printf("\n");

    // Step 1: compute proxy address via CREATE2
    State s{};
    uint8_t buf[136] = {0};
    buf[0] = 0xff;
    memcpy(buf + 1, factory, 20);
    memcpy(buf + 21, salt,    32);
    memcpy(buf + 53, SOLADY_PROXY_INITCODE_HASH, 32);
    buf[85]  = 0x01;
    buf[135] = 0x80;

    uint64_t* s_ptr   = reinterpret_cast<uint64_t*>(s.data());
    const uint64_t* b = reinterpret_cast<const uint64_t*>(buf);
    for (int i = 0; i < 17; i++) s_ptr[i] = b[i];

    State res{};
    keccak_f1600_cpu(s, res);

    // Extract proxy address (bytes 12-31)
    uint8_t proxy[20];
    memcpy(proxy, reinterpret_cast<uint8_t*>(res.data()) + 12, 20);

    // Step 2: compute final address via RLP CREATE (nonce=1)
    // Buffer: [0xd6, 0x94, proxy(20), 0x01] = 23 bytes + keccak padding
    uint8_t rlp_buf[136] = {0};
    rlp_buf[0] = 0xd6;
    rlp_buf[1] = 0x94;
    memcpy(rlp_buf + 2, proxy, 20);
    rlp_buf[22]  = 0x01;  // nonce
    rlp_buf[23]  = 0x01;  // keccak padding
    rlp_buf[135] = 0x80;  // padding end bit

    State s2{};
    uint64_t* s2_ptr   = reinterpret_cast<uint64_t*>(s2.data());
    const uint64_t* b2 = reinterpret_cast<const uint64_t*>(rlp_buf);
    for (int i = 0; i < 17; i++) s2_ptr[i] = b2[i];

    State res2{};
    keccak_f1600_cpu(s2, res2);

    std::array<uint8_t,20> out;
    memcpy(out.data(), reinterpret_cast<uint8_t*>(res2.data()) + 12, 20);
    return out;
}