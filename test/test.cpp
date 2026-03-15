#include "miner.hpp"
#include "util.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <cstring>

int main() {
    // === CREATE2 Test: Uniswap v4 parameters ===
    std::string deployer_str    = "0x48E516B34A1274f49457b9C6182097796D0498Cb";
    std::string salt_str        = "0x72bed203c9a5eff37e1f55be91f742def6e0e5c7bd40398de517b6047b87ee78";
    std::string initcode_hash_str = "0x94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9";

    auto deployer    = hex_to_bytes<20>(deployer_str);
    auto salt        = hex_to_bytes<32>(salt_str);
    auto init_hash   = hex_to_bytes<32>(initcode_hash_str);

#ifdef HAVE_CUDA
    auto addr_gpu    = create2_address_gpu(deployer.data(), salt.data(), init_hash.data());
#endif
    auto addr_cpu    = create2_address_cpu(deployer.data(), salt.data(), init_hash.data());

    std::cout << "CREATE2 computed address (CPU): 0x" << to_hex(addr_cpu) << std::endl;

#ifdef HAVE_CUDA
    std::cout << "CREATE2 computed address (GPU): 0x" << to_hex(addr_gpu) << std::endl;
    assert(addr_gpu == addr_cpu);
    std::cout << "[PASS] CREATE2 GPU == CPU\n";
#endif

    // === CREATE3 Test: verify CPU and GPU produce the same result ===
    // Use an arbitrary factory address and salt
    std::string factory_str = "0x48E516B34A1274f49457b9C6182097796D0498Cb";
    std::string c3_salt_str = "0x0000000000000000000000000000000000000000000000000000000000000001";

    auto factory  = hex_to_bytes<20>(factory_str);
    auto c3_salt  = hex_to_bytes<32>(c3_salt_str);

    auto c3_addr_cpu = create3_address_cpu(factory.data(), c3_salt.data());
    std::cout << "CREATE3 computed address (CPU): 0x" << to_hex(c3_addr_cpu) << std::endl;

#ifdef HAVE_CUDA
    auto c3_addr_gpu = create3_address_gpu(factory.data(), c3_salt.data());
    std::cout << "CREATE3 computed address (GPU): 0x" << to_hex(c3_addr_gpu) << std::endl;
    assert(c3_addr_gpu == c3_addr_cpu);
    std::cout << "[PASS] CREATE3 GPU == CPU\n";
#endif

    // === CREATE3 structural test: verify two-step derivation ===
    // Manually compute: proxy = CREATE2(factory, salt, PROXY_HASH), then RLP CREATE
    auto proxy_addr = create2_address_cpu(factory.data(), c3_salt.data(), SOLADY_PROXY_INITCODE_HASH);
    std::cout << "CREATE3 proxy address (CPU):    0x" << to_hex(proxy_addr) << std::endl;

    // Verify that the intermediate proxy address is NOT the final CREATE3 address
    assert(proxy_addr != c3_addr_cpu);
    std::cout << "[PASS] CREATE3 final address != proxy address (two-step derivation confirmed)\n";

    // Verify the proxy address follows expected CREATE2 formula
    // by building the RLP buffer manually and checking the result
    {
        // Build RLP: [0xd6, 0x94, proxy(20), 0x01]
        uint8_t rlp_buf[136] = {0};
        rlp_buf[0] = 0xd6;
        rlp_buf[1] = 0x94;
        memcpy(rlp_buf + 2, proxy_addr.data(), 20);
        rlp_buf[22]  = 0x01;  // nonce
        rlp_buf[23]  = 0x01;  // keccak padding
        rlp_buf[135] = 0x80;  // padding end bit

        State s2{};
        uint64_t* s2_ptr   = reinterpret_cast<uint64_t*>(s2.data());
        const uint64_t* b2 = reinterpret_cast<const uint64_t*>(rlp_buf);
        for (int i = 0; i < 17; i++) s2_ptr[i] = b2[i];

        State res2{};
        keccak_f1600_cpu(s2, res2);

        std::array<uint8_t,20> manual_c3_addr;
        memcpy(manual_c3_addr.data(), reinterpret_cast<uint8_t*>(res2.data()) + 12, 20);

        std::cout << "CREATE3 manual derivation:      0x" << to_hex(manual_c3_addr) << std::endl;
        assert(manual_c3_addr == c3_addr_cpu);
        std::cout << "[PASS] CREATE3 manual two-step derivation matches create3_address_cpu\n";
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
}
