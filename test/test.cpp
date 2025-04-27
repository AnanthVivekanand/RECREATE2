#include "miner.hpp"
#include "util.hpp"
#include <cassert>
#include <iostream>
#include <string>

int main() {
    // Uniswap v4 parameters
    std::string deployer_str    = "0x48E516B34A1274f49457b9C6182097796D0498Cb";
    std::string salt_str        = "0x72bed203c9a5eff37e1f55be91f742def6e0e5c7bd40398de517b6047b87ee78";
    std::string initcode_hash_str = "0x94d114296a5af85c1fd2dc039cdaa32f1ed4b0fe0868f02d888bfc91feb645d9";

    // Convert hex strings to byte arrays
    auto deployer    = hex_to_bytes<20>(deployer_str);
    auto salt        = hex_to_bytes<32>(salt_str);
    auto init_hash   = hex_to_bytes<32>(initcode_hash_str);

    // Compute CREATE2 address on GPU and CPU
#ifdef HAVE_CUDA
    auto addr_gpu    = create2_address_gpu(deployer.data(), salt.data(), init_hash.data());
#endif
    auto addr_cpu    = create2_address_cpu(deployer.data(), salt.data(), init_hash.data());

#ifdef HAVE_CUDA
    // Verify both implementations match
    assert(addr_gpu == addr_cpu);
#endif

    // Print the computed address
    std::cout << "CREATE2 computed address: 0x" << to_hex(addr_cpu) << std::endl;
    return 0;
}
