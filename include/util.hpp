#pragma once
#include <array>
#include <string>
#include <stdexcept>
#include <cctype>

inline uint8_t hex_char(uint8_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("Invalid hex digit");
}

template <size_t N>
std::array<uint8_t,N> hex_to_bytes(const std::string &s) {
    if (s.size() != 2*N && !(s.size()==2*N+2 && s[0]=='0'&&s[1]=='x'))
        throw std::runtime_error("hex_to_bytes: wrong length");
    size_t offset = (s.size()==2*N+2 ? 2 : 0);
    std::array<uint8_t,N> out;
    for (size_t i=0;i<N;i++){
        out[i] = (hex_char(s[offset+2*i])<<4) | hex_char(s[offset+2*i+1]);
    }
    return out;
}

template <size_t N>
std::string to_hex(const std::array<uint8_t,N> &v) {
    static const char* hex ="0123456789abcdef";
    std::string s; s.reserve(2*N);
    for (auto b:v){
        s.push_back(hex[b>>4]);
        s.push_back(hex[b&0xf]);
    }
    return s;
}