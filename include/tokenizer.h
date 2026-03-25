#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct LlamaTokenizer { 
    std::vector<std::string> vocab; 
    std::unordered_map<std::string, int> token_to_id;

    void load(const char* path); 
    std::string decode(int id); 
    std::string decode(const std::vector<int>& ids); 
    std::vector<int> encode(const std::string& text, bool add_bos = true);
};