#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

void LlamaTokenizer::load(const char* path) { 
    FILE* f = fopen(path, "rb"); 
    if (!f) { printf("[FATAL] Could not open %s\n", path); exit(1); }
    
    int vs; 
    if (fread(&vs, 4, 1, f) != 1) { fclose(f); return; }
    
    vocab.resize(vs, ""); 
    for (int i = 0; i < vs; i++) { 
        int l = 0; 
        if (fread(&l, 4, 1, f) != 1) break;
        
        if (l > 0 && l < 4096) {
            std::string s(l, '\0'); 
            fread(&s[0], 1, l, f); 
            vocab[i] = s; 
        }
    } 
    fclose(f); 
    printf("Tokenizer loaded %d tokens.\n", vs);
}

std::string LlamaTokenizer::decode(int id) { 
    if (id < 0 || id >= vocab.size()) return ""; 
    
    std::string token = vocab[id];
    
    if (token == "<unk>" || token == "<s>" || token == "</s>") return "";
    if (token == "<0x0A>") return "\n";
    if (token == "<0x09>") return "\t";
    
    std::string clean = "";
    for (size_t i = 0; i < token.size(); ) {
        if (i + 2 < token.size() && 
            (unsigned char)token[i] == 0xE2 && 
            (unsigned char)token[i+1] == 0x80 && 
            (unsigned char)token[i+2] == 0x99) {
            clean += "'";
            i += 3;
        } 
        else if (i + 2 < token.size() && 
            (unsigned char)token[i] == 0xE2 && 
            (unsigned char)token[i+1] == 0x96 && 
            (unsigned char)token[i+2] == 0x81) {
            clean += " ";
            i += 3;
        } else {
            clean += token[i];
            i++;
        }
    }
    
    if (clean.size() == 6 && clean.substr(0,3) == "<0x" && clean.back() == '>') {
        int code;
        if (sscanf(clean.c_str(), "<0x%X>", &code) == 1) {
            return std::string(1, (char)code);
        }
    }

    return clean;
}

std::string LlamaTokenizer::decode(const std::vector<int>& ids) { 
    std::string s; 
    for (int id : ids) s += decode(id); 
    return s; 
}