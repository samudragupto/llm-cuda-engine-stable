#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

void LlamaTokenizer::load(const char* path) { 
    FILE* f = fopen(path, "rb"); 
    if (!f) exit(1); 
    
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
            token_to_id[s] = i;
        }
    } 
    fclose(f); 
}

std::string LlamaTokenizer::decode(int id) { 
    if (id < 0 || id >= vocab.size()) return ""; 
    std::string token = vocab[id];
    if (token == "<unk>" || token == "<s>" || token == "</s>") return "";
    if (token == "<0x0A>") return "\n";
    if (token == "<0x09>") return "\t";
    
    std::string clean = "";
    for (size_t i = 0; i < token.size(); ) {
        if (i + 2 < token.size() && (unsigned char)token[i] == 0xE2 && (unsigned char)token[i+1] == 0x80 && (unsigned char)token[i+2] == 0x99) {
            clean += "'"; i += 3;
        } else if (i + 2 < token.size() && (unsigned char)token[i] == 0xE2 && (unsigned char)token[i+1] == 0x96 && (unsigned char)token[i+2] == 0x81) {
            clean += " "; i += 3;
        } else {
            clean += token[i]; i++;
        }
    }
    
    if (clean.size() == 6 && clean.substr(0,3) == "<0x" && clean.back() == '>') {
        int code;
        if (sscanf(clean.c_str(), "<0x%X>", &code) == 1) return std::string(1, (char)code);
    }
    return clean;
}

std::string LlamaTokenizer::decode(const std::vector<int>& ids) { 
    std::string s; 
    for (int id : ids) s += decode(id); 
    return s; 
}

std::vector<int> LlamaTokenizer::encode(const std::string& text, bool add_bos) {
    std::vector<int> tokens;
    if (add_bos) tokens.push_back(1);
    
    std::string sp_text = "";
    for (size_t i = 0; i < text.size(); i++) {
        if (text[i] == ' ') sp_text += "\xE2\x96\x81";
        else sp_text += text[i];
    }
    
    if (!sp_text.empty() && sp_text.substr(0, 3) != "\xE2\x96\x81") {
        sp_text = "\xE2\x96\x81" + sp_text;
    }

    size_t i = 0;
    while (i < sp_text.size()) {
        int best_id = -1;
        size_t best_len = 0;
        
        for (size_t len = 1; len <= sp_text.size() - i; len++) {
            std::string sub = sp_text.substr(i, len);
            if (token_to_id.count(sub)) {
                best_id = token_to_id[sub];
                best_len = len;
            }
        }
        
        if (best_id != -1) {
            tokens.push_back(best_id);
            i += best_len;
        } else {
            char byte_str[10];
            snprintf(byte_str, sizeof(byte_str), "<0x%02X>", (unsigned char)sp_text[i]);
            std::string fallback = byte_str;
            if (token_to_id.count(fallback)) tokens.push_back(token_to_id[fallback]);
            i++;
        }
    }
    return tokens;
}