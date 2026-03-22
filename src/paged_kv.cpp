#include "paged_kv.h"

PagedKVManager::PagedKVManager(int total_blocks, int block_size) 
    : total_blocks(total_blocks), block_size(block_size) {
    for (int i = total_blocks - 1; i >= 0; i--) {
        free_blocks.push_back(i);
    }
}

bool PagedKVManager::allocate_sequence(int seq_id) {
    if (active_sequences.find(seq_id) != active_sequences.end()) return false;
    if (free_blocks.empty()) return false;

    Sequence seq;
    seq.seq_id = seq_id;
    seq.length = 0;
    
    int block_id = free_blocks.back();
    free_blocks.pop_back();
    seq.blocks.push_back(block_id);

    active_sequences[seq_id] = seq;
    return true;
}

bool PagedKVManager::append_token(int seq_id) {
    auto it = active_sequences.find(seq_id);
    if (it == active_sequences.end()) return false;
    
    Sequence& seq = it->second;
    seq.length++;

    if (seq.length > seq.blocks.size() * block_size) {
        if (free_blocks.empty()) return false;
        
        int new_block_id = free_blocks.back();
        free_blocks.pop_back();
        seq.blocks.push_back(new_block_id);
    }
    return true;
}

void PagedKVManager::free_sequence(int seq_id) {
    auto it = active_sequences.find(seq_id);
    if (it != active_sequences.end()) {
        for (int block_id : it->second.blocks) {
            free_blocks.push_back(block_id);
        }
        active_sequences.erase(it);
    }
}