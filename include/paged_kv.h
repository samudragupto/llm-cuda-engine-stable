#pragma once
#include <vector>
#include <unordered_map>

struct Sequence {
    int seq_id;
    int length;
    std::vector<int> blocks;
};

class PagedKVManager {
private:
    int total_blocks;
    int block_size;
    std::vector<int> free_blocks;

public:
    std::unordered_map<int, Sequence> active_sequences;

    PagedKVManager(int total_blocks, int block_size);
    bool allocate_sequence(int seq_id);
    bool append_token(int seq_id);
    void free_sequence(int seq_id);
};