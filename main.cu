#include "memory_pool.h"
#include "model.h"

int main() {
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    MemPool scratch_pool(256ULL * 1024 * 1024);      

    Llama2Paged model(model_pool);
    model.load_weights("model_mixed.bin");

    std::vector<std::vector<int>> concurrent_prompts = {
        {1, 450, 7483, 310, 3444, 338}, 
        {1, 450, 7483, 310, 28705, 338}, 
        {1, 450, 7483, 310, 5635, 338}  
    };
    
    GenerationConfig cfg;
    cfg.max_new_tokens = 50;
    cfg.repetition_penalty = 1.1f;

    model.chat_batched(scratch_pool, concurrent_prompts, cfg);

    return 0;
}