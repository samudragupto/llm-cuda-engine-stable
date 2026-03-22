#include "memory_pool.h"
#include "model.h"

int main() {
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    MemPool scratch_pool(256ULL * 1024 * 1024);      

    Llama2Paged model(model_pool);
    model.load_weights("model_mixed.bin");

    std::vector<std::vector<int>> concurrent_prompts = {
        {1, 450, 7483, 310, 3444, 338},  // "The capital of France is"
        {1, 450, 7483, 310, 4363, 338},  // "The capital of Japan is" (Assuming 4363 is Japan)
        {1, 450, 7483, 310, 5634, 338}   // "The capital of Germany is" (Assuming 5634 is Germany)
    };
    
    GenerationConfig cfg;
    cfg.max_new_tokens = 50;
    cfg.repetition_penalty = 1.1f;

    model.chat_batched(scratch_pool, concurrent_prompts, cfg);

    return 0;
}