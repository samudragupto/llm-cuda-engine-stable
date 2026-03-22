#include "model.h"
#include <iostream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("Initializing GPU Memory Pools...\n");
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);
    MemPool scratch_pool(256ULL * 1024 * 1024);

    printf("Loading LLaMA Engine (INT8/FP16)...\n");
    Llama2Paged model(model_pool);
    model.load_weights("model_mixed.bin");

    printf("\n================================================\n");
    printf("Custom LLM Engine Ready!\n");
    printf("Type your prompt and press Enter. Type '/bye' to exit.\n");
    printf("(Supported keywords for demo: France, Japan, Germany, life)\n");
    printf("================================================\n\n");

    GenerationConfig cfg;
    cfg.max_new_tokens = 60;
    cfg.repetition_penalty = 1.1f;
    cfg.temperature = 1.0f;

    std::string user_input;
    while (true) {
        std::cout << "\n>>> You: ";
        std::getline(std::cin, user_input);

        if (user_input == "/bye" || user_input == "exit") {
            printf("Shutting down.\n");
            break;
        }

        if (user_input.empty()) continue;

        std::vector<int> prompt_ids = {1}; 

        if (user_input == "France") {
             prompt_ids = {1, 450, 7483, 310, 3444, 338}; 
        } else if (user_input == "Japan") {
             prompt_ids = {1, 450, 7483, 310, 4363, 338}; 
        } else if (user_input == "Germany") {
             prompt_ids = {1, 450, 7483, 310, 5634, 338};
        } else if (user_input == "life") {
             prompt_ids = {1, 1724, 338, 263, 2983, 29889};
        } else {
             prompt_ids = {1, 450, 7483, 310, 3444, 338}; 
        }

        std::vector<std::vector<int>> batch = { prompt_ids };
        model.chat_batched(scratch_pool, batch, cfg);
    }

    return 0;
}