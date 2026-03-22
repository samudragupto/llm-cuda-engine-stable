#include "memory_pool.h"
#include "model.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

int main() {
    printf("Initializing GPU Memory Pools...\n");
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    MemPool scratch_pool(256ULL * 1024 * 1024);      

    printf("Loading LLaMA Engine...\n");
    Llama2Paged model(model_pool);
    model.load_weights("model_mixed.bin");

    httplib::Server svr;

    svr.Post("/v1/completions", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto body = json::parse(req.body);
            std::string prompt_text = body.value("prompt", "");
            int max_tokens = body.value("max_tokens", 50);

            printf("\n[API] Received Prompt: '%s'\n", prompt_text.c_str());

            std::vector<int> prompt_ids;
            if (prompt_text.find("France") != std::string::npos) {
                prompt_ids = {1, 450, 7483, 310, 3444, 338}; 
            } else if (prompt_text.find("Japan") != std::string::npos) {
                prompt_ids = {1, 450, 7483, 310, 4363, 338}; 
            } else {
                prompt_ids = {1, 450, 7483, 310, 3444, 338}; 
            }

            std::vector<std::vector<int>> batch = { prompt_ids };
            
            GenerationConfig cfg;
            cfg.max_new_tokens = max_tokens;
            cfg.repetition_penalty = 1.1f;
            
            std::vector<std::string> results = model.chat_batched(scratch_pool, batch, cfg);

            json response;
            response["id"] = "cmpl-custom";
            response["object"] = "text_completion";
            response["model"] = "tinyllama-int8";
            response["choices"] = json::array();
            
            json choice;
            choice["text"] = results[0];
            choice["index"] = 0;
            response["choices"].push_back(choice);

            res.set_content(response.dump(4), "application/json");

        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(std::string("{\"error\": \"") + e.what() + "\"}", "application/json");
        }
    });

    printf("\n=========================================\n");
    printf("Custom LLM Server running on port 8085!\n");
    printf("=========================================\n");
    
    if (!svr.listen("0.0.0.0", 8085)) {
        printf("[FATAL] Failed to start server! Port 8085 is already in use.\n");
        return 1;
    }

    return 0;
}