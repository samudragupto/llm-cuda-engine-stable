#include "model.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

std::mutex inference_mutex;
Llama2Paged* global_model = nullptr;
MemPool* global_scratch = nullptr;

void handle_chat(const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content("{\"error\": \"Invalid JSON\"}", "application/json");
        return;
    }

    std::string prompt = "";
    if (body.contains("messages")) {
        auto msgs = body["messages"];
        if (msgs.is_array() && !msgs.empty()) {
            prompt = msgs.back()["content"].get<std::string>();
        }
    } else if (body.contains("prompt")) {
        prompt = body["prompt"].get<std::string>();
    }

    GenerationConfig cfg;
    if (body.contains("max_tokens")) cfg.max_new_tokens = body["max_tokens"];
    if (body.contains("temperature")) cfg.temperature = body["temperature"];
    if (body.contains("repetition_penalty")) cfg.repetition_penalty = body["repetition_penalty"];

    printf("[API] Received Prompt: '%s'\n", prompt.c_str());

    std::vector<std::string> results;
    float tokens_per_sec = 0.0f;
    {
        std::lock_guard<std::mutex> lock(inference_mutex);
        
        std::vector<int> tokens = global_model->tokenizer.encode(prompt, true);
        std::vector<std::vector<int>> batch = { tokens };

        auto t1 = std::chrono::high_resolution_clock::now();
        
        results = global_model->chat_batched(*global_scratch, batch, cfg);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t2 - t1;
        tokens_per_sec = cfg.max_new_tokens / dt.count();
    }

    json response;
    response["id"] = "chatcmpl-cuda-engine";
    response["object"] = "chat.completion";
    response["created"] = std::time(nullptr);
    response["model"] = "tinyllama-1.1b-int8";
    
    json choice;
    choice["index"] = 0;
    choice["message"] = { {"role", "assistant"}, {"content", results[0]} };
    choice["finish_reason"] = "stop";
    
    response["choices"] = json::array({ choice });
    response["usage"] = { 
        {"completion_tokens", cfg.max_new_tokens},
        {"prompt_tokens", 0}, 
        {"total_tokens", cfg.max_new_tokens}
    };
    response["backend_stats"] = {
        {"tokens_per_sec", tokens_per_sec}
    };

    res.set_content(response.dump(), "application/json");
}

int main() {
#ifdef _WIN32
    system("chcp 65001"); 
#endif

    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    global_scratch = new MemPool(256ULL * 1024 * 1024);

    global_model = new Llama2Paged(model_pool);
    global_model->load_weights("model_mixed.bin");

    httplib::Server svr;

    svr.Post("/v1/chat/completions", handle_chat);
    
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.listen("0.0.0.0", 8085);

    return 0;
}