// HttpClient.cpp

#include "HttpClient.h"
#include <iostream>
#include <thread>
#include "spdlog/spdlog.h"

HttpClient::HttpClient(){
    // 
}

void HttpClient::sendGetRequest(const std::string& url){
    spdlog::info("Sending request to {}", url);

    std::thread([url]() {
        cpr::Response r = cpr::Get(cpr::Url{url},  cpr::Timeout{5000});
        if (r.status_code == 200){
            spdlog::info("Success request to: {}", url);
        }
        else {
            spdlog::error("Failed request to: {} | Code: {}", url, r.status_code);
        }
    }).detach();
}
