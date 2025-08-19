// HttpClient.h

# pragma once

#include <string>
#include <iostream>
#include <thread>
#include <cpr/cpr.h>

class HttpClient {
    public:
        HttpClient();
        void sendGetRequest(const std::string& url);
        std::string sendHandData(const std::vector<unsigned char>& imageData);
};
