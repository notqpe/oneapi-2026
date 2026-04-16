#include "permutations_cxx.h"
#include <algorithm>
#include <string>

static std::string getSignature(const std::string& s) {
    std::string sig = s;
    std::sort(sig.begin(), sig.end());
    return sig;
}

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& pair : dictionary) {
        groups[getSignature(pair.first)].push_back(pair.first);
    }

    for (auto& pair : dictionary) {
        const std::string& key = pair.first;
        std::vector<std::string>& perms = pair.second;

        std::string sig = getSignature(key);
        const auto& group = groups.at(sig);

        for (const auto& s : group) {
            if (s != key) {
                perms.push_back(s);
            }
        }
        std::sort(perms.begin(), perms.end(), std::greater<std::string>());
    }
}