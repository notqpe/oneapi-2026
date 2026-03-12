#include "permutations_cxx.h"
#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    groups.reserve(dictionary.size());
    
    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        std::string sorted = it->first;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(it->first);
    }
    
    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        const auto& key = it->first;
        auto& permutations = it->second;
        
        std::string sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());
        
        const auto& group = groups[sorted_key];
        
        if (group.size() > 1) {
            permutations.reserve(group.size() - 1);
        }
        
        permutations.insert(permutations.end(), group.begin(), group.end());
        
        auto key_pos = std::find(permutations.begin(), permutations.end(), key);
        if (key_pos != permutations.end()) {
            permutations.erase(key_pos);
        }
        
        std::sort(permutations.begin(), permutations.end(), 
                  std::greater<std::string>());
    }
}