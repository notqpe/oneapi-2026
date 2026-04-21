#include "permutations_cxx.h"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> canonicalMap;
    
    for (const auto& entry : dictionary) {
        std::string sortedKey = entry.first;
        std::sort(sortedKey.begin(), sortedKey.end());
        canonicalMap[sortedKey].push_back(entry.first);
    }
    
    for (auto& entry : dictionary) {
        const std::string& key = entry.first;
        std::vector<std::string>& permutations = entry.second;
        
        std::string sortedKey = key;
        std::sort(sortedKey.begin(), sortedKey.end());
        
        const std::vector<std::string>& allPermutations = canonicalMap[sortedKey];
        
        for (const std::string& perm : allPermutations) {
            if (perm != key) {
                permutations.push_back(perm);
            }
        }
        
        std::sort(permutations.begin(), permutations.end(),
                  [](const std::string& a, const std::string& b) {
                      return a > b;
                  });
    }
}