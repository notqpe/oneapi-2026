#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> anagram_groups;

    for (const auto& pair : dictionary) {
        std::string sorted_word = pair.first;
        std::sort(sorted_word.begin(), sorted_word.end());
        anagram_groups[sorted_word].push_back(pair.first);
    }

    for (auto& pair : dictionary) {
        std::string sorted_word = pair.first;
        std::sort(sorted_word.begin(), sorted_word.end());

        const auto& group = anagram_groups[sorted_word];
        std::vector<std::string>& permutations = pair.second;
        permutations.clear();

        for (const auto& candidate : group) {
            if (candidate != pair.first) {
                permutations.push_back(candidate);
            }
        }

        std::sort(permutations.rbegin(), permutations.rend());
    }
}
