#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> groups;

  for (const auto& pair : dictionary) {
    std::string key = pair.first;
    std::sort(key.begin(), key.end());
    groups[key].push_back(pair.first);
  }

  for (auto& pair : dictionary) {
    const std::string& word = pair.first;

    std::string key = word;
    std::sort(key.begin(), key.end());

    const auto& group = groups[key];

    std::vector<std::string> permutations;

    for (const auto& w : group) {
      if (w != word) permutations.push_back(w);
    }

    std::sort(permutations.rbegin(), permutations.rend());

    pair.second = permutations;
  }
}