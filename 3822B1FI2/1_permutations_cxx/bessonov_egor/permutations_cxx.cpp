#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> groups;

  for (const auto& entry : dictionary) {
    std::string signature = entry.first;
    std::sort(signature.begin(), signature.end());
    groups[signature].push_back(entry.first);
  }

  for (auto& entry : dictionary) {
    const std::string& word = entry.first;

    std::string signature = word;
    std::sort(signature.begin(), signature.end());

    const auto& group = groups[signature];
    std::vector<std::string>& permutations = entry.second;

    permutations.clear();
    for (const auto& candidate : group) {
      if (candidate != word) {
        permutations.push_back(candidate);
      }
    }

    std::sort(permutations.begin(), permutations.end(), std::greater<std::string>());
  }
}