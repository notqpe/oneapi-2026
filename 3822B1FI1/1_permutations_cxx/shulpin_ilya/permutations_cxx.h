#ifndef __PERMUTATIONS_CXX_H
#define __PERMUTATIONS_CXX_H

#include <algorithm>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <utility>

using dictionary_t = std::map<std::string, std::vector<std::string>>;

void Permutations(dictionary_t& dictionary);

#endif  // __PERMUTATIONS_CXX_H
