#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    struct Entry
    {
        std::string_view word;
        std::string sorted;
        dictionary_t::iterator it;
    };
    
    std::vector<Entry> entries;
    entries.reserve(dictionary.size());

    for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
        std::string s = it->first;
        std::sort(s.begin(), s.end());
        entries.push_back({it->first, std::move(s), it});
    }

    std::unordered_map<std::string_view, std::vector<Entry*>> groups;
    groups.reserve(dictionary.size());

    for (auto& e : entries) {
        groups[e.sorted].push_back(&e);
    }

    for (auto [_, vec] : groups) {
        std::sort(vec.begin(), vec.end(),
            [](const Entry* a, const Entry* b) {
                return a->word > b->word;
            });

        for (const Entry* self : vec) {
            auto& target = self->it->second;
            target.reserve(vec.size() - 1);

            for (const Entry* other : vec) {
                if (other != self) {
                    target.push_back(std::string(other->word));
                }
            }
        }
    }
}