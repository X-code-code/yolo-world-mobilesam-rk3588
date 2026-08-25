// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>

#include "clip_tokenizer.h"

int main(int argc, char** argv)
{
    CLIPTokenizer tokenizer;
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::vector<int> tokens = tokenizer.tokenize(argv[argument], 20, true);
        for (size_t index = 0; index < tokens.size(); ++index)
        {
            if (index != 0)
            {
                std::cout << ',';
            }
            std::cout << tokens[index];
        }
        std::cout << '\n';
    }
    return 0;
}
