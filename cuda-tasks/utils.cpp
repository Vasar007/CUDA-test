// Copyright (C) 2018 Vasily Vasilyev (vasar007@yandex.ru)

#include <ctime>
#include <random>

#include "utils.hpp"


namespace utils
{

std::mt19937 create_random_engine()
{
    // Obtain a time-based seed:
    const auto seed = static_cast<unsigned long>(std::time(nullptr));
    return std::mt19937(seed);
}

} // namespace utils
