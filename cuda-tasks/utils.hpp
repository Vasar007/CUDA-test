// Copyright (C) 2018 Vasily Vasilyev (vasar007@yandex.ru)

#pragma once

#include <cassert>
#include <ctime>
#include <iterator>
#include <random>
#include <type_traits>


namespace utils
{

std::mt19937 create_random_engine()
{
    // Obtain a time-based seed:
    const auto seed = static_cast<unsigned long>(std::time(nullptr));
    return std::mt19937(seed);
}

std::mt19937 RANDOM_ENGINE = create_random_engine();

template <class Type>
typename std::enable_if<std::is_arithmetic<Type>::value, Type>::type
random_number(const Type& a = 0, const Type& b = std::numeric_limits<Type>::max())
{
    assert(a <= b);
    //static std::mt19937 RANDOM_ENGINE = create_random_engine();
    std::uniform_int_distribution<Type> distribution(a, b);
    return distribution(RANDOM_ENGINE);
}

template <class Container>
typename Container::value_type take_accidentally(const Container& cont)
{
    //static std::mt19937 RANDOM_ENGINE = create_random_engine();
    std::uniform_int_distribution<std::size_t> distribution(0, cont.size() - 1);
    return *std::next(std::begin(cont), distribution(RANDOM_ENGINE));
}

} // namespace utils
