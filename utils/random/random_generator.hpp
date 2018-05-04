#ifndef RANDOM_GENERATOR_HPP
#define RANDOM_GENERATOR_HPP

#include "rand_utils.hpp"
#include <iostream>
#include <random>
#include <type_traits>

namespace samsar {
    namespace tools {

        class RandomGenerator {
        public:
            RandomGenerator() : _rgen(randutils::auto_seed_128{}.base()) {}
            std::mt19937 gen() const { return _rgen; }

        private:
            std::mt19937 _rgen;
        };

        template <typename T,
            typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
        T random_in_range(T lower_bound, T upper_bound)
        {
            std::uniform_real_distribution<T> distribution(lower_bound, upper_bound);
            auto gen = RandomGenerator().gen();
            return distribution(gen);
        }

        template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
        T random_in_range(T lower_bound, T upper_bound)
        {
            std::uniform_int_distribution<T> distribution(lower_bound, upper_bound);
            auto gen = RandomGenerator().gen();
            return distribution(gen);
        }

        template <typename T,
            typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
        T normal_random(T mean, T sigma)
        {
            std::normal_distribution<T> distribution(mean, sigma);
            auto gen = RandomGenerator().gen();
            return distribution(gen);
        }

        int8_t random_sgn();

    } // namespace tools
} // namespace samsar

#endif
