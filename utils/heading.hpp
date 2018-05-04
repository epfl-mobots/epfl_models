#ifndef HEADING_HPP
#define HEADING_HPP

#include "random/random_generator.hpp"

#include <boost/algorithm/string.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace samsar {
    namespace types {
        enum Heading : int { CLOCKWISE = -1, UNDEFINED = 0, COUNTER_CLOCKWISE = 1 };

        template <typename T> Heading to_heading(T candidate)
        {
            if (candidate == 0)
                return Heading::UNDEFINED;
            else if (candidate > 0)
                return Heading::COUNTER_CLOCKWISE;
            else
                return Heading::CLOCKWISE;
        }

        //        Heading to_heading(const std::string& hdg);

        std::string to_str(Heading heading);
        Heading reverse_heading(Heading hdg);
        bool is_same_heading(Heading hdg1, Heading hdg2);
        Heading random_heading();

    } // namespace types
} // namespace samsar

#endif
