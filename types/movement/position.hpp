#ifndef SIMU_TYPES_POSITION_HPP
#define SIMU_TYPES_POSITION_HPP

namespace simu {
    namespace types {

        template <typename T = float>
        struct Position {
            Position() : x(0), y(0), z(0) {}

            T x;
            T y;
            T z;
        };

    } // namespace types
} // namespace simu

#endif
