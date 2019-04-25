#ifndef SIMU_TYPES_SPEED_HPP
#define SIMU_TYPES_SPEED_HPP

namespace simu {
    namespace types {

        template <typename T = float>
        struct Speed {
            Speed() : vx(0), vy(0), vz(0) {}

            T vx;
            T vy;
            T vz;
        };

    } // namespace types
} // namespace simu

#endif
