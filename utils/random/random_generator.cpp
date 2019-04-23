#include "random_generator.hpp"

namespace simu {
    namespace tools {

        int8_t random_sgn()
        {
            if (random_in_range(0, 1) == 0)
                return 1;
            else
                return -1;
        }

    } // namespace tools
} // namespace simu
