#ifndef TOULOUSEMODEL_HPP
#define TOULOUSEMODEL_HPP

#include <AgentState.hpp>
#include <CoordinatesConversion.hpp>

#include "model.hpp"
#include "utils/heading.hpp"
#include "zones.hpp"

#include <map>

namespace Fishmodel {

    using namespace samsar;
    using namespace types;

    class ToulouseModel : public Behavior {
    public:
        ToulouseModel(Simulation& simulation, Agent* agent = nullptr);

        void init();
        virtual void reinit() override;
        virtual void step() override;

    protected:
        CoordinatesConversionPtr _coordinatesConversion;
        const Coord_t MIN_XY;
        const Coord_t ARENA_CENTER;
        const double RADIUS;
    };

} // namespace Fishmodel

#endif // TOULOUSEMODEL_HPP
