/**
 * @file
 * @author Leo Cazenille <leo.cazenille@gmail.com>
 *
 *
 */

#ifndef FACTORY_H
#define FACTORY_H

#include <string>
#include <vector>

#include "bm.hpp"
#include "bmWithWalls.hpp"
#include "densestPointModel.hpp"
#include "model.hpp"
#include "socialFishModel.hpp"
#include "zones.hpp"

namespace Fishmodel {

    class EpflSimulationFactory {
    public:
        Arena& arena;
        size_t nbFishes = 7;
        size_t nbRobots = 2;
        size_t nbVirtuals = 1;
        std::string behaviorFishes = "SFM";
        std::string behaviorRobots = "SFM";
        std::string behaviorVirtuals = "SFM";

        std::vector<std::pair<Coord_t, Coord_t>> wallsCoord;

        std::vector<std::vector<Coord3D_t>> trajectories;
        size_t currentTrajectoryIndex = 0;

    protected:
        Simulation* _sim = nullptr;

        template <class A>
        std::pair<Agent*, Behavior*> _createAgent(std::string const& behaviorType);

    public:
        EpflSimulationFactory(Arena& _arena) : arena(_arena) {}
        std::unique_ptr<Simulation> create();
    };

} // namespace Fishmodel
#endif
