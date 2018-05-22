#include "densestPointModel.hpp"

#include <model/socialFishModel.hpp>

namespace Fishmodel {

    DensestPointModel::DensestPointModel(Simulation& simulation, Agent* agent)
        : SocialFishModel(simulation, agent)
    {
    }

    void DensestPointModel::reinit() {}

    void DensestPointModel::step()
    {
        auto ipos = invertedFishTable(_simulation.fishes);
        auto max_element = std::max_element(
            ipos.begin(), ipos.end(), [](const std::pair<int, std::vector<size_t>>& p1,
                                          const std::pair<int, std::vector<size_t>>& p2) {
                return p1.second.size() < p2.second.size();
            });
        int tgt_position = max_element->first;

        float dir = _approximate_angle(_agent->headPos, _agent->tailPos);
        if (dir > 180)
            dir -= 360;
        dir *= M_PI / 180.0;

        _agent->direction = static_cast<real_t>(dir);
        _agent->headPos.first
            = RADIUS * cos(static_cast<double>(_deg2cell[tgt_position]) * M_PI / 180.0)
            + ARENA_CENTER.first;
        _agent->headPos.second
            = RADIUS * sin(static_cast<double>(_deg2cell[tgt_position]) * M_PI / 180.0)
            + ARENA_CENTER.second;

        std::cout << "Densest point was postion: " << tgt_position << std::endl;

        _agent->updateAgentPosition(_simulation.dt);
    }

} // namespace Fishmodel
