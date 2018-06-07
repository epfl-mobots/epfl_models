#include "biomimeticFishModel.hpp"

#include <SetupType.hpp>
#include <settings/CalibrationSettings.hpp>
#include <settings/CommandLineParameters.hpp>
#include <settings/RobotControlSettings.hpp>

#include <QDebug>

#include <algorithm>
#include <boost/range.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>
#include <eigen3/Eigen/Core>
#include <memory>

namespace Fishmodel {

    BiomimeticFishModel::BiomimeticFishModel(Simulation& simulation, Agent* agent)
        : Behavior(simulation, agent),
          MIN_XY({-0.221247, -0.165585}),
          ARENA_CENTER(
              {RobotControlSettings::get().setupMap().polygon().center().x() - MIN_XY.first,
                  RobotControlSettings::get().setupMap().polygon().center().y() - MIN_XY.second}),
          RADIUS(0.24f)
    {
        init();
    }

    void BiomimeticFishModel::init()
    {
        _perception_range = {0.20, 0.20};
        _perception_capability = {3, 3};
    }

    void BiomimeticFishModel::reinit()
    {
        _heading = random_heading();
        _position = _agent->headPos;
    }

    void BiomimeticFishModel::step()
    {
        std::pair<std::vector<State>, std::vector<State>> perceived_agent_states
            = _getPerceivedAgents();

        if ((perceived_agent_states.first.size() > 0)
            || (perceived_agent_states.second.size() > 0)) {

            // TODO: calculate weighted sum for heading
            // TODO: if heading != prev heading create turning trajectory
            // TODO: if back sum falls below a threshold create a turning trajectory
            // TODO: match forward speed
            // TODO: if too long in follow -> accelerate
            // TODO: if too long in lead increase probability of turning
            // TODO: if individuals forward = 0 high chance of turning
        }
        else {
            // TODO: remember last position where an individual was seen ?
        }

        _agent->updateAgentPosition(_simulation.dt);
    }

    std::pair<std::vector<State>, std::vector<State>> BiomimeticFishModel::_getPerceivedAgents()
    {
        int front = 0;
        int back = 0;
        std::pair<std::vector<State>, std::vector<State>> states;
        for (const auto& a : _simulation.agents) {
            if (a
                == std::make_pair(std::shared_ptr<Agent>(_agent), std::shared_ptr<Behavior>(this)))
                continue;

            State pfish_state = dynamic_cast<BiomimeticFishModel*>(a.second.get())->state();

            real_t r = (euclidean_distance(pfish_state.position, ARENA_CENTER)
                           + euclidean_distance(state().position, ARENA_CENTER))
                / 2;

            real_t theta1 = to_360(std::atan2(state().position.second - ARENA_CENTER.second,
                                       state().position.first - ARENA_CENTER.first)
                * 180 / M_PI);

            real_t theta2 = to_360(std::atan2(pfish_state.position.second - ARENA_CENTER.second,
                                       pfish_state.position.first - ARENA_CENTER.first)
                * 180 / M_PI);

            real_t theta = std::abs(theta1 - theta2);
            if (theta > 180)
                theta = 360 - theta;
            real_t arc_length = (theta / 360) * 2 * M_PI * r;
            pfish_state.distance_to_focal = arc_length;

            if (theta1 < theta2) {
                if (((_agent->direction * 180 / M_PI > 90)
                        || (_agent->direction * 180 / M_PI < -90))
                    && (front < _perception_capability[0])
                    && (arc_length < _perception_range[0])) // front
                {
                    pfish_state.position_description = PositionDescription::FRONT;
                    states.first.push_back(pfish_state);
                    ++front;
                }
                else if ((back < _perception_capability[1])
                    && (arc_length < _perception_range[1])) { // back
                    pfish_state.position_description = PositionDescription::BACK;
                    states.second.push_back(pfish_state);
                    ++back;
                }
            }
            else {
                if (((_agent->direction * 180 / M_PI < 90)
                        || (_agent->direction * 180 / M_PI > -90))
                    && (front < _perception_capability[0])
                    && (arc_length < _perception_range[0])) // front
                {
                    pfish_state.position_description = PositionDescription::FRONT;
                    states.first.push_back(pfish_state);
                    ++front;
                }
                else if ((back < _perception_capability[1])
                    && (arc_length < _perception_range[1])) { // back
                    pfish_state.position_description = PositionDescription::BACK;
                    states.second.push_back(pfish_state);
                    ++back;
                }
            }
        }
        return states;
    }

    inline real_t BiomimeticFishModel::euclidean_distance(
        const Coord_t& p1, const Coord_t& p2) const
    {
        return std::sqrt(std::pow(p1.first - p2.first, 2) + std::pow(p1.second - p2.second, 2));
    }

} // namespace Fishmodel
