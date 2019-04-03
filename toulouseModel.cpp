#include "toulouseModel.hpp"
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

    ToulouseModel::ToulouseModel(Simulation& simulation, Agent* agent)
        : Behavior(simulation, agent),
          MIN_XY({-0.221247, -0.165585}),
          ARENA_CENTER(
              {RobotControlSettings::get().setupMap().polygon().center().x() - MIN_XY.first,
                  RobotControlSettings::get().setupMap().polygon().center().y() - MIN_XY.second}),
          //          ARENA_CENTER({0.300, 0.295}),
          RADIUS(0.24)
    {
        init();
    }

    void ToulouseModel::init()
    {
        reinit();
    }

    void ToulouseModel::reinit()
    {
    }

    void ToulouseModel::step()
    {
        _agent->updateAgentPosition(_simulation.dt);
    }

} // namespace Fishmodel
