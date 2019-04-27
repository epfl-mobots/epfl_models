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
#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>

namespace Fishmodel {

    ToulouseModel::ToulouseModel(Simulation& simulation, Agent* agent)
        : Behavior(simulation, agent),
          // clang-format off
          ARENA_CENTER(
//              {0.3093, 0.2965}
               {0.262, 0.255}

              // clang-format on
          )
    {
        init();
    }

    void ToulouseModel::init() { reinit(); }

    void ToulouseModel::reinit()
    {
        _time = 0;
        _current_time = 0;

        if (perceived_agents >= _simulation.agents.size()) {
            qDebug() << "Correcting the number of perceived individuals to N-1";
            perceived_agents = _simulation.agents.size() - 1;
        }

        stepper();
        _position.x = -(_id - 1. - _simulation.agents.size() / 2) * body_length;
        angular_direction() = _id * 2. * M_PI / (_simulation.agents.size() + 1);
        _position.y = -0.1;
    }

    void ToulouseModel::step()
    {
        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end()) {
            _position.x = _agent->headPos.first - ARENA_CENTER.first;
            _position.y = _agent->headPos.second - ARENA_CENTER.second;
            return;
        }

        qDebug() << "robot " << _position.x << " " << _position.y << " " << _angular_direction
                 << " " << angle_to_pipi(_agent->direction);

        //        {
        //            std::lock_guard<std::mutex> lock(_mtx);
        //            if (_kick_flag) {
        //                _kick_flag = false;

        //                auto model = reinterpret_cast<ToulouseModel*>(_simulation.robots[0].second);
        //                _kicking_idx = model->id();
        //                double tkicker = model->time_kicker();
        //                for (uint i = 1; i < _simulation.robots.size(); ++i) {
        //                    auto model = reinterpret_cast<ToulouseModel*>(_simulation.robots[i].second);
        //                    if (model->time_kicker() < tkicker) {
        //                        auto prev_kicker = reinterpret_cast<ToulouseModel*>(_simulation.robots[_kicking_idx].second);
        //                        model->is_kicking() = true;
        //                        prev_kicker->is_kicking() = false;
        //                        _kicking_idx = i;
        //                        tkicker = model->time_kicker();
        //                    }
        //                }
        //            }
        //            else {
        //                for (uint i = 0; i < _simulation.robots.size(); ++i) {
        //                    auto model = reinterpret_cast<ToulouseModel*>(_simulation.robots[i].second);
        //                    if (model->is_kicking())
        //                        _kicking_idx = i;

        //                    if (model->time_kicker() < tkicker) {
        //                        auto prev_kicker = reinterpret_cast<ToulouseModel*>(_simulation.robots[_kicking_idx].second);
        //                        model->is_kicking() = true;
        //                        prev_kicker->is_kicking() = false;
        //                        _kicking_idx = i;
        //                        tkicker = model->time_kicker();
        //                    }
        //                }
        //            }
        //        }

        _is_kicking = true;

        // the individuals decide on the desired position
        stimulate(); // kicking individual goes first

        // apply attractors/repulsors and update the fish intuitions
        // (for the kicker, the rest are in their gliding phase)
        interact();

        // update position and velocity information -- actual move step
        move();

        {
            std::lock_guard<std::mutex> lock(_mtx);
            _is_kicking = true;
            _kicking_idx = -1;
        }
    }

    void ToulouseModel::stimulate()
    {

        {
            std::lock_guard<std::mutex> lock(_val_mtx);
            _time += _kick_duration;
        }
        _desired_position.x = _position.x + _kick_length * std::cos(_angular_direction);
        _desired_position.y = _position.y + _kick_length * std::sin(_angular_direction);

        _desired_speed.vx = (_desired_position.x - _position.x) / _kick_duration;
        _desired_speed.vy = (_desired_position.y - _position.y) / _kick_duration;
    }

    void ToulouseModel::interact()
    {
        int num_fish = _simulation.agents.size();

        // computing the state for the focal individual
        // distances -> distances to neighbours
        // perception -> angle of focal individual compared to neighbours
        // thetas -> angles to center
        // phis -> relative bearing difference
        Eigen::VectorXd distances, perception, thetas, phis;
        std::tie(distances, perception, thetas, phis) = compute_state();

        // indices to nearest neighbours
        std::vector<int> nn_idcs = sort_neighbours(distances, _id, Order::INCREASING);

        // compute influence from the environment to the focal fish
        Eigen::VectorXd influence = Eigen::VectorXd::Zero(num_fish);
        for (int i = 0; i < num_fish; ++i) {
            auto model = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            if (model->id() == _id)
                continue;

            double attraction = wall_distance_attractor(distances(i), radius)
                * wall_perception_attractor(perception(i)) * wall_angle_attractor(phis(i));

            double alignment = alignment_distance_attractor(distances(i), radius)
                * alignment_perception_attractor(perception(i))
                * alignment_angle_attractor(phis(i));

            influence(i) = std::abs(attraction + alignment);
        }

        // indices to highly influential individuals
        std::vector<int> inf_idcs = sort_neighbours(influence, _id, Order::DECREASING);

        // in case the influence from neighbouring fish is insignificant,
        // then use the nearest neighbours
        double inf_sum = std::accumulate(influence.data(), influence.data() + influence.size(), 0.);
        std::vector<int> idcs = inf_idcs;
        if (inf_sum < 1.0e-6)
            idcs = nn_idcs;

        // step using the model
        double r_w, theta_w;
        std::tie(r_w, theta_w) = model_stepper(radius);

        double qx, qy;
        do {
            stepper(); // decide on the next kick length, duration, peak velocity
            free_will({distances, perception, thetas, phis}, {r_w, theta_w},
                idcs); // throw in some free will

            // rejection test -- don't want to hit the wall
            qx = _desired_position.x + (_kick_length + body_length) * std::cos(_angular_direction);
            qy = _desired_position.y + (_kick_length + body_length) * std::sin(_angular_direction);

            qDebug() << std::sqrt(qx * qx + qy * qy) << " " << qx << " " << qy << " " << _position.x
                     << " " << _position.y;
        } while (std::sqrt(qx * qx + qy * qy) > radius);
    }

    void ToulouseModel::move()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);

        // kicker advancing to the new position
        _position = _desired_position;
        _speed = _desired_speed;

        // up to this point everything is calculated on a circle of
        // radius r and origin (0, 0), but the control command
        // is given with respect to the setup center
        _agent->headPos.first = _position.x + ARENA_CENTER.first;
        _agent->headPos.second = _position.y + ARENA_CENTER.second;
        //        _agent->updateAgentPosition(_simulation.dt);
        _agent->updateAgentPosition(_kick_duration);
    }

    state_t ToulouseModel::compute_state() const
    {
        size_t num_fish = _simulation.agents.size();

        Eigen::VectorXd distances = Eigen::VectorXd::Zero(num_fish);
        Eigen::VectorXd perception = Eigen::VectorXd::Zero(num_fish);
        Eigen::VectorXd thetas = Eigen::VectorXd::Zero(num_fish);
        Eigen::VectorXd phis = Eigen::VectorXd::Zero(num_fish);

        for (uint i = 0; i < num_fish; ++i) {
            auto model
                = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            if (model->id() == _id)
                continue;

            auto fish = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            double posx = fish->position().x;
            double posy = fish->position().y;
            double direction = fish->angular_direction();

            distances(i) = std::sqrt(
                std::pow(_desired_position.x - posx, 2) + std::pow(_desired_position.y - posy, 2));

            thetas(i) = std::atan2(posy - _desired_position.y, posx - _desired_position.x);

            perception(i) = angle_to_pipi(thetas(i) - _angular_direction);

            phis(i) = angle_to_pipi(direction - _angular_direction);
        }

        return {distances, perception, thetas, phis};
    }

    std::vector<int> ToulouseModel::sort_neighbours(
        const Eigen::VectorXd& values, const int kicker_idx, Order order) const
    {
        std::vector<int> neigh_idcs;
        for (int i = 0; i < values.rows(); ++i) {
            auto model
                = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            if (model->id() == _id)
                continue;
            neigh_idcs.push_back(i);
        }

        std::sort(std::begin(neigh_idcs), std::end(neigh_idcs), [&](int lhs, int rhs) {
            return (order == Order::INCREASING) ? (values(lhs) < values(rhs))
                                                : (values(lhs) > values(rhs));
        });

        return neigh_idcs;
    }

    void ToulouseModel::stepper()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);

        double bb;

        bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0, 1.) + 1.0e-16));
        _peak_velocity = velocity_coef * std::sqrt(2. / M_PI) * bb;

        bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0, 1.) + 1.0e-16));
        _kick_length = length_coef * std::sqrt(2. / M_PI) * bb;

        bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0, 1.) + 1.0e-16));
        _kick_duration = time_coef * std::sqrt(2. / M_PI) * bb;

        _kick_length = _peak_velocity * tau0 * (1. - std::exp(-_kick_duration / tau0));
    }

    void ToulouseModel::free_will(const_state_t state, const std::tuple<double, double>& model_out,
        const std::vector<int>& idcs)
    {
        double r_w, theta_w;
        Eigen::VectorXd distances, perception, thetas, phis;
        std::tie(r_w, theta_w) = model_out;
        std::tie(distances, perception, thetas, phis) = state;

        double g = std::sqrt(-2. * std::log(tools::random_in_range(0., 1.) + 1.0e-16))
            * std::sin(2. * M_PI * tools::random_in_range(0., 1.));

        double q = 1. * alpha
            * wall_distance_interaction(gamma_wall, wall_interaction_range, r_w, radius)
            / gamma_wall;

        double dphi_rand = gamma_rand * (1. - q) * g;
        double dphi_wall
            = wall_distance_interaction(gamma_wall, wall_interaction_range, r_w, radius)
            * wall_angle_interaction(theta_w);

        double dphi_attraction = 0;
        double dphi_ali = 0;

        for (int j = 0; j < perceived_agents; ++j) {
            int fidx = idcs[j];
            dphi_attraction += wall_distance_attractor(distances(fidx), radius)
                * wall_perception_attractor(perception(fidx)) * wall_angle_attractor(phis(fidx));
            dphi_ali += alignment_distance_attractor(distances(fidx), radius)
                * alignment_perception_attractor(perception(fidx))
                * alignment_angle_attractor(phis(fidx));
        }

        double dphi = dphi_rand + dphi_wall + dphi_attraction + dphi_ali;
        _angular_direction = angle_to_pipi(_angular_direction + dphi);
    }

    std::tuple<double, double> ToulouseModel::model_stepper(double radius) const
    {
        double r = std::sqrt(std::pow(_desired_position.x, 2) + std::pow(_desired_position.y, 2));
        double rw = radius - r;
        double theta = std::atan2(_desired_position.y, _desired_position.x);
        double thetaW = angle_to_pipi(_angular_direction - theta);
        return {rw, thetaW};
    }

    double ToulouseModel::wall_distance_interaction(
        double gamma_wall, double wall_interaction_range, double ag_radius, double radius) const
    {
        double x = std::max(0., ag_radius);
        return gamma_wall * std::exp(-std::pow(x / wall_interaction_range, 2));
    }

    double ToulouseModel::wall_angle_interaction(double theta) const
    {
        double a0 = 1.915651;
        return a0 * std::sin(theta) * (1. + .7 * std::cos(2. * theta));
    }

    double ToulouseModel::wall_distance_attractor(double distance, double radius) const
    {
        double a0 = 4.;
        double a1 = .03;
        double a2 = .2;
        return a0 * (distance - a1) / (1. + std::pow(distance / a2, 2));
    }

    double ToulouseModel::wall_perception_attractor(double perception) const
    {
        return 1.395 * std::sin(perception) * (1. - 0.33 * std::cos(perception));
    }

    double ToulouseModel::wall_angle_attractor(double phi) const
    {
        return 0.93263 * (1. - 0.48 * std::cos(phi) - 0.31 * std::cos(2. * phi));
    }

    double ToulouseModel::alignment_distance_attractor(double distance, double radius) const
    {
        double a0 = 1.5;
        double a1 = .03;
        double a2 = .2;
        return a0 * (distance + a1) * std::exp(-std::pow(distance / a2, 2));
    }

    double ToulouseModel::alignment_perception_attractor(double perception) const
    {
        return 0.90123 * (1. + .6 * std::cos(perception) - .32 * std::cos(2. * perception));
    }

    double ToulouseModel::alignment_angle_attractor(double phi) const
    {
        return 1.6385 * std::sin(phi) * (1. + .3 * std::cos(2. * phi));
    }

    double ToulouseModel::angle_to_pipi(double difference) const
    {
        do {
            if (difference < -M_PI)
                difference += 2. * M_PI;
            if (difference > M_PI)
                difference -= 2. * M_PI;
        } while (std::abs(difference) > M_PI);
        return difference;
    }

    Position<double> ToulouseModel::position() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _position;
    }

    Position<double>& ToulouseModel::position()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _position;
    }

    double ToulouseModel::time_kicker() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _time + _kick_duration;
    }

    double ToulouseModel::time() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _time;
    }

    double& ToulouseModel::time()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _time;
    }

    double ToulouseModel::angular_direction() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _angular_direction;
    }

    double& ToulouseModel::angular_direction()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _angular_direction;
    }

    double ToulouseModel::peak_velocity() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _peak_velocity;
    }

    double& ToulouseModel::peak_velocity()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _peak_velocity;
    }

    double ToulouseModel::kick_length() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _kick_length;
    }

    double& ToulouseModel::kick_length()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _kick_length;
    }

    double& ToulouseModel::kick_duration()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _kick_duration;
    }

    double ToulouseModel::kick_duration() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _kick_duration;
    }

    bool& ToulouseModel::is_kicking()
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _is_kicking;
    }

    bool ToulouseModel::is_kicking() const
    {
        std::lock_guard<std::mutex> lock(_val_mtx);
        return _is_kicking;
    }

    int ToulouseModel::id() const { return _id; }
    int& ToulouseModel::id() { return _id; }

} // namespace Fishmodel
