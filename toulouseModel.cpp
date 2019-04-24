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
        : Behavior(simulation, agent), ARENA_CENTER({0.3093, 0.2965})
    {
        init();
    }

    void ToulouseModel::init() { reinit(); }

    void ToulouseModel::reinit()
    {
        _time = 0;
        _position.x = ARENA_CENTER.first;
        _position.y = ARENA_CENTER.second;
    }

    void ToulouseModel::step()
    {
#if 0
        qDebug() << _agent->headPos.first << " " << _agent->headPos.second;

        _position.x = _agent->headPos.first;
        _position.y = _agent->headPos.second;

        _agent->headPos.first = ARENA_CENTER.first + 0.29 * std::cos(_time * M_PI / 180);
        _agent->headPos.second = ARENA_CENTER.second + 0.29 * std::sin(_time * M_PI / 180);
        _agent->direction = 0;

        _time += 5;
        if (_time > 360)
            _time = 0;
        _agent->updateAgentPosition(_simulation.dt);

#else

        _speed.vx = (_agent->headPos.first - _position.x) / _simulation.dt;
        _speed.vy = (_agent->headPos.second - _position.y) / _simulation.dt;

        _position.x = _agent->headPos.first - ARENA_CENTER.first;
        _position.y = _agent->headPos.second - ARENA_CENTER.second;

        _angular_direction = angle_to_pipi(_agent->direction);

        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result
            = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end()) {
            qDebug() << "fish " << _position.x << " " << _position.y << " " << _agent->direction
                     << " " << angle_to_pipi(_agent->direction);
            return;
        }
        qDebug() << "robot " << _position.x << " " << _position.y << " " << _agent->direction << " "
                 << angle_to_pipi(_agent->direction);

        //        std::cout << "v " << _speed.vx << " " << _speed.vy << std::endl;
        //        std::cout << "p " << _position.x << " " << _position.y << std::endl;

        _is_kicking = true;

        // // decide which individual is the next one to kick
        // _kicking_idx = _fish[0]->id();
        // double tkicker = _fish[0]->time_kicker();
        // for (uint i = 1; i < _fish.size(); ++i) {
        //     if (_fish[i]->time_kicker() < tkicker) {
        //         _kicking_idx = i;
        //         tkicker = _fish[i]->time_kicker();
        //     }
        // }
        // _fish[_kicking_idx]->is_kicking() = true;

        // the individuals decide on the desired position
        stimulate(); // kicking individual goes first

        // apply attractors/repulsors and update the fish intuitions
        // (for the kicker, the rest are in their gliding phase)
        interact();

        // update position and velocity information -- actual move step
        move();

        _is_kicking = false;
#endif
    }

    void ToulouseModel::stimulate()
    {
        _time += _kick_duration;
        _desired_position.x = _position.x + _kick_length * std::cos(_angular_direction);
        _desired_position.y = _position.y + _kick_length * std::sin(_angular_direction);

        _desired_speed.vx = (_desired_position.x - _position.x) / _kick_duration;
        _desired_speed.vy = (_desired_position.y - _position.y) / _kick_duration;
    }

    void ToulouseModel::interact()
    {
        int num_fish = _simulation.fishes.size();

        // kicker advancing to the new position
        _position = _desired_position;
        _speed = _desired_speed;

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
        } while (std::sqrt(qx * qx + qy * qy) > radius);
    }

    void ToulouseModel::move()
    {
        _position.x += ARENA_CENTER.first;
        _position.y += ARENA_CENTER.second;

        _agent->headPos.first = _position.x;
        _agent->headPos.second = _position.y;
        _agent->speed = (std::pow(_speed.vx, 2) + std::pow(_speed.vy, 2)
                            + 2 * std::abs(_speed.vx) * std::abs(_speed.vy)
                                * std::cos(std::atan2(_speed.vy, _speed.vx)))
            / _simulation.dt;
        _agent->updateAgentPosition(_simulation.dt);
    }

    state_t ToulouseModel::compute_state() const
    {
        size_t num_fish = _simulation.fishes.size();

        Eigen::VectorXd distances(num_fish);
        Eigen::VectorXd perception(num_fish);
        Eigen::VectorXd thetas(num_fish);
        Eigen::VectorXd phis(num_fish);

        for (uint i = 0; i < num_fish; ++i) {
            auto fish = reinterpret_cast<ToulouseModel*>(_simulation.fishes[i].second);
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

    Position<double> ToulouseModel::position() const { return _position; }

    double ToulouseModel::time_kicker() const { return _time + _kick_duration; }

    double ToulouseModel::time() const { return _time; }
    double& ToulouseModel::time() { return _time; }

    double ToulouseModel::angular_direction() const { return _angular_direction; }
    double& ToulouseModel::angular_direction() { return _angular_direction; }

    double ToulouseModel::peak_velocity() const { return _peak_velocity; }
    double& ToulouseModel::peak_velocity() { return _peak_velocity; }

    double ToulouseModel::kick_length() const { return _kick_length; }
    double& ToulouseModel::kick_length() { return _kick_length; }

    double& ToulouseModel::kick_duration() { return _kick_duration; }
    double ToulouseModel::kick_duration() const { return _kick_duration; }

    bool& ToulouseModel::is_kicking() { return _is_kicking; }
    bool ToulouseModel::is_kicking() const { return _is_kicking; }

    int ToulouseModel::id() const { return _id; }
    int& ToulouseModel::id() { return _id; }

} // namespace Fishmodel
