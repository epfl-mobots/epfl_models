#include "socialFishModel.hpp"
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

namespace samsar {
    namespace types {
        struct WeightFunc : public defaults::WeightFunc {
            WeightFunc(const std::vector<float> w) : defaults::WeightFunc(w)
            {
                assert(w.size() == 2);
            }

            float operator()(const Simulation& /*sim*/, const AgentBehavior_t& ff,
                const AgentBehaviorStorage_t& f) const override
            {
                auto ff_social = reinterpret_cast<SocialFishModel*>(ff.second);
                auto f_social = reinterpret_cast<SocialFishModel*>(f.second.get());
                int num_cells = static_cast<int>(ff_social->num_cells());
                int cells_forward = ff_social->cells_forward();
                int cells_backward = ff_social->cells_backward();

                std::vector<int> forward;
                boost::push_back(forward,
                    boost::irange(ff_social->position() + ff_social->heading() * cells_forward,
                        ff_social->position() - ff_social->heading(), -ff_social->heading()));
                std::vector<int> backward;
                boost::push_back(
                    backward, boost::irange(ff_social->position() - ff_social->heading(),
                                  ff_social->position() + (-ff_social->heading() * cells_backward)
                                      - ff_social->heading(),
                                  -ff_social->heading()));
                std::for_each(forward.begin(), forward.end(),
                    [&](int& v) { (v < 0) ? v += num_cells : v %= num_cells; });
                std::for_each(backward.begin(), backward.end(),
                    [&](int& v) { (v < 0) ? v += num_cells : v %= num_cells; });

                const auto itf = std::find(forward.begin(), forward.end(), f_social->position());
                if (!(forward.end() == itf)) {
                    auto idx = std::distance(forward.begin(), itf);
                    return std::exp(_w[0] * idx);
                }

                const auto itb = std::find(backward.begin(), backward.end(), f_social->position());
                if (!(backward.end() == itb)) {
                    auto idx = std::distance(backward.begin(), itb);
                    return std::exp(_w[1] * idx);
                }
                return 0;
            }
        };
    } // namespace types
} // namespace samsar

namespace Fishmodel {

    SocialFishModel::SocialFishModel(Simulation& simulation, Agent* agent)
        : Behavior(simulation, agent),
          //          ARENA_CENTER({RobotControlSettings::get().setupMap().polygon().center().x(),
          //              RobotControlSettings::get().setupMap().polygon().center().y()}),
          ARENA_CENTER({0.300, 0.295}),
          RADIUS(0.237)
    {
        reinit();
    }

    void SocialFishModel::reinit()
    {
        _num_cells = 40;
        _group_threshold = 3;
        _cells_forward = 5;
        _cells_backward = 5;
        _min_speed = 1;
        _max_speed = 1;
        _prob_obey = 1.0f;
        _prob_move = 0.901f;
        _prob_change_speed = 0.1f;
        _heading_change_duration = 3;
        _sum_weight = {0.3f, -2.0f};
        _influence_alpha = 4;
        _heading_bias = Heading::UNDEFINED;

        _heading = random_heading();
        _next_heading = _heading;
        _heading_change = false;
        _speed = tools::random_in_range(_min_speed, _max_speed);

        _target_reset_threshold = 3;
        _heading_change_count = 0;
        _history_reset = 5;
        _history_count = 0;
        _heading_failed_attempts = 0;

        _create_deg_to_cell_map();

        _position = _approximate_discrete_pos(_agent->headPos, _agent->tailPos);
    }

    void SocialFishModel::step()
    {
        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result
            = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end())
            _position = _approximate_discrete_pos(_agent->headPos, _agent->tailPos);
        else {
            int current_pos = _approximate_discrete_pos(_agent->headPos, _agent->tailPos);
            int dist = std::abs(current_pos - _position);
            if (dist > static_cast<int>(_num_cells / 2))
                dist = 40 - std::max(current_pos, _position) + std::min(current_pos, _position);
            if (dist > _target_reset_threshold) {
                _position = _approximate_discrete_pos(_agent->headPos, _agent->tailPos);
                _position = (_position + _target_reset_threshold * _heading)
                    % static_cast<int>(_num_cells);
                if (_position < 0)
                    _position += _num_cells;
            }

            if (_estimated_heading != _heading)
                ++_heading_failed_attempts;
            else
                _heading_failed_attempts = 0;
            if (_heading_failed_attempts > 3 * static_cast<int>(std::ceil(1 / _simulation.dt))) { //
                robot is stuck
                    // attempt to move in the other direction for a while to get
                    unstuck _next_heading
                    = reverse_heading(_heading);
                std::cout << "robot is stuck" << std::endl;
            }
        }
        _update_history();

        _stimulate();
        _move();

        _agent->updateAgentPosition(_simulation.dt);
    }

    void SocialFishModel::_update_history()
    {
        _position_history.push_back(_approximate_discrete_pos(_agent->headPos, _agent->tailPos));

        if (_history_count++ > _history_reset) {
            int sum_hdg = 0;
            int avg_pos = 0;
            for (size_t i = 0; i < _position_history.size() - 1; ++i) {
                int diff = _position_history[i + 1] - _position_history[i];
                Heading est = to_heading(diff);
                if ((diff <= -static_cast<int>(_num_cells) / 2)
                    || (diff > static_cast<int>(_num_cells) / 2))
                    est = reverse_heading(est);
                else
                    avg_pos += _position_history[i];
                sum_hdg += est;
            }
            _estimated_heading = to_heading(sum_hdg);
            if (_estimated_heading == Heading::UNDEFINED)
                _estimated_heading = _heading;
            _position_history.clear();
            _history_count = 0;
        }
    }

    void SocialFishModel::_stimulate()
    {
        _my_group();
        float prob_obey_tstep = _social_influence();

        if (_heading_change) {
            if (_heading_change_count++ >= _heading_change_duration) {
                _heading_change_count = 0;
                _heading_change = false;
            }
            else
                return;
        }

        // heading
        if (_my_group_idcs.size() > 0) {
            FishGroup fg(_my_group_idcs);
            _next_heading = to_heading(fg.weighted_heading(_simulation,
                std::make_pair(_agent, this), std::make_shared<WeightFunc>(_sum_weight)));

            if (tools::random_in_range(0.0f, 1.0f) < 1 - prob_obey_tstep) {
                if (_heading_bias == Heading::UNDEFINED) {
                    _next_heading = reverse_heading(fg.sum_heading(_simulation.fishes));
                }
                else
                    _next_heading = _heading_bias;
                _heading_change = true;
            }
        }
        else {
            _next_heading = _heading;

            if (_heading_bias == Heading::UNDEFINED) {
                if (tools::random_in_range(0.0f, 1.0f) < 1 - prob_obey_tstep) {
                    _next_heading = reverse_heading(_heading);
                    _heading_change = true;
                }
            }
            else
                _next_heading = _heading_bias;
        }

        if (_next_heading == Heading::UNDEFINED) {
            if (_heading_bias == Heading::UNDEFINED)
                _next_heading = to_heading(random_heading());
            else
                _next_heading = _heading_bias;
        }
    }
    void SocialFishModel::_move()
    {

        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result
            = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);

        _heading = _next_heading;

        if (_heading_change && _heading_change_count == 1)
            return; // heading change costs one timestep

        float dir = _approximate_angle(_agent->headPos, _agent->tailPos);
        if (dir > 180)
            dir -= 360;
        dir *= M_PI / 180.0;

        if (_heading_change_count == 2) {
            _agent->direction = _agent->direction = static_cast<real_t>(dir) * -1;
        }
        else {
            _agent->direction = static_cast<real_t>(dir);
        }

        int tgt_position = _position;
        if (tools::random_in_range(0.0f, 1.0f) < _prob_move) {
            tgt_position = (_position + _speed * _heading) % static_cast<int>(_num_cells);
            if (tgt_position < 0)
                tgt_position += _num_cells;
        }

        _agent->headPos.first
            = RADIUS * cos(static_cast<double>(_deg2cell[tgt_position]) * M_PI / 180.0)
            + ARENA_CENTER.first;
        _agent->headPos.second
            = RADIUS * sin(static_cast<double>(_deg2cell[tgt_position]) * M_PI / 180.0)
            + ARENA_CENTER.second;

        if (result != _simulation.robots.end()) {
            _position = tgt_position;
        }
    }

    void SocialFishModel::_my_group()
    {
        //        int current_pos = _approximate_discrete_pos(_agent->headPos,
        //        _agent->tailPos); std::vector<int> pos; boost::push_back(pos,
        //        boost::irange(current_pos + 1,
        //                                  current_pos + _heading * _cells_forward +
        //                                  _heading, _heading));
        //        boost::push_back(pos, boost::irange(current_pos + (-_heading *
        //        _cells_backward),
        //                                  current_pos + _heading, _heading));
        //        std::for_each(
        //            pos.begin(), pos.end(), [&](int& v) { (v < 0) ? v += _num_cells
        //            : v %= _num_cells; });

        std::vector<int> pos;
        boost::push_back(pos, boost::irange(_position + 1,
                                  _position + _heading * _cells_forward + _heading, _heading));
        boost::push_back(pos, boost::irange(_position + (-_heading * _cells_backward),
                                  _position + _heading, _heading));
        std::for_each(
            pos.begin(), pos.end(), [&](int& v) { (v < 0) ? v += _num_cells : v %= _num_cells; });

        std::vector<size_t> candidate;
        auto ipos = invertedFishTable(_simulation.fishes);
        for (const auto& p : ipos) {
            auto result = std::find(pos.begin(), pos.end(), p.first);
            if (result == pos.end())
                continue;
            candidate.insert(candidate.end(), p.second.begin(), p.second.end());
        }

        _my_group_idcs.clear();
        if (candidate.size() >= static_cast<size_t>(_group_threshold))
            _my_group_idcs = candidate;
    }

    float SocialFishModel::_social_influence()
    {
        if (_my_group_idcs.size() == 0)
            return _prob_obey;

        // we take into account the fish that are located in front of the focal fish,
        // i.e. in its field of view. Fish that do not see a lot of neighbors in front
        // of them have higher probability to change direction, while ones that have
        // a lot of fish in front of them, are less prone to disobey the group.

        //        int current_pos = _approximate_discrete_pos(_agent->headPos,
        //        _agent->tailPos); std::vector<int> pos; boost::push_back(pos,
        //        boost::irange(current_pos,
        //                                  current_pos + _heading * _cells_forward +
        //                                  _heading, _heading));

        std::vector<int> pos;
        boost::push_back(pos,
            boost::irange(_position, _position + _heading * _cells_forward + _heading, _heading));

        std::for_each(
            pos.begin(), pos.end(), [&](int& v) { (v < 0) ? v += _num_cells : v %= _num_cells; });

        int neighs = 0;
        auto ipos = invertedFishTable(_simulation.fishes);
        for (int p : pos) {
            if (ipos.find(p) == ipos.end())
                continue;
            ++neighs; // we count the focal fish
        }

        return _prob_obey * (1 - 1.0f / static_cast<float>(std::pow(neighs + 1, _influence_alpha)));
    }

    float SocialFishModel::_approximate_angle(
        const Coord_t& head_p, const Coord_t& /*tail_p*/) const
    {
        Coord_t com_p = {/*std::abs(*/ head_p.first /*- tail_p.first) / 2*/,
            /*std::abs(*/ head_p.second /*- tail_p.second) / 2*/};
        Coord_t com_centered_p
            = {com_p.first - ARENA_CENTER.first, com_p.second - ARENA_CENTER.second};
        float deg = atan2f(static_cast<float>(com_centered_p.second),
                        static_cast<float>(com_centered_p.first))
            * 180.0f / static_cast<float>(M_PI);
        if (deg < 0)
            deg += 360;
        return deg;
    }

    int SocialFishModel::_approximate_discrete_pos(
        const Coord_t& head_p, const Coord_t& tail_p) const
    {
        float deg = _approximate_angle(head_p, tail_p);
        return _to_cell(deg);
    }

    void SocialFishModel::_create_deg_to_cell_map()
    {
        Eigen::VectorXf ls;
        ls.setLinSpaced(static_cast<int>(_num_cells), 0.0, 360.0);
        for (int i = 0; i < ls.size(); ++i)
            _deg2cell[static_cast<int>(i)] = ls[i];
    }

    int SocialFishModel::_to_cell(float deg) const
    {
        for (const auto& p : _deg2cell)
            if (deg < p.second)
                return p.first;
        return 0;
    }

} // namespace Fishmodel
