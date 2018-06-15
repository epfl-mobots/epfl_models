#ifndef SOCIALFISHMODEL_HPP
#define SOCIALFISHMODEL_HPP

#include <AgentState.hpp>
#include <CoordinatesConversion.hpp>

#include "model.hpp"
#include "utils/heading.hpp"
#include "zones.hpp"

#include <map>

namespace Fishmodel {

    using namespace samsar;
    using namespace types;

    class SocialFishModel : public Behavior {
    public:
        SocialFishModel(Simulation& simulation, Agent* agent = nullptr);

        void init();
        virtual void reinit() override;
        virtual void step() override;

        size_t num_cells() const { return _num_cells; }
        int cells_forward() const { return _cells_forward; }
        int cells_backward() const { return _cells_backward; }
        int position() const { return _position; }
        Heading heading() const { return _heading; }
        Heading estimated_heading() const { return _estimated_heading; }

        template <typename T> int sgn(T val) { return (T(0) < val) - (val < T(0)); }

    protected:
        virtual void _stimulate();
        virtual void _move();

        std::map<int, std::vector<size_t>> invertedFishTable(
            const std::vector<AgentBehavior_t>& fish)
        {
            std::map<int, std::vector<size_t>> ipos;
            for (size_t i = 0; i < fish.size(); ++i) {
                int pos = _approximate_discrete_pos(fish[i].first->headPos, fish[i].first->tailPos);
                if (ipos.find(pos) == ipos.end())
                    ipos[pos] = std::vector<size_t>();
                ipos.at(pos).push_back(i);
            }
            return ipos;
        }

    public:
        Heading _heading;
        size_t _num_cells;
        int _group_threshold;
        int _cells_forward;
        int _cells_backward;
        int _min_speed;
        int _max_speed;
        float _prob_obey;
        float _prob_move;
        float _prob_change_speed;
        int _heading_change_duration;
        std::vector<float> _sum_weight;
        int _influence_alpha;
        Heading _heading_bias;

        // model member funcs
    protected:
        void _my_group();
        float _social_influence();

        std::vector<size_t> _my_group_idcs;
        Heading _next_heading;
        bool _heading_change;
        int _heading_change_count;
        int _position;
        int _speed;

        // discretization specific member funcs
    public:
        int _target_reset_threshold;
        int _history_reset;
        int _heading_failed_attempts;

    protected:
        float _approximate_angle(const Coord_t& head_p, const Coord_t& tail_p) const;
        int _approximate_discrete_pos(const Coord_t& head_p, const Coord_t& tail_p) const;
        void _create_deg_to_cell_map();
        int _to_cell(float deg) const;
        void _update_history();

        std::map<int, float> _deg2cell;

        CoordinatesConversionPtr _coordinatesConversion;
        const Coord_t MIN_XY;
        const Coord_t ARENA_CENTER;
        const double RADIUS;

        int _history_count;
        Heading _estimated_heading;
        int _direction_history;
        std::vector<int> _position_history;
    };

} // namespace Fishmodel

namespace samsar {
    namespace types {
        using namespace Fishmodel;

        namespace defaults {
            struct WeightFunc {
                WeightFunc(const std::vector<float> w) : _w(w) {}
                virtual ~WeightFunc() {}

                virtual float operator()(const Simulation& sim, const AgentBehavior_t& ff,
                    const AgentBehaviorStorage_t& f) const = 0;

                std::vector<float> _w;
            };
        } // namespace defaults

        class FishGroup {
        public:
            FishGroup(const std::vector<size_t>& idcs) : _idcs(idcs) {}

            bool has(int id) const;
            void clear();

            Heading sum_heading(const std::vector<AgentBehavior_t>& fish) const
            {
                int hdg = 0;
                for (size_t i : _idcs) {
                    auto social_behav = reinterpret_cast<SocialFishModel*>(fish[i].second);
                    hdg += social_behav->estimated_heading();
                }
                return to_heading(hdg);
            }

            Heading weighted_heading(const Simulation& sim, const AgentBehavior_t& focal_fish,
                const std::shared_ptr<defaults::WeightFunc>& weight_func) const
            {
                std::vector<AgentBehaviorStorage_t> fish = sim.agents;
                float sum = 0.0;
                for (size_t i : _idcs) {
                    if (fish[i].first.get() == focal_fish.first)
                        continue;
                    auto social_behav = reinterpret_cast<SocialFishModel*>(fish[i].second.get());
                    sum += (*weight_func)(sim, focal_fish, fish[i])
                        * social_behav->estimated_heading();
                }
                return to_heading(sum);
            }

            float average_speed(const Simulation& sim, const AgentBehavior_t& /*focal_fish*/) const
            {
                float avg = 0;
                for (const AgentBehaviorStorage_t& ab : sim.agents)
                    avg += ab.first->speed;
                return avg / sim.agents.size();
            }

            size_t size() const { return _idcs.size(); }
            std::vector<size_t>& idcs() { return _idcs; }
            std::vector<size_t> idcs() const { return _idcs; }

        private:
            std::vector<size_t> _idcs;
        };
    } // namespace types
} // namespace samsar

#endif // SOCIALFISHMODEL_HPP
