#ifndef BIOMIMETICFISHMODEL_HPP
#define BIOMIMETICFISHMODEL_HPP

#include <AgentState.hpp>
#include <CoordinatesConversion.hpp>

#include "model.hpp"
#include "utils/heading.hpp"

namespace Fishmodel {

    using namespace samsar;
    using namespace types;

    enum PositionDescription : int { UNDEFINED = 0, FRONT = 1, BACK = 2 };

    struct State {
        State() {}
        State(real_t v, Coord_t p, real_t h) : velocity(v), position(p), heading(h) {}

        real_t velocity;
        Coord_t position;
        real_t heading;

        PositionDescription position_description;
        real_t distance_to_focal;
    };

    class BiomimeticFishModel : public Behavior {
    public:
        BiomimeticFishModel(Simulation& simulation, Agent* agent = nullptr);

        void init();
        virtual void reinit() override;
        virtual void step() override;

        State state() const { return State(_agent->speed, _agent->headPos, _agent->direction); }

        real_t euclidean_distance(const Coord_t& p1, const Coord_t& p2) const;

        template <typename T> T to_360(T theta)
        {
            if (theta < 0)
                return 360 + theta;
            else
                return theta;
        }

    public: // model param getters & setters
        Heading heading() const { return _heading; }
        Heading& heading() { return _heading; }

        Coord_t position() const { return _position; }
        Coord_t& posittion() { return _position; }

    protected: // tunable params
        real_t _prob_follow;
        real_t _prob_turn;

        std::vector<real_t> _perception_range;
        std::vector<int> _perception_capability;

    protected:
        std::pair<std::vector<State>, std::vector<State>> _getPerceivedAgents();

        Heading _heading;
        Coord_t _position;

    private:
        std::vector<State> _trajectory;

        const Coord_t MIN_XY;
        const Coord_t ARENA_CENTER;
        const float RADIUS;
    };

} // namespace Fishmodel

#endif // SOCIALFISHMODEL_HPP
