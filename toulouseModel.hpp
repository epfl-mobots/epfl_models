#ifndef TOULOUSEMODEL_HPP
#define TOULOUSEMODEL_HPP

#include <AgentState.hpp>
#include <CoordinatesConversion.hpp>

#include "model.hpp"
#include "utils/heading.hpp"
#include "zones.hpp"

#include "types/types.hpp"
#include "utils/random/random_generator.hpp"

#include <elastic-band/TebConfig.hpp>
#include <elastic-band/TebPlanner.hpp>
#include <elastic-band/TebPlot.hpp>
#include <elastic-band/TebVisualization.hpp>
#include <elastic-band/RobotFootprintModel.hpp>
#include <elastic-band/Obstacles.hpp>
#include <elastic-band/Distances.hpp>
#include <elastic-band/kinematics/PoseSE2.hpp>
#include <elastic-band/kinematics/Velocity.hpp>
#include <elastic-band/kinematics/Acceleration.hpp>
#include <elastic-band/kinematics/Timestamp.hpp>
#include <elastic-band/kinematics/Trajectory.hpp>

#include <eigen3/Eigen/Core>

#include <map>
#include <mutex>

namespace Fishmodel {

    using namespace simu;
    using namespace types;

    enum Order { DECREASING, INCREASING };

    class ToulouseModel;
    using state_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>;
    using const_state_t
        = const std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>&;

    class ToulouseModel : public Behavior {
    public:
        ToulouseModel(Simulation& simulation, Agent* agent = nullptr);

        void init();
        virtual void reinit() override;
        virtual void step() override;

        QList<double> getSpeedCommands() const;

        void stimulate();
        void interact();
        virtual void move();

        void stepper();
        std::tuple<double, double> model_stepper(double radius) const;
        void free_will(const_state_t state, const std::tuple<double, double>& model_out,
            const std::vector<int>& idcs);

        Position<double> position() const;
        Position<double>& position();

        double time_kicker() const;

        double time() const;
        double& time();
        double angular_direction() const;
        double& angular_direction();
        double kick_duration() const;
        double& kick_duration();
        double kick_length() const;
        double& kick_length();
        double peak_velocity() const;
        double& peak_velocity();

        bool& is_kicking();
        bool is_kicking() const;

        int id() const;
        int& id();

        double radius;

        int perceived_agents;
        double gamma_rand;
        double gamma_wall;
        double wall_interaction_range;
        double body_length;

        double alpha;
        double tau0;
        double velocity_coef;
        double length_coef;
        double time_coef;

    protected:
        double wall_distance_interaction(double gamma_wall, double wall_interaction_range,
            double ag_radius, double radius) const;
        double wall_angle_interaction(double theta) const;

        double wall_distance_attractor(double distance, double radius) const;
        double wall_perception_attractor(double perception) const;
        double wall_angle_attractor(double phi) const;

        double alignment_distance_attractor(double distance, double radius) const;
        double alignment_perception_attractor(double perception) const;
        double alignment_angle_attractor(double phi) const;

        state_t compute_state() const;
        std::vector<int> sort_neighbours(const Eigen::VectorXd& values, const int kicker_idx,
            Order order = Order::INCREASING) const;

        double angle_to_pipi(double difference) const;

        // FishParams _fish_params;
        Position<double> _desired_position;
        Speed<double> _desired_speed;
        Position<double> _position;
        Speed<double> _speed;

        bool _is_kicking;

        double _angular_direction;
        double _peak_velocity;
        double _kick_length;
        double _kick_duration;
        double _current_time;
        double _time;
        int _id;

        // Timed Elastic Band
        elastic_band::TebConfig              _config;
        elastic_band::TebPlannerPtr          _planner;
        elastic_band::TebVisualizationPtr    _visualization;
        elastic_band::RobotFootprintModelPtr _robot_model;
        elastic_band::Point2dContainer       _robot_shape;
        elastic_band::ViaPointContainer      _viapoints;
        elastic_band::ObstacleContainer      _obstacles;
        elastic_band::TrajectoryPtr          _trajectory_ref;
        elastic_band::TrajectoryPtr          _trajectory_opt;
        QMainWindow* _plot_pth_ref = new QMainWindow();
        QMainWindow* _plot_pth_opt = new QMainWindow();
        QMainWindow* _plot_pos_ref = new QMainWindow();
        QMainWindow* _plot_pos_opt = new QMainWindow();
        QMainWindow* _plot_spd_ref = new QMainWindow();
        QMainWindow* _plot_spd_opt = new QMainWindow();
        QMainWindow* _plot_vel_ref = new QMainWindow();
        QMainWindow* _plot_vel_opt = new QMainWindow();
        QMainWindow* _plot_acc_ref = new QMainWindow();
        QMainWindow* _plot_acc_opt = new QMainWindow();

        // std::mutex _mtx;
        // std::vector<Coord_t> _trajectory;

        CoordinatesConversionPtr _coordinatesConversion;
        const Coord_t ARENA_CENTER;
    };

} // namespace Fishmodel

#endif // TOULOUSEMODEL_HPP
