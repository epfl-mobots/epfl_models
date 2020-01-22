#ifndef TOULOUSEMODEL_HPP
#define TOULOUSEMODEL_HPP

#include <Timer.hpp>
#include <FishBot.hpp>
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

namespace Fishmodel {

    using namespace simu;
    using namespace types;

    enum Order { DECREASING, INCREASING };

    class ToulouseModel;
    using state_t = std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>;
    using const_state_t = const std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>&;

    class ToulouseModel : public Behavior {
    public:
        ToulouseModel(Simulation& simulation, Agent* agent = nullptr);
        ~ToulouseModel();

        void init();
        virtual void reinit() override;
        virtual void step() override;

        std::tuple<int, QList<double>> getSpeedCommands() const;

        void logDataPrepare();
        void logDataFinish();
        void logDataWrite();

        virtual void move();
        void stimulate();
        void interact();

        std::tuple<double, double> model_stepper(double radius) const;
        void stepper();
        void free_will(const_state_t state, const std::tuple<double, double>& model_out, const std::vector<int>& idcs);

        Position<double> position() const;
        Position<double>& position();
        Speed<double> speed() const;
        Speed<double>& speed();
        double orientation() const;
        double& orientation();

        bool is_kicking() const;
        bool& is_kicking();
        bool is_gliding() const;
        bool& is_gliding();
        bool has_stepped() const;
        bool& has_stepped();
        bool to_be_optimized() const;
        bool& to_be_optimized();

        double angular_direction() const;
        double& angular_direction();
        double peak_velocity() const;
        double& peak_velocity();
        double kick_length() const;
        double& kick_length();
        double kick_duration() const;
        double& kick_duration();

        double time_kicker() const;

        double time() const;
        double& time();
        double timestep() const;
        double& timestep();
        double timestamp() const;
        double& timestamp();

        int id() const;
        int& id();

        FishBot* robot() const;
        FishBot*& robot();

        elastic_band::TrajectoryPtr referenceTrajectory() const;
        elastic_band::TrajectoryPtr& referenceTrajectory();
        elastic_band::TrajectoryPtr optimizedTrajectory() const;
        elastic_band::TrajectoryPtr& optimizedTrajectory();
        std::shared_ptr<std::vector<size_t>> fixedPoses() const;
        std::shared_ptr<std::vector<size_t>>& fixedPoses();

        QMainWindow* plotReferencePth() const;
        QMainWindow*& plotReferencePth();
        QMainWindow* plotOptimizedPth() const;
        QMainWindow*& plotOptimizedPth();
        QMainWindow* plotReferencePos() const;
        QMainWindow*& plotReferencePos();
        QMainWindow* plotOptimizedPos() const;
        QMainWindow*& plotOptimizedPos();
        QMainWindow* plotReferenceSpd() const;
        QMainWindow*& plotReferenceSpd();
        QMainWindow* plotOptimizedSpd() const;
        QMainWindow*& plotOptimizedSpd();
        QMainWindow* plotReferenceVel() const;
        QMainWindow*& plotReferenceVel();
        QMainWindow* plotOptimizedVel() const;
        QMainWindow*& plotOptimizedVel();
        QMainWindow* plotReferenceAcc() const;
        QMainWindow*& plotReferenceAcc();
        QMainWindow* plotOptimizedAcc() const;
        QMainWindow*& plotOptimizedAcc();

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
        void findNeighbors();
        void determineState(elastic_band::PoseSE2& pose, elastic_band::Velocity& velocity, elastic_band::Timestamp& timestamp);
        void planTrajectory(elastic_band::PoseSE2& pose, elastic_band::Velocity& velocity, elastic_band::Timestamp& timestamp);
        void initializePlanner();
        void optimizeTrajectory();
        void fetchTrajectory();
        void computePerformance();
        bool isTrajectoryOptimized();
        bool isTrajectoryFeasible();
        void visualizeReferenceTrajectory();
        void visualizeOptimizedTrajectory();

        state_t compute_state() const;
        std::vector<int> sort_neighbors(const Eigen::VectorXd& values, const int kicker_idx, Order order = Order::INCREASING) const;

        double wall_distance_interaction(double gamma_wall, double wall_interaction_range, double ag_radius, double radius) const;
        double wall_angle_interaction(double theta) const;

        double wall_distance_attractor(double distance, double radius) const;
        double wall_perception_attractor(double perception) const;
        double wall_angle_attractor(double phi) const;

        double alignment_distance_attractor(double distance, double radius) const;
        double alignment_perception_attractor(double perception) const;
        double alignment_angle_attractor(double phi) const;

        double angle_to_pipi(double difference) const;
        double round_time_resolution(double period_sec) const;

        Position<double> _desired_position;
        Speed<double> _desired_speed;
        Position<double> _position;
        Speed<double> _speed;
        double _orientation;

        bool _is_kicking;
        bool _is_gliding;
        bool _has_stepped;
        bool _to_be_optimized;

        double _angular_direction;
        double _peak_velocity;
        double _kick_length;
        double _kick_duration;

        Timer _timer;
        double _time;
        double _timestep;
        double _timestamp;

        Timer _timer_loop;
        Timer _timer_exec;
        Timer _timer_opti;
        double _time_loop;
        double _time_exec;
        double _time_opti;

        // Agent identifier assigned by the control mode
        int _id = -1;

        // Robot controlled by this behavior model
        FishBot* _robot = nullptr;

        // List of neighboring robots
        QList<ToulouseModel*> _neighbors;

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
        std::shared_ptr<std::vector<size_t>> _fixed_poses;
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

        // Log files and streams
        QFile _logFileTrajectory;
        QFile _logFileTracking;
        QFile _logFileKicking;
        QFile _logFileTiming;
        QTextStream _logStreamTrajectory;
        QTextStream _logStreamTracking;
        QTextStream _logStreamKicking;
        QTextStream _logStreamTiming;

        // Arena coordinates
        const Coord_t ARENA_CENTER;
    };

} // namespace Fishmodel

#endif // TOULOUSEMODEL_HPP
