#include "toulouseModel.hpp"

#include <AgentData.hpp>
#include <SetupType.hpp>
#include <settings/CalibrationSettings.hpp>
#include <settings/CommandLineParameters.hpp>
#include <settings/RobotControlSettings.hpp>

#include <QApplication>
#include <QDebug>
#include <QQueue>

#include <eigen3/Eigen/Core>

#include <boost/range.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>

#include <memory>
#include <cmath>
#include <algorithm>

#define TIMESTEP 0.020 // [s]

namespace Fishmodel {

    ToulouseModel::ToulouseModel(Simulation& simulation, Agent* agent)
        : Behavior(simulation, agent),
          ARENA_CENTER({0.075, 0.110})
    {
        init();
    }

    ToulouseModel::~ToulouseModel()
    {
        logDataFinish();
    }

    void ToulouseModel::init()
    {
        reinit();
    }

    void ToulouseModel::reinit()
    {
        if (_id < 0)
            return;

        if (perceived_agents >= _simulation.agents.size()) {
            qDebug() << "Correcting the number of perceived individuals to N-1";
            perceived_agents = _simulation.agents.size() - 1;
        }

        _position.x = -(_id - 1. - _simulation.agents.size() / 2) * body_length;
        _position.y = -0.1;
        _orientation = _id * 2. * M_PI / (_simulation.agents.size() + 1);
        _angular_direction = angle_to_pipi(_orientation);

        stepper();

        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end())
            return;

        _timer.clear();
        _time = 0;
        _timestep = 0;
        _timestamp = 0;

        _timer_loop.clear();
        _timer_exec.clear();
        _timer_opti.clear();
        _time_loop = 0;
        _time_exec = 0;
        _time_opti = 0;

        _is_kicking = false;
        _has_stepped = false;
        _to_be_optimized = true;

        _config.trajectory.teb_autosize = false;
        _config.trajectory.dt_ref = 0.3;
        _config.trajectory.dt_hysteresis = 0.1;
        _config.trajectory.min_samples = 3;
        _config.trajectory.max_samples = 5000;
        _config.trajectory.global_plan_overwrite_orientation = false;
        _config.trajectory.allow_init_with_backwards_motion = false;
        _config.trajectory.global_plan_viapoint_sep = -1;
        _config.trajectory.via_points_ordered = false;
        _config.trajectory.max_global_plan_lookahead_dist = 1;
        _config.trajectory.global_plan_prune_distance = 1;
        _config.trajectory.exact_arc_length = false;
        _config.trajectory.force_reinit_new_goal_dist = 0;
        _config.trajectory.feasibility_check_no_poses = 5;
        _config.trajectory.publish_feedback = false;
        _config.trajectory.min_resolution_collision_check_angular = M_PI;
        _config.trajectory.control_look_ahead_poses = 1;
        _config.robot.max_vel_x = 0.3;
        _config.robot.max_vel_x_backwards = 0.0;
        _config.robot.max_vel_y = 0.0;
        _config.robot.max_vel_theta = 18;
        _config.robot.acc_lim_x = 1.3;
        _config.robot.acc_lim_y = 0.0;
        _config.robot.acc_lim_theta = 200;
        _config.robot.min_turning_radius = 0;
        _config.robot.wheelbase = 1.0;
        _config.robot.cmd_angle_instead_rotvel = false;
        _config.robot.is_footprint_dynamic = false;
        _config.goal_tolerance.xy_goal_tolerance = 0.2;
        _config.goal_tolerance.yaw_goal_tolerance = 0.2;
        _config.goal_tolerance.free_goal_vel = false;
        _config.goal_tolerance.complete_global_plan = true;
        _config.neighbors.association_dist = 0.25;
        _config.neighbors.min_neighbor_dist = 0.010;
        _config.obstacles.min_obstacle_dist = 0.001;
        _config.obstacles.inflation_dist = 0.010;
        _config.obstacles.influence_dist = 0.010;
        _config.obstacles.dynamic_obstacle_inflation_dist = 0.0;
        _config.obstacles.include_dynamic_obstacles = true;
        _config.obstacles.include_costmap_obstacles = true;
        _config.obstacles.costmap_obstacles_behind_robot_dist = 1.5;
        _config.obstacles.obstacle_poses_affected = 1000;
        _config.obstacles.legacy_obstacle_association = false;
        _config.obstacles.obstacle_association_force_inclusion_factor = 1000;
        _config.obstacles.obstacle_association_cutoff_factor = 1000000;
        _config.obstacles.costmap_converter_plugin = "";
        _config.obstacles.costmap_converter_spin_thread = true;
        _config.obstacles.costmap_converter_rate = 5;
        _config.optim.no_inner_iterations = 10;
        _config.optim.no_outer_iterations = 5;
        _config.optim.stop_below_significant_error_chi2 = 0.01;
        _config.optim.stop_below_percentage_improvement = 0.01;//0.00001;
        _config.optim.stop_after_elapsed_time_microsecs = 250000;
        _config.optim.optimization_activate = true;
        _config.optim.optimization_verbose = false;
        _config.optim.single_dynamics_edge = true;
        _config.optim.set_orientate_action = false;
        _config.optim.save_optimized_graph = false;
        _config.optim.file_optimized_graph = "/home/fishbot/Documents/CATS2/Toulouse/g2o_graphs/test.g2o";
        _config.optim.penalty_epsilon = 0.001;
        _config.optim.weight_max_vel_x = 1000000000000;
        _config.optim.weight_max_vel_y = 1000000000000;
        _config.optim.weight_max_vel_theta = 1000000000000;
        _config.optim.weight_acc_lim_x = 1000000000000;
        _config.optim.weight_acc_lim_y = 1000000000000;
        _config.optim.weight_acc_lim_theta = 1000000000000;
        _config.optim.weight_kinematics_nh = 100000000;
        _config.optim.weight_kinematics_forward_drive = 0;
        _config.optim.weight_kinematics_turning_radius = 0;
        _config.optim.weight_optimaltime = 0;
        _config.optim.weight_shortest_path = 0;
        _config.optim.weight_profile_fidelity_v = 10000;
        _config.optim.weight_profile_fidelity_w = 1;
        _config.optim.weight_profile_fidelity_t = 0;
        _config.optim.weight_neighbor = 50;
        _config.optim.weight_obstacle = 0;//1000000000000;
        _config.optim.weight_obstacle_inflation = 0;//1000000;
        _config.optim.weight_obstacle_influence = 25;//1000000;
        _config.optim.weight_dynamic_obstacle = 50;
        _config.optim.weight_dynamic_obstacle_inflation = 0.1;
        _config.optim.weight_viapoint = 1;
        _config.optim.weight_prefer_rotdir = 0;
        _config.optim.weight_adapt_factor = 1;
        _config.optim.obstacle_cost_exponent = 1;
        _config.hcp.enable_homotopy_class_planning = false;
        _config.hcp.enable_multithreading = true;
        _config.hcp.simple_exploration = false;
        _config.hcp.max_number_classes = 5;
        _config.hcp.selection_cost_hysteresis = 1.0;
        _config.hcp.selection_prefer_initial_plan = 0.95;
        _config.hcp.selection_obst_cost_scale = 100.0;
        _config.hcp.selection_viapoint_cost_scale = 1.0;
        _config.hcp.selection_alternative_time_cost = false;
        _config.hcp.obstacle_keypoint_offset = 0.1;
        _config.hcp.obstacle_heading_threshold = 0.45;
        _config.hcp.roadmap_graph_no_samples = 15;
        _config.hcp.roadmap_graph_area_width = 6;
        _config.hcp.roadmap_graph_area_length_scale = 1.0;
        _config.hcp.h_signature_prescaler = 1;
        _config.hcp.h_signature_threshold = 0.1;
        _config.hcp.switching_blocking_period = 0.0;
        _config.hcp.viapoints_all_candidates = true;
        _config.hcp.visualize_hc_graph = false;
        _config.hcp.visualize_with_time_as_z_axis_scale = 0.0;
        _config.hcp.delete_detours_backwards = true;
        _config.hcp.detours_orientation_tolerance = M_PI / 2.0;
        _config.hcp.length_start_orientation_vector = 0.4;
        _config.hcp.max_ratio_detours_duration_best_duration = 3.0;
        _config.recovery.shrink_horizon_backup = true;
        _config.recovery.shrink_horizon_min_duration = 10;
        _config.recovery.oscillation_recovery = true;
        _config.recovery.oscillation_v_eps = 0.1;
        _config.recovery.oscillation_omega_eps = 0.1;
        _config.recovery.oscillation_recovery_min_duration = 10;
        _config.recovery.oscillation_filter_duration = 10;
        _fixed_poses.reset();
        _viapoints.clear();
        _obstacles.clear();
        _robot_shape.clear();
        //_robot_shape.push_back(Eigen::Vector2d(-0.044/2, -0.022/2));
        //_robot_shape.push_back(Eigen::Vector2d(+0.044/2, -0.022/2));
        //_robot_shape.push_back(Eigen::Vector2d(+0.044/2, +0.022/2));
        //_robot_shape.push_back(Eigen::Vector2d(-0.044/2, +0.022/2));
        //_robot_model    = elastic_band::RobotFootprintModelPtr(new elastic_band::PolygonRobotFootprint(_robot_shape));
        _robot_shape.push_back(Eigen::Vector2d(+0.055/2/2, 0.033/2)); // (front offset, front radius)
        _robot_shape.push_back(Eigen::Vector2d(-0.055/2/2, 0.033/2)); // (rear  offset, rear  radius)
        _robot_model    = elastic_band::RobotFootprintModelPtr(new elastic_band::TwoCirclesRobotFootprint(_robot_shape.front().coeff(0), _robot_shape.front().coeff(1), _robot_shape.back().coeff(0), _robot_shape.back().coeff(1)));
        _planner        = elastic_band::TebPlannerPtr(new elastic_band::TebPlanner());
        _trajectory_ref = elastic_band::TrajectoryPtr(new elastic_band::Trajectory());
        _trajectory_opt = elastic_band::TrajectoryPtr(new elastic_band::Trajectory());

        logDataPrepare();
    }

    void ToulouseModel::step()
    {
        // Stop the process if the agent has already stepped
        if (_has_stepped)
            return;

        // Stop the process if the agent is not a robot
        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end())
            return;

        // Store the control loop time and
        // reset the control loop timer and
        // start the execution timer and
        // clear the optimization timer
        _time_loop = _timer_loop.runTimeSec();
        if (_timer_loop.isSet()) // Reduce the communication overload
            while (_timer_loop.runTimeSec() < 0.050);
        _timer_loop.reset();
        _timer_exec.reset();
        _timer_opti.clear();
        _time_exec = 0;
        _time_opti = 0;

        // Update the robot pose and speed as tracked by the camera and
        // translate the position with respect to the arena center
        _position.x = _agent->headPos.first  - ARENA_CENTER.first;
        _position.y = _agent->headPos.second - ARENA_CENTER.second;
        _orientation = angle_to_pipi(_agent->direction);
        _speed.vx = 0;
        _speed.vy = 0;
        if (_robot != nullptr) {
            if (_robot->state().position().isValid()) {
                _position.x = _robot->state().position().x() - ARENA_CENTER.first;
                _position.y = _robot->state().position().y() - ARENA_CENTER.second;
            }
            //if (_robot->state().orientation().isValid()) // FIXME: should be always valid
                _orientation = angle_to_pipi(_robot->state().orientation().angleRad());
            if (_robot->isValidSpeed()) {
                _speed.vx = _robot->speed() * std::cos(_orientation);
                _speed.vy = _robot->speed() * std::sin(_orientation);
            }
        }
        //qDebug() << "robot" << _position.x << _position.y << _speed.vx << _speed.vy << _orientation;

        // Determine if the robot should trigger a new kick
        _is_kicking = !_timer.isSet() || _timer.isTimedOutSec(_kick_duration);

        // Update the current time information
        if (_timer.isSet()) {
            const double runtime = round_time_resolution(_timer.runTimeSec());
            _timestamp = runtime + _timestamp;
            _timestep = runtime - _time;
            _time = runtime;
        } else {
            _timer.reset();
            _time = 0;
            _timestep = 0;
            _timestamp = 0;
        }
        //qDebug() << "current model time" << "=" << _time << "s" << "&" << "elapsed time" << "=" << _timestamp << "s";

        // Perform the kick if required
        if (_is_kicking) {
            _timer.reset();
            _time = 0;
            //qDebug() << "reset model time" << "=" << _time << "s";

            // Apply attractors/repulsors and update the fish intuitions
            // (for the kicker, the other fish are in their gliding phase)
            interact();

            // The individuals decide on the desired position
            stimulate(); // Kicking individual goes first
        }

        // Find neighboring robots
        findNeighbors();

        // Determine current pose and velocity
        elastic_band::PoseSE2 pose;
        elastic_band::Velocity velocity;
        elastic_band::Timestamp timestamp;
        determineState(pose, velocity, timestamp);

        // Compute reference trajectory
        planTrajectory(pose, velocity, timestamp);
        //std::cout << *_trajectory_ref << std::endl;

        // Visualize reference trajectory
        //visualizeReferenceTrajectory();

        // Initialize environment
        initializePlanner();

        // Plan subsequent trajectory
        optimizeTrajectory();

        if (isTrajectoryOptimized()) {
            if (isTrajectoryFeasible()) {
                // Store optimized trajectory
                fetchTrajectory();
                //std::cout << *_trajectory_opt << std::endl;

                // Compute optimization performance
                //computePerformance();

                // Visualize optimized trajectory
                //visualizeOptimizedTrajectory();

                // Send control commands
                // (not to be done here)
            } else if (_to_be_optimized) {
                qDebug() << "The TEB planner was not able to find a feasible solution for the trajectory.";
            }
        } else if (_to_be_optimized) {
            qDebug() << "The TEB planner was not able to find an optimized solution for the trajectory.";
        }

        if (_to_be_optimized) {
            // Update position and velocity information
            // (actual move step)
            move();

            // Measure execution time
            _time_exec = _timer_exec.runTimeSec();

            // Save results
            logDataWrite();
        }

        // Update status flag
        _has_stepped = true;
    }

    void ToulouseModel::findNeighbors()
    {
        _neighbors.clear();
        const bool multi_agent_optimization = true;
        if (!_to_be_optimized || !multi_agent_optimization || _robot == nullptr)
            return;
        const double neighboring_radius = 2 * radius;
        for (AgentDataWorld robot : _robot->otherRobotsData()) {
            if (robot.type() != AgentType::CASU){
                continue;}
            if (/*_robot->state().position().closeTo(robot.state().position(), neighboring_radius)*/true) {
                QString id = robot.id();
                ToulouseModel* behavior = nullptr;
                for (std::vector<AgentBehavior_t>::const_iterator rob = _simulation.robots.begin(); rob != _simulation.robots.end(); ++rob) {
                    ToulouseModel* rbt = reinterpret_cast<ToulouseModel*>(rob->second);
                    if (rbt && rbt->robot() && rbt->robot()->id() == id) {
                        behavior = rbt;
                        break;
                    }
                }
                if (behavior == nullptr || behavior->has_stepped())
                    continue;
                _neighbors.append(behavior);
            }
        }
    }

    void ToulouseModel::determineState(elastic_band::PoseSE2& pose, elastic_band::Velocity& velocity, elastic_band::Timestamp& timestamp)
    {
        /*int idx = -1;
        const double accuracy = 0.002;
        double distance = std::numeric_limits<double>::max();
        if (!_trajectory_opt->trajectory().empty()) {
            // Set last velocity equal to penultimate velocity if it exists
            if (_trajectory_opt->trajectory().size() > 1)
                _trajectory_opt->trajectory().back()->velocity() = _trajectory_opt->trajectory().at(_trajectory_opt->trajectory().size()-2)->velocity();
            // Find the trajectory index corresponding to the current position
            const int window = 30;
            const int target = _timestep / TIMESTEP;
            const int length = static_cast<int>(_trajectory_opt->trajectory().size()) - 1;
            const int timestep = std::min(target, length);
            const int limit_ahead  = std::min(timestep + window / 2, length);
            const int limit_behind = std::max(timestep - window / 2, 0);
            Eigen::Vector2d position = Eigen::Vector2d(_position.x, _position.y);
            for (int i = limit_ahead; i >= limit_behind; --i) {
                const double dist = (_trajectory_opt->trajectory().at(i)->pose().position() - position).norm();
                if (dist < distance) {
                    distance = dist;
                    idx = i;
                }
                if (distance <= accuracy)
                    break;
            }
        }
        if (idx >= 0 && distance <= accuracy) {
            pose      = _trajectory_opt->trajectory().at(idx)->pose();
            velocity  = _trajectory_opt->trajectory().at(idx)->velocity();
            timestamp = _is_kicking == false
                      ? _trajectory_opt->trajectory().at(idx)->timestamp()
                      : elastic_band::Timestamp(elastic_band::timestamp_t(0));
        } else if (idx >= 0) {
            pose      = elastic_band::PoseSE2(_position.x, _position.y, _orientation);
            velocity  = elastic_band::Velocity(Eigen::Vector2d(_speed.vx, _speed.vy).norm(), 0, _orientation);
            timestamp = elastic_band::Timestamp(elastic_band::timestamp_t(_time));
        } else {
            pose      = elastic_band::PoseSE2(_position.x, _position.y, _orientation);
            velocity  = elastic_band::Velocity(0, 0, _orientation);
            timestamp = elastic_band::Timestamp(elastic_band::timestamp_t(0));
        }*/
        if (!_trajectory_opt->trajectory().empty() && _time > _trajectory_opt->trajectory().back()->timestamp().count())
            _time = _trajectory_opt->trajectory().back()->timestamp().count();
        pose      = elastic_band::PoseSE2(_position.x, _position.y, _orientation);
        velocity  = elastic_band::Velocity(_robot->speed(), 0, _orientation);
        timestamp = elastic_band::Timestamp(elastic_band::timestamp_t(_time));
    }

    void ToulouseModel::planTrajectory(elastic_band::PoseSE2& pose, elastic_band::Velocity& velocity, elastic_band::Timestamp& timestamp)
    {
        const double timestep = TIMESTEP; // [s]
        const double horizon_control_loop = (15 + 1) * timestep; // [s]
        const double horizon_optimization = (15 + 1) * timestep; // [s]
        const double horizon_modelization = _kick_duration/* - timestamp.count()*/; // [s]
        const double horizon = true ? horizon_optimization : std::max(std::min(horizon_modelization, horizon_optimization), horizon_control_loop); // [s]
        const size_t nb_commands = static_cast<size_t>(std::max(std::floor(horizon / timestep), 1.)); // [#]

        elastic_band::TimestepPtr       timestep_ptr(new elastic_band::Timestep(elastic_band::timestep_t(timestep)));
        elastic_band::TimestepContainer timestep_profile(nb_commands - 1, timestep_ptr);
        elastic_band:: PoseSE2Container     pose_profile(nb_commands);

        const double acceleration = _config.robot.acc_lim_x - _config.optim.penalty_epsilon; // [m/s^2]

        static size_t burst_idx = 0;
        static size_t coast_idx = 0;

        const size_t glide_idx = _is_gliding ? 0 : coast_idx;
        const size_t glide_p = glide_idx + 1;
        const size_t glide_v = glide_idx;
        const size_t glide_t = glide_idx;

        _is_gliding = (_is_gliding
                    || (!_is_kicking
                     && !_trajectory_opt->trajectory().empty()
                     && _trajectory_opt->trajectory().front()->timestamp().count() == 0))
                   && _trajectory_opt != nullptr
                   && _trajectory_opt->trajectory().size() > glide_idx
                   && timestamp.count() < _trajectory_opt->trajectory().at(glide_idx)->timestamp().count()
                   && _trajectory_opt->trajectory().back()->timestamp().count() > _trajectory_opt->trajectory().at(glide_idx)->timestamp().count() + timestep;

        const Eigen::Vector2d position_init = _is_gliding ? _trajectory_opt->trajectory().at(glide_p)->pose().position()        : pose.position();        // [m]
        const double       orientation_init = _is_gliding ? _trajectory_opt->trajectory().at(glide_p)->pose().orientation()     : pose.orientation();     // [rad]
        const double       translation_init = _is_gliding ? _trajectory_opt->trajectory().at(glide_v)->velocity().translation() : velocity.translation(); // [m/s]
        const double          rotation_init = _is_gliding ? _trajectory_opt->trajectory().at(glide_v)->velocity().rotation()    : velocity.rotation();    // [rad/s]
        const double          duration_init = _is_gliding ? _trajectory_opt->trajectory().at(glide_t)->timestamp().count()      : timestamp.count();      // [s]

        static double duration = duration_init;
        static double duration_phase1 = duration;
        static double duration_phase2 = duration;
        static double duration_phase3 = duration;
        static double translation_peak = translation_init;
        double translation = translation_init;
        double rotation = rotation_init;
        Eigen::Vector2d position = position_init;
        Eigen::Vector2d position_phase1 = position;
        Eigen::Vector2d position_phase2 = position;
        Eigen::Vector2d position_phase3 = position;
        double theta = orientation_init;
        double x = position.x();
        double y = position.y();

        const double Kp = 50.0;
        const double Ki = 1.00;
        const double Kd = 0.05;
        double error_old = 0;
        double error_new = 0;
        double error_dif = 0;
        double error_sum = 0;
        QQueue<double> errors;

        if (_is_kicking || (duration_phase2 == 0 && duration_phase3 == 0)) { // Restart piecewise swimming pattern
            burst_idx = 0;
            coast_idx = 0;
            duration_phase1 = 0;
            duration_phase2 = 0;
            duration_phase3 = 0;
            timestamp = elastic_band::Timestamp(elastic_band::timestamp_t(0));
        }
        if (_is_gliding) { // Assume that rotation has been completed during last optimization and pose at beginning of coast is accurate
            timestamp = elastic_band::Timestamp(elastic_band::timestamp_t(duration_init));
            duration_phase1 = 0;
            duration_phase2 = duration_init;
            duration_phase3 = 0;
            translation_peak = translation_init;
            pose_profile.front() = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(_trajectory_opt->trajectory().at(glide_idx)->pose()));
        }

        // Implement burst-and-coast swimming
        pose_profile.at(_is_gliding ? 1 : 0) = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(x, y, theta));
        for (size_t i = _is_gliding ? 2 : 1; i < pose_profile.size(); ++i) {
            duration = i * timestep + duration_init;
            if (duration_phase2 == 0 && duration_phase3 == 0 && std::abs(angle_to_pipi(_angular_direction - theta)) > 1e-2) { // 1. Orientation
                duration_phase1 = duration;
                burst_idx = i - 1;
                coast_idx = i - 1;
                error_new = angle_to_pipi(_angular_direction - theta);
                error_dif = 0;
                error_sum = 0;
                if (errors.size() > 0)
                    error_dif = error_new - error_old;
                if (rotation > -_config.robot.max_vel_theta && rotation < _config.robot.max_vel_theta) {
                    errors.enqueue(error_new);
                    for (double error : errors)
                        error_sum += error;
                }
                rotation = duration_phase1 > timestep + duration_init ? Kp * error_new + Ki * error_sum * timestep + Kd * error_dif / timestep : rotation_init;
                if (rotation < -_config.robot.max_vel_theta)
                    rotation = -_config.robot.max_vel_theta;
                if (rotation > +_config.robot.max_vel_theta)
                    rotation = +_config.robot.max_vel_theta;
                error_old = error_new;
                double dtheta = 0;
                if (std::abs(rotation) > 1e-6) {
                    dtheta = rotation * timestep;
                    Eigen::Vector2d r = Eigen::Vector2d(std::sin(theta), -std::cos(theta)) * translation / rotation;
                    position.x() = (r.x()*std::cos(dtheta) - r.y()*std::sin(dtheta)) - r.x() + position_phase1.x();
                    position.y() = (r.x()*std::sin(dtheta) + r.y()*std::cos(dtheta)) - r.y() + position_phase1.y();
                } else {
                    position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * translation * timestep + position_phase1;
                }
                theta = angle_to_pipi(theta + dtheta);
                position_phase1 = position;
                position_phase2 = position_phase1;
                position_phase3 = position_phase2;
            } else if (duration_phase3 == 0 && acceleration * (duration - duration_phase1) + translation_init < _peak_velocity) { // 2. Acceleration
                duration_phase2 = duration - duration_phase1;
                coast_idx = i - 1;
                translation_peak = acceleration * duration_phase2 + translation_init;
                translation = translation_peak;
                position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * (acceleration * duration_phase2 * duration_phase2 / 2 + translation_init * duration_phase2) + position_phase1;
                position_phase2 = position;
                position_phase3 = position_phase2;
            } else if (duration - duration_phase2 - duration_phase1 <= _kick_duration) { // 3. Relaxation
                duration_phase3 = duration - duration_phase2 - duration_phase1;
                translation = std::min(translation_peak * std::exp(-duration_phase3 / tau0), translation);
                position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * translation * timestep + position_phase3;
                position_phase3 = position;
            } else {
                position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * translation * timestep + position;
            }
            x = position.x();
            y = position.y();
            pose_profile.at(i) = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(x, y, theta));
        }

        // Mark key trajectory poses as fixed
        _fixed_poses.reset();
        _fixed_poses = std::make_shared<std::vector<size_t>>();
        if (pose_profile.size() > 1)
            _fixed_poses->push_back(1); // Force velocity continuity
        //if (burst_idx > 0)
        //    _fixed_poses->push_back(burst_idx); // Force proper direction

        // Construct full reference trajectory profiles
        _trajectory_ref->robotParameters().wheel_radius = 0.005; // [m]
        _trajectory_ref->robotParameters().wheel_distance = 0.018; // [m]
        _trajectory_ref->setProfileTimestep(timestep_profile, timestamp, false);
        _trajectory_ref->setProfilePose(pose_profile);
    }

    void ToulouseModel::initializePlanner()
    {
        _viapoints.clear();
        _obstacles.clear();
        _obstacles.push_back(elastic_band::ObstaclePtr(new elastic_band::AnnularObstacle(0, 0, radius)));

        if (_to_be_optimized)
            _planner->initialize(_config, &_obstacles, _robot_model, _visualization, &_viapoints);

        if (_trajectory_ref->trajectory().size() > 0)
            _planner->setVelocityStart(_trajectory_ref->trajectory().front()->velocity(), false);
        else
            _planner->setVelocityStart(elastic_band::Velocity(), false);

        if (_trajectory_ref->trajectory().size() > 1)
            _planner->setVelocityGoal(_trajectory_ref->trajectory().at(_trajectory_ref->trajectory().size()-2)->velocity(), false);
        else
            _planner->setVelocityGoal(elastic_band::Velocity(), false);
    }

    void ToulouseModel::optimizeTrajectory()
    {
        if (!_to_be_optimized)
            return;
        if (_neighbors.isEmpty()) {
            _timer_opti.reset();
            _planner->plan(*_trajectory_ref, _fixed_poses, true, false, false);
            _time_opti = _timer_opti.runTimeSec();
        } else {
            std::vector<elastic_band::TrajectoryPtr> trajectories;
            std::vector<std::shared_ptr<std::vector<size_t>>> indices;
            trajectories.push_back(_trajectory_ref);
            indices.push_back(_fixed_poses);
            for (ToulouseModel* robot : _neighbors) {
                robot->to_be_optimized() = false;
                robot->step();
                trajectories.push_back(robot->referenceTrajectory());
                indices.push_back(robot->fixedPoses());
            }
            _timer_opti.reset();
            _planner->plan(trajectories, indices, true, false, false);
            _time_opti = _timer_opti.runTimeSec();
        }
    }

    void ToulouseModel::fetchTrajectory()
    {
        if (!_to_be_optimized)
            return;
        _trajectory_opt->robotParameters() = _trajectory_ref->robotParameters();
        if (_neighbors.isEmpty()) {
            _planner->getFullTrajectory(*_trajectory_opt, ! _trajectory_ref->trajectory().empty()
                                                          ? _trajectory_ref->trajectory().front()->timestamp()
                                                          : elastic_band::Timestamp::zero());
        } else {
            std::vector<elastic_band::TrajectoryPtr> trajectories;
            std::vector<elastic_band::Timestamp> timestamps;
            trajectories.push_back(_trajectory_opt);
            timestamps.push_back(! _trajectory_ref->trajectory().empty()
                                 ? _trajectory_ref->trajectory().front()->timestamp()
                                 : elastic_band::Timestamp::zero());
            for (ToulouseModel* robot : _neighbors) {
                robot->optimizedTrajectory()->robotParameters() = robot->referenceTrajectory()->robotParameters();
                trajectories.push_back(robot->optimizedTrajectory());
                timestamps.push_back(! robot->referenceTrajectory()->trajectory().empty()
                                     ? robot->referenceTrajectory()->trajectory().front()->timestamp()
                                     : elastic_band::Timestamp::zero());
            }
            _planner->getFullTrajectory(trajectories, timestamps);
            for (ToulouseModel* robot : _neighbors) {
                robot->move();
                robot->logDataWrite();
            }
        }
    }

    void ToulouseModel::computePerformance()
    {
        if (!_to_be_optimized)
            return;
        _planner->computeCurrentCost(&(*_trajectory_ref));
        const double optim_cost = _planner->getCurrentCost();
        qDebug() << "Cost of the last optimization process =" << optim_cost;
    }

    bool ToulouseModel::isTrajectoryOptimized()
    {
        if (!_to_be_optimized)
            return false;
        const bool optimized = _planner->isOptimized();
        return optimized;
    }

    bool ToulouseModel::isTrajectoryFeasible()
    {
        if (!_to_be_optimized)
            return false;
        const bool feasible = true; // TODO: check trajectory feasibility
        return feasible;
    }

    void ToulouseModel::visualizeReferenceTrajectory()
    {
        if (!_to_be_optimized)
            return;
        const QString name = QString(" (robot %1)");
        std::vector<elastic_band::TrajectoryPtr> trajectories;
        std::vector<QMainWindow*> plots_pth;
        std::vector<QMainWindow*> plots_pos;
        std::vector<QMainWindow*> plots_spd;
        std::vector<QMainWindow*> plots_vel;
        std::vector<QMainWindow*> plots_acc;
        std::vector<QString> names;
        trajectories.push_back(_trajectory_ref);
        plots_pth.push_back(_plot_pth_ref);
        plots_pos.push_back(_plot_pos_ref);
        plots_spd.push_back(_plot_spd_ref);
        plots_vel.push_back(_plot_vel_ref);
        plots_acc.push_back(_plot_acc_ref);
        if (_neighbors.isEmpty()) {
            names.push_back(QString());
        } else {
            names.push_back(name.arg(_id));
            for (ToulouseModel* robot : _neighbors) {
                trajectories.push_back(robot->referenceTrajectory());
                plots_pth.push_back(robot->plotReferencePth());
                plots_pos.push_back(robot->plotReferencePos());
                plots_spd.push_back(robot->plotReferenceSpd());
                plots_vel.push_back(robot->plotReferenceVel());
                plots_acc.push_back(robot->plotReferenceAcc());
                names.push_back(name.arg(robot->id()));
            }
        }
        for (size_t i = 0; i < trajectories.size(); ++i) {
            _plot_pth_ref = elastic_band::TebPlot::plotPath               (&(*trajectories.at(i)), plots_pth.at(i), "Reference path"         + names.at(i));
            _plot_pos_ref = elastic_band::TebPlot::plotProfilePose        (&(*trajectories.at(i)), plots_pos.at(i), "Reference pose"         + names.at(i));
            _plot_spd_ref = elastic_band::TebPlot::plotProfileSpeed       (&(*trajectories.at(i)), plots_spd.at(i), "Reference speed"        + names.at(i));
            _plot_vel_ref = elastic_band::TebPlot::plotProfileVelocity    (&(*trajectories.at(i)), plots_vel.at(i), "Reference velocity"     + names.at(i));
            _plot_acc_ref = elastic_band::TebPlot::plotProfileAcceleration(&(*trajectories.at(i)), plots_acc.at(i), "Reference acceleration" + names.at(i));
        }
    }

    void ToulouseModel::visualizeOptimizedTrajectory()
    {
        if (!_to_be_optimized)
            return;
        const QString name = QString(" (robot %1)");
        std::vector<elastic_band::TrajectoryPtr> trajectories;
        std::vector<QMainWindow*> plots_pth;
        std::vector<QMainWindow*> plots_pos;
        std::vector<QMainWindow*> plots_spd;
        std::vector<QMainWindow*> plots_vel;
        std::vector<QMainWindow*> plots_acc;
        std::vector<QString> names;
        trajectories.push_back(_trajectory_opt);
        plots_pth.push_back(_plot_pth_opt);
        plots_pos.push_back(_plot_pos_opt);
        plots_spd.push_back(_plot_spd_opt);
        plots_vel.push_back(_plot_vel_opt);
        plots_acc.push_back(_plot_acc_opt);
        if (_neighbors.isEmpty()) {
            names.push_back(QString());
        } else {
            names.push_back(name.arg(_id));
            for (ToulouseModel* robot : _neighbors) {
                trajectories.push_back(robot->optimizedTrajectory());
                plots_pth.push_back(robot->plotOptimizedPth());
                plots_pos.push_back(robot->plotOptimizedPos());
                plots_spd.push_back(robot->plotOptimizedSpd());
                plots_vel.push_back(robot->plotOptimizedVel());
                plots_acc.push_back(robot->plotOptimizedAcc());
                names.push_back(name.arg(robot->id()));
            }
        }
        for (size_t i = 0; i < trajectories.size(); ++i) {
            _plot_pth_ref = elastic_band::TebPlot::plotPath               (&(*trajectories.at(i)), plots_pth.at(i), "Optimized path"         + names.at(i));
            _plot_pos_ref = elastic_band::TebPlot::plotProfilePose        (&(*trajectories.at(i)), plots_pos.at(i), "Optimized pose"         + names.at(i));
            _plot_spd_ref = elastic_band::TebPlot::plotProfileSpeed       (&(*trajectories.at(i)), plots_spd.at(i), "Optimized speed"        + names.at(i));
            _plot_vel_ref = elastic_band::TebPlot::plotProfileVelocity    (&(*trajectories.at(i)), plots_vel.at(i), "Optimized velocity"     + names.at(i));
            _plot_acc_ref = elastic_band::TebPlot::plotProfileAcceleration(&(*trajectories.at(i)), plots_acc.at(i), "Optimized acceleration" + names.at(i));
        }
    }

    std::tuple<int, QList<double>> ToulouseModel::getSpeedCommands() const
    {
        QList<double> speeds;
        elastic_band::VelocityContainer velocity_profile;

        _trajectory_opt->getProfileVelocity(velocity_profile);
        const size_t horizon = std::min(velocity_profile.size(), static_cast<size_t>(15));
        speeds.reserve(static_cast<int>(2 * horizon));

        for (size_t i = 0; i < horizon; ++i) {
            // Convert speeds from [rad/s] to [cm/s]
            velocity_profile.at(i)->wheel() *= velocity_profile.at(i)->getRadius() * 100.;
            // Interchange left and right wheel speeds due to hardware
            speeds.append(velocity_profile.at(i)->wheel()[1]);
            speeds.append(velocity_profile.at(i)->wheel()[0]);
        }

        if (speeds.size() > 1) {
            // Force constant linear speed and zero angular speed as intermediate velocity
            if (speeds[speeds.size()-1] > speeds[speeds.size()-2])
                speeds[speeds.size()-1] = speeds[speeds.size()-2];
            else
                speeds[speeds.size()-2] = speeds[speeds.size()-1];
        }

        static double previous_timestamp = 0;
        int index = 0;
        if (!_is_kicking && !_trajectory_opt->trajectory().empty()) {
            // Glue new piece of trajectory to previous piece
            const double current_timestamp = _trajectory_opt->trajectory().front()->timestamp().count();
            if (_is_gliding && current_timestamp == previous_timestamp)
                index = -1;
            else
                index = std::max(static_cast<int>((current_timestamp - previous_timestamp) / TIMESTEP), 0);
            previous_timestamp = current_timestamp;
        } else {
            previous_timestamp = 0;
        }

        return std::tuple<int, QList<double>>{index, speeds};
    }

    void ToulouseModel::logDataPrepare()
    {
        const QString sep("\t");
        const QString folder("data");
        const QString project("Toulouse");
        const QString application(QApplication::applicationName());
        const QString directory(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
        const QString datetime(QDateTime::currentDateTime().toString("yyyy.MM.dd-hh:mm:ss"));

        // Create the output folder for data logs if necessary
        QString loggingPath = directory;
        QDir(loggingPath).mkdir(application);
        loggingPath += QDir::separator() + application;
        QDir(loggingPath).mkdir(project);
        loggingPath += QDir::separator() + project;
        QDir(loggingPath).mkdir(folder);
        loggingPath += QDir::separator() + folder;

        // Create and open the file and write the header for the trajectory results
        QString fileNameTrajectory = QString("trajectory-%1-%2.txt").arg(_id).arg(datetime);
        _logFileTrajectory.setFileName(loggingPath + QDir::separator() + fileNameTrajectory);
        if (_logFileTrajectory.open(QIODevice::WriteOnly | QIODevice::Text)) {
            _logStreamTrajectory.setDevice(&_logFileTrajectory);
            _logStreamTrajectory << "iskicking" << sep
                                 << "clocktime" << sep
                                 << "timestamp" << sep
                                 << "positionX" << sep
                                 << "positionY" << sep
                                 << "orientation" << sep
                                 << "translation" << sep
                                 << "rotation" << sep
                                 << "acceleration" << endl;
        }

        // Create and open the file and write the header for the tracking results
        QString fileNameTracking = QString("tracking-%1-%2.txt").arg(_id).arg(datetime);
        _logFileTracking.setFileName(loggingPath + QDir::separator() + fileNameTracking);
        if (_logFileTracking.open(QIODevice::WriteOnly | QIODevice::Text)) {
            _logStreamTracking.setDevice(&_logFileTracking);
            _logStreamTracking << "positionX" << sep
                               << "positionY" << sep
                               << "orientation" << endl;
        }

        // Create and open the file and write the header for the kicking results
        QString fileNameKicking = QString("kicking-%1-%2.txt").arg(_id).arg(datetime);
        _logFileKicking.setFileName(loggingPath + QDir::separator() + fileNameKicking);
        if (_logFileKicking.open(QIODevice::WriteOnly | QIODevice::Text)) {
            _logStreamKicking.setDevice(&_logFileKicking);
            _logStreamKicking << "timestamp_milliseconds" << sep
                              << "angular_direction" << sep
                              << "peak_velocity" << sep
                              << "kick_length" << sep
                              << "kick_duration" << sep
                              << "target_positionX" << sep
                              << "target_positionY" << endl;
        }

        // Create and open the file and write the header for the timing results
        QString fileNameTiming = QString("timing-%1-%2.txt").arg(_id).arg(datetime);
        _logFileTiming.setFileName(loggingPath + QDir::separator() + fileNameTiming);
        if (_logFileTiming.open(QIODevice::WriteOnly | QIODevice::Text)) {
            _logStreamTiming.setDevice(&_logFileTiming);
            _logStreamTiming << "time_loop" << sep
                             << "time_execution" << sep
                             << "time_optimization" << endl;
        }
    }

    void ToulouseModel::logDataFinish()
    {
        // Close the file containing the trajectory results
        if (_logFileTrajectory.isOpen())
            _logFileTrajectory.close();

        // Close the file containing the tracking results
        if (_logFileTracking.isOpen())
            _logFileTracking.close();

        // Close the file containing the kicking results
        if (_logFileKicking.isOpen())
            _logFileKicking.close();

        // Close the file containing the timing results
        if (_logFileTiming.isOpen())
            _logFileTiming.close();
    }

    void ToulouseModel::logDataWrite()
    {
        const QString sep("\t");

        // Write the trajectory data
        if (_logFileTrajectory.isOpen() && !_trajectory_opt->trajectory().empty()) {
            for (size_t i = 0; i < _trajectory_opt->trajectory().size(); ++i) {
                _logStreamTrajectory << _is_kicking << sep
                                     << _timestamp << sep
                                     << _trajectory_opt->trajectory().at(i)->timestamp().count() << sep
                                     << _trajectory_opt->trajectory().at(i)->pose().x() << sep
                                     << _trajectory_opt->trajectory().at(i)->pose().y() << sep
                                     << _trajectory_opt->trajectory().at(i)->pose().theta() << sep
                                     << _trajectory_opt->trajectory().at(i)->velocity().translation() << sep
                                     << _trajectory_opt->trajectory().at(i)->velocity().rotation() << sep
                                     << _trajectory_opt->trajectory().at(i)->acceleration().translation() << endl;
            }
        }

        // Write the tracking data
        if (_logFileTracking.isOpen() && _robot != nullptr) {
            _logStreamTracking << _robot->state().position().x() << sep
                               << _robot->state().position().y() << sep
                               << _robot->state().orientation().angleRad() << endl;
        }

        // Write the kicking data
        if (_logFileKicking.isOpen() && _is_kicking) {
            _logStreamKicking << QDateTime::currentMSecsSinceEpoch() << sep
                              << _angular_direction << sep
                              << _peak_velocity << sep
                              << _kick_length << sep
                              << _kick_duration << sep
                              << _desired_position.x << sep
                              << _desired_position.y << endl;
        }

        // Write the timing data
        if (_logFileTiming.isOpen() && _to_be_optimized) {
            _logStreamTiming << _time_loop << sep
                             << _time_exec << sep
                             << _time_opti << endl;
        }
    }

    void ToulouseModel::move()
    {
        _agent->speed = _peak_velocity;

        _agent->direction = _angular_direction;

        _agent->headPos.first  = (_trajectory_opt != nullptr ? _trajectory_opt->trajectory().back()->pose().x() : _position.x) + ARENA_CENTER.first;
        _agent->headPos.second = (_trajectory_opt != nullptr ? _trajectory_opt->trajectory().back()->pose().y() : _position.y) + ARENA_CENTER.second;

        _agent->tailPos.first  = _agent->headPos.first  - _agent->length * std::cos(_agent->direction);
        _agent->tailPos.second = _agent->headPos.second - _agent->length * std::sin(_agent->direction);
    }

    void ToulouseModel::stimulate()
    {
        _desired_position.x = _position.x + _kick_length * std::cos(_angular_direction);
        _desired_position.y = _position.y + _kick_length * std::sin(_angular_direction);

        _desired_speed.vx = (_desired_position.x - _position.x) / _kick_duration;
        _desired_speed.vy = (_desired_position.y - _position.y) / _kick_duration;
    }

    void ToulouseModel::interact()
    {
        int num_fish = _simulation.agents.size();

        // Compute the state for the focal individual
        // distances -> distances to neighbors
        // perception -> angle of focal individual compared to neighbors
        // thetas -> angles to center
        // phis -> relative bearing difference
        Eigen::VectorXd distances, perception, thetas, phis;
        std::tie(distances, perception, thetas, phis) = compute_state();

        // Indices to nearest neighbors
        std::vector<int> nn_idcs = sort_neighbors(distances, _id, Order::INCREASING);

        // Compute influence from the environment to the focal fish
        Eigen::VectorXd influence = Eigen::VectorXd::Zero(num_fish);
        for (int i = 0; i < num_fish; ++i) {
            auto model = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            if (model->id() == _id)
                continue;

            double attraction = wall_distance_attractor(distances(i), radius)
                              * wall_perception_attractor(perception(i))
                              * wall_angle_attractor(phis(i));

            double alignment = alignment_distance_attractor(distances(i), radius)
                             * alignment_perception_attractor(perception(i))
                             * alignment_angle_attractor(phis(i));

            influence(i) = std::abs(attraction + alignment);
        }

        // Indices to highly influential individuals
        std::vector<int> inf_idcs = sort_neighbors(influence, _id, Order::DECREASING);

        // In case the influence from neighboring fish is insignificant,
        // then use the nearest neighbors
        double inf_sum = std::accumulate(influence.data(), influence.data() + influence.size(), 0.);
        std::vector<int> idcs = inf_idcs;
        if (inf_sum < 1.0e-6)
            idcs = nn_idcs;

        // Step using the model
        double r_w, theta_w;
        std::tie(r_w, theta_w) = model_stepper(radius);

        // Compute the new target
        const double robot_safety_margin = 0.010;//0.030; // TODO: tune the safety margin
        const size_t max_count = 1000;
        size_t count = 0;
        double qx, qy;
        do {
            // Decide on the next kick length, kick duration, peak velocity
            stepper();

            if (++count > max_count) {
                //qDebug() << "stuck";
                // Select a direction in reflection to the wall if stuck for too long
                _angular_direction = -_orientation;
            } else {
                // Throw in some free will
                free_will(state_t{distances, perception, thetas, phis},
                          std::tuple<double, double>{r_w, theta_w},
                          idcs);
            }

            // Rejection test (do not want to hit the wall)
            //qx = _position.x + (_kick_length + body_length) * std::cos(_angular_direction);
            //qy = _position.y + (_kick_length + body_length) * std::sin(_angular_direction);
            qx = _position.x + _speed.vx * 0.100 + (_kick_length + body_length + 0.025) * std::cos(_angular_direction); // TODO: tune the optimization time
            qy = _position.y + _speed.vy * 0.100 + (_kick_length + body_length + 0.025) * std::sin(_angular_direction); // TODO: tune the optimization time

            //qDebug() << std::sqrt(qx * qx + qy * qy) << qx << qy << _position.x << _position.y;
        } while (std::sqrt(qx * qx + qy * qy) > radius - robot_safety_margin);
    }

    state_t ToulouseModel::compute_state() const
    {
        size_t num_fish = _simulation.agents.size();

        Eigen::VectorXd distances(num_fish);
        Eigen::VectorXd perception(num_fish);
        Eigen::VectorXd thetas(num_fish);
        Eigen::VectorXd phis(num_fish);

        for (uint i = 0; i < num_fish; ++i) {
            auto fish = std::static_pointer_cast<ToulouseModel>(_simulation.agents[i].second);
            double posx = fish->position().x;
            double posy = fish->position().y;
            double direction = fish->angular_direction();

            distances(i) = std::sqrt(std::pow(_position.x - posx, 2) + std::pow(_position.y - posy, 2));

            thetas(i) = std::atan2(posy - _position.y, posx - _position.x);

            perception(i) = angle_to_pipi(thetas(i) - _angular_direction);

            phis(i) = angle_to_pipi(direction - _angular_direction);
        }

        return state_t{distances, perception, thetas, phis};
    }

    std::vector<int> ToulouseModel::sort_neighbors(const Eigen::VectorXd& values, const int kicker_idx, Order order) const
    {
        std::vector<int> neigh_idcs;
        for (int i = 0; i < values.rows(); ++i) {
            auto model = std::static_pointer_cast<ToulouseModel>(_simulation.agents[kicker_idx].second);
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

    std::tuple<double, double> ToulouseModel::model_stepper(double radius) const
    {
        double r = std::sqrt(std::pow(_position.x, 2) + std::pow(_position.y, 2));
        double rw = radius - r;
        double theta = std::atan2(_position.y, _position.x);
        double thetaW = angle_to_pipi(_angular_direction - theta);
        return std::tuple<double, double>{rw, thetaW};
    }

    void ToulouseModel::stepper()
    {
        double bb;

        const double current_velocity = Eigen::Vector2d(_speed.vx, _speed.vy).norm();
        const double maximum_velocity = _config.robot.max_vel_x - _config.optim.penalty_epsilon;
        const double vel_diff = 0.050;
        const size_t max_count = 1000;
        if (current_velocity + vel_diff < _config.robot.max_vel_x) {
            size_t count = 0;
            do {
                bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0 + 1.0e-16, 1.)));
                _peak_velocity = velocity_coef * std::sqrt(2. / M_PI) * bb;
            } while (_peak_velocity < current_velocity + vel_diff && count++ < max_count);
            if (count >= max_count)
                _peak_velocity = current_velocity + vel_diff;
            if (_peak_velocity > maximum_velocity)
                _peak_velocity = maximum_velocity;
        } else {
            _peak_velocity = maximum_velocity;
        }

        bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0 + 1.0e-16, 1.)));
        _kick_length = length_coef * std::sqrt(2. / M_PI) * bb;

        bb = std::sqrt(-2. * std::log(simu::tools::random_in_range(.0 + 1.0e-16, 1.)));
        _kick_duration = time_coef * std::sqrt(2. / M_PI) * bb;

        _kick_length = _peak_velocity * tau0 * (1. - std::exp(-_kick_duration / tau0));
    }

    void ToulouseModel::free_will(const_state_t state, const std::tuple<double, double>& model_out, const std::vector<int>& idcs)
    {
        double r_w, theta_w;
        Eigen::VectorXd distances, perception, thetas, phis;
        std::tie(r_w, theta_w) = model_out;
        std::tie(distances, perception, thetas, phis) = state;

        double g = std::sqrt(-2. * std::log(tools::random_in_range(0., 1.) + 1.0e-16))
                 * std::sin(2. * M_PI * tools::random_in_range(0., 1.));

        double q = alpha * wall_distance_interaction(gamma_wall, wall_interaction_range, r_w, radius) / gamma_wall;

        double dphi_rand = gamma_rand * (1. - q) * g;
        double dphi_wall = wall_distance_interaction(gamma_wall, wall_interaction_range, r_w, radius)
                         * wall_angle_interaction(theta_w);

        double dphi_attraction = 0;
        double dphi_ali = 0;
        if (idcs.size() >= perceived_agents) {
            for (int i = 0; i < perceived_agents; ++i) {
                int fidx = idcs[i];
                dphi_attraction += wall_distance_attractor(distances(fidx), radius)
                                 * wall_perception_attractor(perception(fidx))
                                 * wall_angle_attractor(phis(fidx));
                dphi_ali += alignment_distance_attractor(distances(fidx), radius)
                          * alignment_perception_attractor(perception(fidx))
                          * alignment_angle_attractor(phis(fidx));
            }
        }

        double dphi = dphi_rand + dphi_wall + dphi_attraction + dphi_ali;
        _angular_direction = angle_to_pipi(_angular_direction + dphi);
    }

    double ToulouseModel::wall_distance_interaction(double gamma_wall, double wall_interaction_range, double ag_radius, double radius) const
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

    double ToulouseModel::round_time_resolution(double period_sec) const
    {
        const double resolution = static_cast<double>(TIMESTEP);
        return resolution * (static_cast<unsigned long int>(period_sec * 1000) /
                             static_cast<unsigned long int>(resolution * 1000));
    }

    Position<double> ToulouseModel::position() const { return _position; }
    Position<double>& ToulouseModel::position() { return _position; }

    Speed<double> ToulouseModel::speed() const { return _speed; }
    Speed<double>& ToulouseModel::speed() { return _speed; }

    bool& ToulouseModel::is_kicking() { return _is_kicking; }
    bool ToulouseModel::is_kicking() const { return _is_kicking; }

    bool& ToulouseModel::is_gliding() { return _is_gliding; }
    bool ToulouseModel::is_gliding() const { return _is_gliding; }

    bool& ToulouseModel::has_stepped() { return _has_stepped; }
    bool ToulouseModel::has_stepped() const { return _has_stepped; }

    bool& ToulouseModel::to_be_optimized() { return _to_be_optimized; }
    bool ToulouseModel::to_be_optimized() const { return _to_be_optimized; }

    double ToulouseModel::orientation() const { return _orientation; }
    double& ToulouseModel::orientation() { return _orientation; }

    double ToulouseModel::angular_direction() const { return _angular_direction; }
    double& ToulouseModel::angular_direction() { return _angular_direction; }

    double ToulouseModel::peak_velocity() const { return _peak_velocity; }
    double& ToulouseModel::peak_velocity() { return _peak_velocity; }

    double ToulouseModel::kick_length() const { return _kick_length; }
    double& ToulouseModel::kick_length() { return _kick_length; }

    double& ToulouseModel::kick_duration() { return _kick_duration; }
    double ToulouseModel::kick_duration() const { return _kick_duration; }

    double ToulouseModel::time_kicker() const { return _time + _kick_duration; }

    double ToulouseModel::time() const { return _time; }
    double& ToulouseModel::time() { return _time; }

    double ToulouseModel::timestep() const { return _timestep; }
    double& ToulouseModel::timestep() { return _timestep; }

    double ToulouseModel::timestamp() const { return _timestamp; }
    double& ToulouseModel::timestamp() { return _timestamp; }

    int ToulouseModel::id() const { return _id; }
    int& ToulouseModel::id() { return _id; }

    FishBot* ToulouseModel::robot() const { return _robot; }
    FishBot*& ToulouseModel::robot() { return _robot; }

    elastic_band::TrajectoryPtr ToulouseModel::referenceTrajectory() const { return _trajectory_ref; }
    elastic_band::TrajectoryPtr& ToulouseModel::referenceTrajectory() { return _trajectory_ref; }

    elastic_band::TrajectoryPtr ToulouseModel::optimizedTrajectory() const { return _trajectory_opt; }
    elastic_band::TrajectoryPtr& ToulouseModel::optimizedTrajectory() { return _trajectory_opt; }

    std::shared_ptr<std::vector<size_t>> ToulouseModel::fixedPoses() const { return _fixed_poses; }
    std::shared_ptr<std::vector<size_t>>& ToulouseModel::fixedPoses() { return _fixed_poses; }

    QMainWindow* ToulouseModel::plotReferencePth() const { return _plot_pth_ref; }
    QMainWindow*& ToulouseModel::plotReferencePth() { return _plot_pth_ref; }

    QMainWindow* ToulouseModel::plotOptimizedPth() const { return _plot_pth_opt; }
    QMainWindow*& ToulouseModel::plotOptimizedPth() { return _plot_pth_opt; }

    QMainWindow* ToulouseModel::plotReferencePos() const { return _plot_pos_ref; }
    QMainWindow*& ToulouseModel::plotReferencePos() { return _plot_pos_ref; }

    QMainWindow* ToulouseModel::plotOptimizedPos() const { return _plot_pos_opt; }
    QMainWindow*& ToulouseModel::plotOptimizedPos() { return _plot_pos_opt; }

    QMainWindow* ToulouseModel::plotReferenceSpd() const { return _plot_spd_ref; }
    QMainWindow*& ToulouseModel::plotReferenceSpd() { return _plot_spd_ref; }

    QMainWindow* ToulouseModel::plotOptimizedSpd() const { return _plot_spd_opt; }
    QMainWindow*& ToulouseModel::plotOptimizedSpd() { return _plot_spd_opt; }

    QMainWindow* ToulouseModel::plotReferenceVel() const { return _plot_vel_ref; }
    QMainWindow*& ToulouseModel::plotReferenceVel() { return _plot_vel_ref; }

    QMainWindow* ToulouseModel::plotOptimizedVel() const { return _plot_vel_opt; }
    QMainWindow*& ToulouseModel::plotOptimizedVel() { return _plot_vel_opt; }

    QMainWindow* ToulouseModel::plotReferenceAcc() const { return _plot_acc_ref; }
    QMainWindow*& ToulouseModel::plotReferenceAcc() { return _plot_acc_ref; }

    QMainWindow* ToulouseModel::plotOptimizedAcc() const { return _plot_acc_opt; }
    QMainWindow*& ToulouseModel::plotOptimizedAcc() { return _plot_acc_opt; }

} // namespace Fishmodel
