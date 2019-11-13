#include "toulouseModel.hpp"

#include <SetupType.hpp>
#include <settings/CalibrationSettings.hpp>
#include <settings/CommandLineParameters.hpp>
#include <settings/RobotControlSettings.hpp>

#include <QDebug>
#include <QQueue>

#include <eigen3/Eigen/Core>

#include <boost/range.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/irange.hpp>

#include <memory>
#include <cmath>
#include <algorithm>

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

        // int argc = 0;
        // char** argv = nullptr;
        // ros::init(argc, argv, "elastic_band");
        // ros::NodeHandle node_handle("~");

        _config.trajectory.teb_autosize = false;
        _config.trajectory.dt_ref = 0.3;
        _config.trajectory.dt_hysteresis = 0.1;
        _config.trajectory.min_samples = 3;
        _config.trajectory.max_samples = 500;
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
        _config.robot.max_vel_x_backwards = 0.3;
        _config.robot.max_vel_y = 0.0;
        _config.robot.max_vel_theta = 18;
        _config.robot.acc_lim_x = 1.3;
        _config.robot.acc_lim_y = 0.0;
        _config.robot.acc_lim_theta = 20;
        _config.robot.min_turning_radius = 0;
        _config.robot.wheelbase = 1.0;
        _config.robot.cmd_angle_instead_rotvel = false;
        _config.robot.is_footprint_dynamic = false;
        _config.goal_tolerance.xy_goal_tolerance = 0.2;
        _config.goal_tolerance.yaw_goal_tolerance = 0.2;
        _config.goal_tolerance.free_goal_vel = false;
        _config.goal_tolerance.complete_global_plan = true;
        _config.obstacles.min_obstacle_dist = 0.005;
        _config.obstacles.inflation_dist = 0.6;
        _config.obstacles.dynamic_obstacle_inflation_dist = 0.6;
        _config.obstacles.include_dynamic_obstacles = true;
        _config.obstacles.include_costmap_obstacles = true;
        _config.obstacles.costmap_obstacles_behind_robot_dist = 1.5;
        _config.obstacles.obstacle_poses_affected = 25;
        _config.obstacles.legacy_obstacle_association = false;
        _config.obstacles.obstacle_association_force_inclusion_factor = 1.5;
        _config.obstacles.obstacle_association_cutoff_factor = 5;
        _config.obstacles.costmap_converter_plugin = "";
        _config.obstacles.costmap_converter_spin_thread = true;
        _config.obstacles.costmap_converter_rate = 5;
        _config.optim.no_inner_iterations = 5;
        _config.optim.no_outer_iterations = 4;
        _config.optim.optimization_activate = true;
        _config.optim.optimization_verbose = true;
        _config.optim.penalty_epsilon = 0.001;
        _config.optim.weight_max_vel_x = 1000;
        _config.optim.weight_max_vel_y = 1000;
        _config.optim.weight_max_vel_theta = 1000;
        _config.optim.weight_acc_lim_x = 1000;
        _config.optim.weight_acc_lim_y = 1000;
        _config.optim.weight_acc_lim_theta = 1000;
        _config.optim.weight_kinematics_nh = 1000;
        _config.optim.weight_kinematics_forward_drive = 0;
        _config.optim.weight_kinematics_turning_radius = 0;
        _config.optim.weight_optimaltime = 0;
        _config.optim.weight_shortest_path = 0;
        _config.optim.weight_profile_fidelity = 10;
        _config.optim.weight_obstacle = 50;
        _config.optim.weight_inflation = 0.1;
        _config.optim.weight_dynamic_obstacle = 50;
        _config.optim.weight_dynamic_obstacle_inflation = 0.1;
        _config.optim.weight_viapoint = 1;
        _config.optim.weight_prefer_rotdir = 0;
        _config.optim.weight_adapt_factor = 1.0;
        _config.optim.obstacle_cost_exponent = 1.0;
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
        _viapoints.clear();
        _obstacles.clear();
        _obstacles.push_back(elastic_band::ObstaclePtr(new elastic_band::CircularObstacle(ARENA_CENTER.first, ARENA_CENTER.second, radius)));
        _robot_shape.clear();
        _robot_shape.push_back(Eigen::Vector2d(-0.22, -0.11));
        _robot_shape.push_back(Eigen::Vector2d(+0.22, -0.11));
        _robot_shape.push_back(Eigen::Vector2d(+0.22, +0.11));
        _robot_shape.push_back(Eigen::Vector2d(-0.22, +0.11));
        _robot_model    = elastic_band::RobotFootprintModelPtr(new elastic_band::PolygonRobotFootprint(_robot_shape));
        // _visualization  = elastic_band::TebVisualizationPtr   (new elastic_band::TebVisualization(node_handle, _config));
        _planner        = elastic_band::TebPlannerPtr(new elastic_band::TebPlanner());
        _trajectory_ref = elastic_band::TrajectoryPtr(new elastic_band::Trajectory());
        _trajectory_opt = elastic_band::TrajectoryPtr(new elastic_band::Trajectory());

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
        qInfo() << "<<<<<<< TOULOUSE MODEL: step";
#if 0
        qDebug() << _agent->headPos.first << " " << _agent->headPos.second;

        _position.x = _agent->headPos.first;
        _position.y = _agent->headPos.second;

        _agent->headPos.first = ARENA_CENTER.first + 0.25 * std::cos(_time * M_PI / 180);
        _agent->headPos.second = ARENA_CENTER.second + 0.25 * std::sin(_time * M_PI / 180);
        //        _agent->headPos.first = ARENA_CENTER.first;
        //        _agent->headPos.second = ARENA_CENTER.second;
        _agent->direction = 0;

        _time += 5;
        if (_time > 360)
            _time = 0;
        _agent->updateAgentPosition(_simulation.dt);

#else
        static int count = 0;

        //        if (count++ == 0) {
        //            _speed.vx = (_agent->headPos.first - _position.x) / _simulation.dt;
        //            _speed.vy = (_agent->headPos.second - _position.y) / _simulation.dt;

        //            _position.x = _agent->headPos.first - ARENA_CENTER.first;
        //            _position.y = _agent->headPos.second - ARENA_CENTER.second;
        //        }
        //        else {
        //        _position.x -= ARENA_CENTER.first;
        //        _position.y -= ARENA_CENTER.second;
        //            qDebug() << "disregarding current position";
        //        }

        //        _angular_direction = angle_to_pipi(_agent->direction);

        std::pair<Agent*, Behavior*> current_agent(_agent, this);
        auto result
            = std::find(_simulation.robots.begin(), _simulation.robots.end(), current_agent);
        if (result == _simulation.robots.end())
            return;

        qDebug() << "robot " << _position.x << " " << _position.y << " " << _angular_direction
                 << " " << angle_to_pipi(_agent->direction);

        _time += _simulation.dt;
        qDebug() << "current model time =" << _time << "s";

        //_is_kicking = true;
        _is_kicking = (_time >= _kick_duration);

        if (_is_kicking) {
            _time = 0;
            qDebug() << "reset model time =" << _time << "s";

            // the individuals decide on the desired position
            stimulate(); // kicking individual goes first

            // apply attractors/repulsors and update the fish intuitions
            // (for the kicker, the rest are in their gliding phase)
            interact();

            // TEST PURPOSES
            _angular_direction = 0.2; // [rad]
            _peak_velocity = 0.15; // [m/s]
            _kick_duration = 1.008; // [s]
            tau0 = 0.8; // [s]

            // compute reference trajectory
            //double timestep = 1. / static_cast<double>(RobotControlSettings::get().controlFrequencyHz()); // [s]
            double timestep = 0.005; // [s]
            unsigned int nb_commands = static_cast<unsigned int>(std::floor(_kick_duration / timestep)); // [#]
            std::chrono::milliseconds timestep_ms(static_cast<long int>(std::ceil(timestep * 1000.))); // [ms]
            elastic_band::TimestepPtr timestep_ptr(new elastic_band::Timestep(timestep_ms));
            elastic_band::TimestepContainer timestep_profile(nb_commands - 1, timestep_ptr);
            elastic_band::VelocityContainer velocity_profile(nb_commands);
            elastic_band:: PoseSE2Container     pose_profile(nb_commands);
/*
            double x, y, theta;
            // double dtheta = 2 * M_PI / 100 / M_PI * 2; // 40 [mrad]
            // double dtheta = 2 * M_PI / 100 / M_PI * 2 / 10 / 8; // 0.5 [mrad]
            // double dtheta = 2 * M_PI / 100 / M_PI * 2 / 10 / 4; // 1 [mrad]
            double dtheta = 2 * M_PI / 100 / M_PI * 2 / 10 / 4 / 1.1; // 0.9 [mrad]
            double radius = 10. / 100.; // 10 [cm]
            for (size_t i = 0; i < pose_profile.size(); i++) {
                theta = i * dtheta + M_PI / 2;
                x = radius * std::cos(theta - M_PI / 2);
                y = radius * std::sin(theta - M_PI / 2);
                pose_profile.at(i) = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(x, y, theta));
            }
*/
            double acceleration = _config.robot.acc_lim_x; // [m/s^2]
            double translation_init = 0.01; // [m/s]
            double rotation_init = 0; // [rad/s]
            double angle_init = 0; // [rad]
            Eigen::Vector2d position_init = 0 * Eigen::Vector2d(std::cos(angle_init), std::sin(angle_init)); // [m]
            double duration = 0; // [s]
            double duration_phase1 = duration;
            double duration_phase2 = duration;
            double duration_phase3 = duration;
            double translation = translation_init;
            double rotation = rotation_init;
            Eigen::Vector2d position = position_init;
            Eigen::Vector2d position_phase1 = position;
            Eigen::Vector2d position_phase2 = position;
            Eigen::Vector2d position_phase3 = position;
            double theta = angle_init;
            double x = position.x();
            double y = position.y();
            QQueue<double> errors;
            double error_old = 0;
            double error_new = 0;
            double error_dif = 0;
            double error_sum = 0;
            // double Kp = RobotControlSettings::get().pidControllerSettings().kp();
            // double Ki = RobotControlSettings::get().pidControllerSettings().ki();
            // double Kd = RobotControlSettings::get().pidControllerSettings().kd();
            double Kp = 50.0;
            double Ki = 1.00;
            double Kd = 0.05;
            _angular_direction = angle_to_pipi(_angular_direction);
            pose_profile.front() = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(x, y, theta));
            for (size_t i = 1; i < pose_profile.size(); i++) {
                duration = i * timestep;
                if (std::abs(_angular_direction - theta) > 1e-3) { // 1. Orientation
                    duration_phase1 = duration;
                    duration_phase2 = duration_phase1;
                    duration_phase3 = duration_phase1;
                    error_new = angle_to_pipi(_angular_direction - theta);
                    error_dif = 0;
                    error_sum = 0;
                    if (errors.size() > 0) {
                        error_dif = error_new - error_old;
                    }
                    if (rotation > -_config.robot.max_vel_theta && rotation < _config.robot.max_vel_theta) {
                        errors.enqueue(error_new);
                        for (double error : errors) {
                            error_sum += error;
                        }
                    }
                    if (duration_phase1 > timestep) {
                        rotation = Kp * error_new + Ki * error_sum * timestep + Kd * error_dif / timestep;
                    } else {
                        rotation = rotation_init;
                    }
                    if (rotation < -_config.robot.max_vel_theta) {
                        rotation = -_config.robot.max_vel_theta;
                    }
                    if (rotation > _config.robot.max_vel_theta) {
                        rotation = _config.robot.max_vel_theta;
                    }
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
                    position_phase3 = position_phase1;
                } else if (acceleration * (duration - duration_phase1/* + timestep*/) + translation < _peak_velocity) { // 2. Acceleration
                    duration_phase2 = duration - duration_phase1;
                    duration_phase3 = duration_phase2;
                    position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * (acceleration * duration_phase2 * duration_phase2 / 2 + translation * duration_phase2) + position_phase1;
                    position_phase2 = position;
                    position_phase3 = position_phase2;
                } else if (duration - duration_phase2 - duration_phase1 <= _kick_duration) { // 3. Relaxation
                    duration_phase3 = duration - duration_phase2 - duration_phase1;
                    position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * _peak_velocity * tau0 * (1. - std::exp(-duration_phase3 / tau0)) + position_phase2;
                    position_phase3 = position;
                } else {
                    translation = duration_phase3 * _peak_velocity * tau0 * std::exp(-duration_phase3 / tau0);
                    position = Eigen::Vector2d(std::cos(theta), std::sin(theta)) * translation * timestep + position_phase3;
                }
                x = position.x();
                y = position.y();
                pose_profile.at(i) = elastic_band::PoseSE2Ptr(new elastic_band::PoseSE2(x, y, theta));
            }

            _trajectory_ref->robotParameters().wheel_radius = 0.005; // [m]
            _trajectory_ref->robotParameters().wheel_distance = 0.018; // [m]
            _trajectory_ref->setProfileTimestep(timestep_profile, false);
            _trajectory_ref->setProfilePose(pose_profile);
            // elastic_band::PoseSE2Container pose_profile = _trajectory_ref->getProfilePose();
            // for (elastic_band::PoseSE2Container::const_iterator pose = pose_profile.begin(); pose != pose_profile.end(); pose++) {
            //     std::cout << **pose << std::endl;
            // }
            // std::cout << *_trajectory_ref << std::endl;

            // visualize reference trajectory
            _plot_path_ref = elastic_band::TebPlot::plotPath               (&(*_trajectory_ref), _plot_path_ref, "Reference path");
            _plot_pose_ref = elastic_band::TebPlot::plotProfilePose        (&(*_trajectory_ref), _plot_pose_ref, "Reference pose");
            _plot_spd_ref  = elastic_band::TebPlot::plotProfileSpeed       (&(*_trajectory_ref), _plot_spd_ref,  "Reference speed");
            _plot_vel_ref  = elastic_band::TebPlot::plotProfileVelocity    (&(*_trajectory_ref), _plot_vel_ref,  "Reference velocity");
            _plot_acc_ref  = elastic_band::TebPlot::plotProfileAcceleration(&(*_trajectory_ref), _plot_acc_ref,  "Reference acceleration");

            // initialize environment
            _viapoints.clear();
            // _viapoints.push_back(Eigen::Vector2d(0,0));
            // _viapoints.push_back(Eigen::Vector2d(1,1));
            _obstacles.clear();
            // _obstacles.push_back(boost::make_shared<elastic_band::PointObstacle>(0,0));
            // _obstacles.push_back(boost::make_shared<elastic_band::PointObstacle>(1,1));
            _planner->initialize(_config, &_obstacles, _robot_model, _visualization, &_viapoints);

            // plan subsequent trajectory
            _planner->setVelocityStart(_trajectory_ref->trajectory().front()->velocity(), false);
            _planner->setVelocityGoal (_trajectory_ref->trajectory().at(_trajectory_ref->trajectory().size()-2)->velocity(), false);
            _planner->plan(*_trajectory_ref, true);

            if (_planner->isOptimized()) {
                // CHECK TRAJECTORY FEASABILITY!!! + automatic detection of convergence included a limited overall computation time availabe to return a resulting trajectory
                const bool feasible = true;
                if (feasible) {
                    // store optimized trajectory
                    // _planner->getVelocityProfile(velocity_profile);
                    // for (elastic_band::VelocityContainer::const_iterator velocity = velocity_profile.begin(); velocity != velocity_profile.end(); velocity++) {
                    //     std::cout << **velocity << std::endl;
                    // }
                    _trajectory_opt->robotParameters().wheel_radius   = _trajectory_ref->robotParameters().wheel_radius;
                    _trajectory_opt->robotParameters().wheel_distance = _trajectory_ref->robotParameters().wheel_distance;
                    _planner->getFullTrajectory(*_trajectory_opt);
                     std::cout << *_trajectory_opt << std::endl;

                    // compute optimization performance
                    _planner->computeCurrentCost();
                    double optim_cost = _planner->getCurrentCost();
                    qDebug() << "Cost of the last optimization process =" << optim_cost;

                    // visualize optimized trajectory
                    // _planner->visualize();
                    // _visualization->publishObstacles(_obstacles);
                    // _visualization->publishViaPoints(_viapoints);
                    _plot_path_opt = elastic_band::TebPlot::plotPath               (&(*_trajectory_opt), _plot_path_opt, "Optimized path");
                    _plot_pose_opt = elastic_band::TebPlot::plotProfilePose        (&(*_trajectory_opt), _plot_pose_opt, "Optimized pose");
                    _plot_spd_opt  = elastic_band::TebPlot::plotProfileSpeed       (&(*_trajectory_opt), _plot_spd_opt,  "Optimized speed");
                    _plot_vel_opt  = elastic_band::TebPlot::plotProfileVelocity    (&(*_trajectory_opt), _plot_vel_opt,  "Optimized velocity");
                    _plot_acc_opt  = elastic_band::TebPlot::plotProfileAcceleration(&(*_trajectory_opt), _plot_acc_opt,  "Optimized acceleration");

                    // send control commands
                    // (not to be done here)
                } else {
                    qDebug() << "The TEB planner was not able to find a feasible solution for the trajectory.";
                }
            } else {
                qDebug() << "The TEB planner was not able to find an optimized solution for the trajectory.";
            }
        }

        // update position and velocity information -- actual move step
        move();

        _is_kicking = false;
#endif
    }

    QList<double> ToulouseModel::getSpeedCommands() const
    {
        QList<double> speeds;
        elastic_band::VelocityContainer velocity_profile;
        _trajectory_opt->getProfileVelocity(velocity_profile);
        speeds.reserve(static_cast<int>(2 * velocity_profile.size()));
        for (size_t i = 0; i < velocity_profile.size(); i++) {
            // convert speeds from [rad/s] to [cm/s]
            velocity_profile.at(i)->wheel() *= velocity_profile.at(i)->getRadius() * 100.;
            speeds.append(velocity_profile.at(i)->wheel()[0]);
            speeds.append(velocity_profile.at(i)->wheel()[1]);
        }
        return speeds;
    }

    void ToulouseModel::stimulate()
    {
        // TODO: kick length is not initialized the first time !!
        //_time += _kick_duration;
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
            qDebug() << "stuck";
            stepper(); // decide on the next kick length, duration, peak velocity
            free_will(state_t{distances, perception, thetas, phis},
                      std::tuple<double, double>{r_w, theta_w},
                      idcs); // throw in some free will

            // rejection test -- don't want to hit the wall
            qx = _desired_position.x + (_kick_length + body_length) * std::cos(_angular_direction);
            qy = _desired_position.y + (_kick_length + body_length) * std::sin(_angular_direction);

            qDebug() << std::sqrt(qx * qx + qy * qy) << " " << qx << " " << qy << " " << _position.x
                     << " " << _position.y;
        } while (std::sqrt(qx * qx + qy * qy) > radius);
    }
/*
    void ToulouseModel::move()
    {
        // kicker advancing to the new position
        _position = _desired_position;
        _speed = _desired_speed;

        //        _position.x += ARENA_CENTER.first;
        //        _position.y += ARENA_CENTER.second;

        _agent->headPos.first = _position.x + ARENA_CENTER.first;
        _agent->headPos.second = _position.y + ARENA_CENTER.second;
        //        _agent->updateAgentPosition(_simulation.dt);
        _agent->updateAgentPosition(_kick_duration);
    }
*/
    void ToulouseModel::move()
    {
        // Compute updated position along trajectory
        double positionUpgrade = _peak_velocity * tau0 *
                                 (std::exp(-(_time > _simulation.dt ? _time - _simulation.dt : 0) / tau0)
                                - std::exp(- _time / tau0));
        _desired_position.x = _position.x + positionUpgrade * std::cos(_angular_direction);
        _desired_position.y = _position.y + positionUpgrade * std::sin(_angular_direction);

        // Kicker advances to the new position
        _speed.vx = (_desired_position.x - _position.x) / _simulation.dt;
        _speed.vy = (_desired_position.y - _position.y) / _simulation.dt;
        _position = _desired_position;

        // Update robot position
        _agent->speed = std::sqrt(std::pow(_speed.vx, 2) + std::pow(_speed.vx, 2));
        _agent->headPos.first  = _position.x + ARENA_CENTER.first;
        _agent->headPos.second = _position.y + ARENA_CENTER.second;
        _agent->updateAgentPosition(_simulation.dt);
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

            distances(i) = std::sqrt(
                std::pow(_desired_position.x - posx, 2) + std::pow(_desired_position.y - posy, 2));

            thetas(i) = std::atan2(posy - _desired_position.y, posx - _desired_position.x);

            perception(i) = angle_to_pipi(thetas(i) - _angular_direction);

            phis(i) = angle_to_pipi(direction - _angular_direction);
        }

        return state_t{distances, perception, thetas, phis};
    }

    std::vector<int> ToulouseModel::sort_neighbours(
        const Eigen::VectorXd& values, const int kicker_idx, Order order) const
    {
        std::vector<int> neigh_idcs;
        for (int i = 0; i < values.rows(); ++i) {
            auto model
                = std::static_pointer_cast<ToulouseModel>(_simulation.agents[kicker_idx].second);
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
        if (idcs.size() >= perceived_agents) {
            for (int j = 0; j < perceived_agents; ++j) {
                int fidx = idcs[j];
                dphi_attraction += wall_distance_attractor(distances(fidx), radius)
                    * wall_perception_attractor(perception(fidx)) * wall_angle_attractor(phis(fidx));
                dphi_ali += alignment_distance_attractor(distances(fidx), radius)
                    * alignment_perception_attractor(perception(fidx))
                    * alignment_angle_attractor(phis(fidx));
            }
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
        return std::tuple<double, double>{rw, thetaW};
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
    Position<double>& ToulouseModel::position() { return _position; }

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
