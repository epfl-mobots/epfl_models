#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#include <limits>

#include "epfl_factory.hpp"

using namespace Fishmodel;

template <class A>
std::pair<Agent*, Behavior*> EpflSimulationFactory::_createAgent(std::string const& behaviorType)
{
    Agent* a = new A(arena);
    Behavior* b = nullptr;
    if (behaviorType == "SFM") {
        b = new SocialFishModel(*_sim, a);
    }
    else if (behaviorType == "DPT") {
        b = new DensestPointModel(*_sim, a);
    }
    else if (behaviorType == "TM") {
        b = new ToulouseModel(*_sim, a);
    }
    else {
        b = new Behavior(*_sim, a);
    }
    return {a, b};
}

std::unique_ptr<Simulation> EpflSimulationFactory::create()
{
    _sim = new Simulation(arena);
    std::unique_ptr<Simulation> sim(_sim);
    currentTrajectoryIndex = 0;

    // Create Agents
    for (size_t i = 0; i < nbFishes; ++i) {
        auto ab = _createAgent<FishAgent>(behaviorFishes);
        sim->agents.push_back(
            {std::unique_ptr<Agent>(ab.first), std::unique_ptr<Behavior>(ab.second)});
        sim->fishes.push_back(ab);
    }
    for (size_t i = 0; i < nbRobots; ++i) {
        auto ab = _createAgent<VirtualAgent>(behaviorRobots);
        sim->agents.push_back(
            {std::unique_ptr<Agent>(ab.first), std::unique_ptr<Behavior>(ab.second)});
        sim->robots.push_back(ab);
    }
    for (size_t i = 0; i < nbVirtuals; ++i) {
        auto ab = _createAgent<VirtualAgent>(behaviorVirtuals);
        sim->agents.push_back(
            {std::unique_ptr<Agent>(ab.first), std::unique_ptr<Behavior>(ab.second)});
        sim->virtuals.push_back(ab);
    }
    return std::move(sim);
}

Simulation* EpflSimulationFactory::createAndShare()
{
    _sim = new Simulation(arena);
    Simulation* sim(_sim);
    currentTrajectoryIndex = 0;

    // Create Agents
    for (size_t i = 0; i < nbFishes; ++i) {
        auto ab = _createAgent<FishAgent>(behaviorFishes);
        sim->agents.push_back(
            {std::shared_ptr<Agent>(ab.first), std::shared_ptr<Behavior>(ab.second)});
        sim->fishes.push_back(ab);
    }
    for (size_t i = 0; i < nbRobots; ++i) {
        auto ab = _createAgent<VirtualAgent>(behaviorRobots);
        sim->agents.push_back(
            {std::shared_ptr<Agent>(ab.first), std::shared_ptr<Behavior>(ab.second)});
        sim->robots.push_back(ab);
    }
    for (size_t i = 0; i < nbVirtuals; ++i) {
        auto ab = _createAgent<VirtualAgent>(behaviorVirtuals);
        sim->agents.push_back(
            {std::shared_ptr<Agent>(ab.first), std::shared_ptr<Behavior>(ab.second)});
        sim->virtuals.push_back(ab);
    }
    return sim;
}
