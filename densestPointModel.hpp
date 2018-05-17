#ifndef DENSESTPOINTMODEL_HPP
#define DENSESTPOINTMODEL_HPP

#include "socialFishModel.hpp"

namespace Fishmodel {

    class DensestPointModel : public SocialFishModel {
    public:
        DensestPointModel(Simulation& simulation, Agent* agent = nullptr);

        void init();
        virtual void reinit() override;
        virtual void step() override;
    };

} // namespace Fishmodel

#endif // SOCIALFISHMODEL_HPP
