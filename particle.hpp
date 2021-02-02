#ifndef PARTICLE_HPP
#define PARTICLE_HPP

namespace generalized_langevin {
    class Particle {
        public:
            Particle(){}
            //座標
            double x;
            double y;
            double z;
            //速度
            double vx;
            double vy;
            double vz;
            double mass;
    };
}//generalized_langevin

#endif