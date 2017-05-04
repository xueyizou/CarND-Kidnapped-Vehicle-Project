/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Number of particles to draw
    num_particles = 1000;

    // noise generation
    default_random_engine gen;
    normal_distribution<double> N_x_init(0, std[0]);
    normal_distribution<double> N_y_init(0, std[1]);
    normal_distribution<double> N_theta_init(0, std[2]);

    for(int i=0; i<num_particles; ++i)
    {
        Particle p;
        p.id=i;
        p.x=x+N_x_init(gen);
        p.y=y+N_y_init(gen);
        p.theta=theta+N_theta_init(gen);
        p.weight=1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }

    // Flag, if filter is initialized
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // noise generation
    default_random_engine gen;
    normal_distribution<double> N_x_init(0, std_pos[0]);
    normal_distribution<double> N_y_init(0, std_pos[1]);
    normal_distribution<double> N_theta_init(0, std_pos[2]);

    for(int i=0; i<num_particles; ++i)
    {
        double theta_0 = particles[i].theta;

        particles[i].x += velocity*(sin(theta_0+yaw_rate*delta_t) - sin(theta_0))/yaw_rate + N_x_init(gen);
        particles[i].y += velocity*(cos(theta_0) - cos(theta_0+yaw_rate*delta_t))/yaw_rate + N_y_init(gen);
        particles[i].theta +=yaw_rate*delta_t + N_theta_init(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
        std::vector<LandmarkObs> observations, Map map) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    int num_landmarks = map.landmark_list.size();
    int num_landmarkObs = observations.size();

    for(int i=0; i<num_particles; ++i)
    {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        std::vector<LandmarkObs> observations_pred;
        for(int j=0; j<num_landmarks; ++j)
        {
            Map::single_landmark_s lmk = map.landmark_list[j];
            float lmk_x = lmk.x_f;
            float lmk_y = lmk.y_f;

            LandmarkObs lmk_obs_pred;
            lmk_obs_pred.x = (lmk_x -x)*cos(theta) + (lmk_y -y)*sin(theta);
            lmk_obs_pred.y = -(lmk_x -x)*sin(theta) + (lmk_y -y)*cos(theta);

            observations_pred.push_back(lmk_obs_pred);

        }

        for(int k=0; k<num_landmarkObs; ++k)
        {
             LandmarkObs lmk_obs = observations[k];
             if(lmk_obs.x*lmk_obs.x + lmk_obs.y*lmk_obs.y > sensor_range*sensor_range)
             {
                 continue;
             }

             double min_dist=99999;
             double matched_lmk_idx=0;

             for(int l=0; l<num_landmarks; ++l)
             {
                 LandmarkObs lmk_obs_pred = observations_pred[l];
                 double distance = dist(lmk_obs.x, lmk_obs.y, lmk_obs_pred.x, lmk_obs_pred.y);
                 if(distance<min_dist)
                 {
                     min_dist=distance;
                     matched_lmk_idx = l;
                 }

             }
             double x_diff = observations_pred[matched_lmk_idx].x-lmk_obs.x;
             double y_diff = observations_pred[matched_lmk_idx].y-lmk_obs.y;
             particles[i].weight *= exp(-x_diff*x_diff/(2*std_landmark[0]*std_landmark[0])
                                        -y_diff*y_diff/(2*std_landmark[1]*std_landmark[1]));
        }
        weights[i]=particles[i].weight;

    }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    discrete_distribution<> d(weights.begin(), weights.end());
    vector<Particle> new_particles;
    for(int i=0; i<num_particles; ++i)
    {
        Particle p;
        int index = d(gen);
        p.id=i;
        p.x = particles[index].x;
        p.y = particles[index].y;
        p.theta = particles[index].theta;
        p.weight= 1.0;

        new_particles.push_back(p);
    }

    particles = new_particles;

    for(int i=0; i<num_particles; ++i)
    {
        weights[i]=1.0;
    }

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
