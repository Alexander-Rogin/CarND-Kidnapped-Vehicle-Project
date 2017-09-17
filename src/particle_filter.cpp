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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random>
#include <initializer_list>

#include "particle_filter.h"

using namespace std;

const int PARTICLE_NUM = 10;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (!is_initialized) {
        num_particles = PARTICLE_NUM;

        std::random_device rd;
        std::mt19937 gen(rd());
        normal_distribution<> x_dist(x, std[0]);
        normal_distribution<> y_dist(y, std[1]);
        normal_distribution<> theta_dist(theta, std[2]);
        for (int i = 0; i < num_particles; i++) {
            Particle p = {i, x_dist(gen), y_dist(gen), theta_dist(gen)};
            particles.push_back(p);

            weights.push_back(1);
        }
        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    for (Particle& p : particles) {
        p.x += (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) * velocity / p.theta;
        p.y += (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) * velocity / p.theta;
        p.theta += yaw_rate * delta_t;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs& pred : predicted) {
        double min_distance = -1.0;
        for (LandmarkObs& obs : observations) {
            double diff_x = pred.x - obs.x;
            double diff_y = pred.y - obs.y;
            double dist = sqrt(diff_x * diff_x + diff_y * diff_y);
            if (min_distance < 0 || dist < min_distance) {
                min_distance = dist;
                obs.id = pred.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double sigma_x = std_landmark[0];
    double sigma_y = std_landmark[1];
    double gauss_norm = (1 / (2 * M_PI * sigma_x * sigma_y));
    std::vector<LandmarkObs> predicted;
    for (Particle& p : particles) {
        for (Map::single_landmark_s& landmark : map_landmarks.landmark_list) {
            LandmarkObs obs;
            obs.id = landmark.id_i;
            obs.x = p.x + (cos(p.theta) * landmark.x_f - sin(p.theta) * landmark.y_f);
            obs.y = p.y + (sin(p.theta) * landmark.x_f + cos(p.theta) * landmark.y_f);
            predicted.push_back(obs);
        }
        dataAssociation(predicted, observations);

        for (LandmarkObs& obs : observations) {
            for (LandmarkObs& landmark : predicted) {
                if (obs.id == landmark.id) {
                    double exponent = pow(obs.x - landmark.x, 2) / (2 * pow(sigma_x, 2)) + pow(obs.y - landmark.y, 2) / (2 * pow(sigma_y, 2));
                    p.weight = gauss_norm * exp(-exponent);
                    weight_sum += p.weight;
                    break;
                }
            }
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::random_device rd;
    std::mt19937 gen(rd());

    initializer_list<double> weights;
    for (Particle& p : particles) {
        weights.push_back(p.weight);
    }

    std::discrete_distribution<> d(weights);
    vector<Particle> sampled_particles;
    for (int i = 0; i < num_particles; i++) {
        sampled_particles.push_back(particles[d(gen)]);
    }
    particles = sampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
