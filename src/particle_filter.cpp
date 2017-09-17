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

#include "particle_filter.h"

using namespace std;

const int PARTICLE_NUM = 50;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if (!is_initialized) {
        num_particles = PARTICLE_NUM;

        normal_distribution<double> x_dist(x, std[0]);
        normal_distribution<double> y_dist(y, std[1]);
        normal_distribution<double> theta_dist(theta, std[2]);
        for (int i = 0; i < num_particles; i++) {
            Particle p = {i, x_dist(gen), y_dist(gen),theta_dist(gen), 1.0};
            particles.push_back(p);
        }
        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> x_dist(0, std_pos[0]);
    normal_distribution<double> y_dist(0, std_pos[1]);
    normal_distribution<double> theta_dist(0, std_pos[2]);


    for (Particle& p : particles) {
        if (fabs(yaw_rate) < 0.00001) {  
            p.x += velocity * delta_t * cos(p.theta) + x_dist(gen);
            p.y += velocity * delta_t * sin(p.theta) + y_dist(gen);
            p.theta += theta_dist(gen);
        } else {
            // particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            // particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            // particles[i].theta += yaw_rate * delta_t;
            p.x += (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) * velocity / yaw_rate + x_dist(gen);
            p.y += (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) * velocity / yaw_rate + y_dist(gen);
            p.theta += yaw_rate * delta_t + theta_dist(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs& obs : observations) {
        double min_distance = -1.0;
        for (LandmarkObs& pred : predicted) {
            double distance = dist(pred.x, pred.y, obs.x, obs.y);
            if (min_distance < 0 || distance < min_distance) {
                min_distance = distance;
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
            if (dist(p.x, p.y, landmark.x_f, landmark.y_f) > sensor_range) {
                continue;
            }

            LandmarkObs obs;
            obs.id = landmark.id_i;
            obs.x = landmark.x_f;
            obs.y = landmark.y_f;
            predicted.push_back(obs);
        }

        vector<LandmarkObs> transformed;
        for (LandmarkObs& obs : observations) {
            LandmarkObs trans;
            trans.id = obs.id;
            trans.x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
            trans.y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
            transformed.push_back(trans);
        }
        dataAssociation(predicted, transformed);

        p.weight = 1.0;
        for (LandmarkObs& trans : transformed) {
            for (LandmarkObs& pred : predicted) {
                if (trans.id == pred.id) {
                    double exponent = pow(trans.x - pred.x, 2) / (2 * pow(sigma_x, 2)) + pow(trans.y - pred.y, 2) / (2 * pow(sigma_y, 2));
                    // cout << "!!!! (" << __LINE__ << ") exp=" << exponent << " exp()=" << exp(-exponent) << " sigma_x=" << pow(trans.x - pred.x, 2) << endl;
                    p.weight *= gauss_norm * exp(-exponent);
                    // break;
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

    vector<double> weights;
    for (Particle& p : particles) {
        weights.push_back(p.weight);
    }

    std::discrete_distribution<> d(weights.begin(), weights.end());
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
