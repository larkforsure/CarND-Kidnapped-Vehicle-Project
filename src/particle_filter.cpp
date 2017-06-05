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

#include "particle_filter.h"


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. 
	num_particles = 80;

	//Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// noise generation
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	// initialize all particles to first position
	particles.resize(num_particles);
	for (int i = 0; i < num_particles; ++i) {
		// initialize particle attributes( with GPS data) and add noises
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
	}
	weights.resize(num_particles, 1.0);
	// set initialized
	is_initialized = true;

	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// noise generation
	std::default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
		// get old theta measurement
		const double theta0 = particles[i].theta;

		// calculate measurement
		if (fabs(yaw_rate) < 0.00001) { 
			particles[i].x += velocity * cos(theta0) * delta_t;
			particles[i].y += velocity * sin(theta0) * delta_t;
			// leave alone theta
		}
		else {
			particles[i].x += (velocity / yaw_rate) * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
			particles[i].y += (velocity / yaw_rate) * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// add measurement and noise to particle position
		std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	// find the observation with the minimum distance to predicted measurement
	for (int i = 0; i<observations.size(); ++i) {

		// Initialize minimum distance from observation to landmarks
		double dist_min = 0.0;
		// Id of matched landmark
		int id_m = 0;

		// Iterate over all landmarks
		for (int j = 0; j<predicted.size(); ++j) {

			// Find the distance from observation to landmark
			double dist_temp = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			// Set initial minimum distance and id of matched landmark
			if (j == 0) {
				dist_min = dist_temp;
				id_m = predicted[j].id;
			}

			// Find the minimum distance and corresponding landmark id
			if (dist_temp < dist_min) {
				dist_min = dist_temp;
				id_m = predicted[j].id;
			}
		}

		// Set the matched landmark id to the observation
		observations[i].id = id_m;
	}
}


static inline double normpdf(double x, double mu, double std) {
	return (1 / sqrt(2 * M_PI) / std)*exp(-0.5*pow((x - mu) / std, 2));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.htm

	// Declare a temporary weights vector
	std::vector<double> weights_temp;

	// Iterate over all particles
	for (int n = 0; n<num_particles; ++n) {

		/*******************************************************
		Transform observations to map coordinates
		*******************************************************/

		// Set values for better readability
		const double x = particles[n].x;
		const double y = particles[n].y;
		const double theta = particles[n].theta;

		// Declare a set of transformed observations
		std::vector<LandmarkObs> observations_trans(observations.size());

		// Transform all observations
		for (int i = 0; i<observations.size(); ++i) {
			// Transform: rotation and translation
			observations_trans[i].x = x + observations[i].x*cos(theta) - observations[i].y*sin(theta);
			observations_trans[i].y = y + observations[i].x*sin(theta) + observations[i].y*cos(theta);
		}

		/*******************************************************
		Find map landmarks within sensor range
		*******************************************************/

		// Declare a set of landmarks within sensor range
		std::vector<LandmarkObs> landmarksInRange;

		// Iterate over all landmarks in map
		for (size_t m = 0; m<map_landmarks.landmark_list.size(); ++m) {

			// Find the coordinates and id of the landmark
			const double x2 = map_landmarks.landmark_list[m].x_f;
			const double y2 = map_landmarks.landmark_list[m].y_f;
			const int id = map_landmarks.landmark_list[m].id_i;

			// Check if the landmark is within sensor range
			if (dist(x, y, x2, y2) <= sensor_range) {

				// Add the within-range landmark to the set
				LandmarkObs landmark;
				landmark.id = id;
				landmark.x = x2;
				landmark.y = y2;
				landmarksInRange.push_back(landmark);
			}
		}

		// Declare importance weight
		double prob = 1.0;

		if (landmarksInRange.size() > 0) {
			/*******************************************************
			Match observations with landmarks
			*******************************************************/

			dataAssociation(landmarksInRange, observations_trans);

			/*******************************************************
			Update importance weight
			*******************************************************/



			// Iterate over all observations
			for (int i = 0; i < observations_trans.size(); ++i) {

				// Get the coordinates of corresponding observation and landmark
				const double x_obs = observations_trans[i].x;
				const double y_obs = observations_trans[i].y;
				const int id_landmark = observations_trans[i].id - 1; // landmap list indicing from 0
				const double x_landmark = map_landmarks.landmark_list[id_landmark].x_f;
				const double y_landmark = map_landmarks.landmark_list[id_landmark].y_f;

				// Calculate the importance weight (measurement probability)
				prob *= normpdf(x_obs, x_landmark, std_landmark[0]);
				prob *= normpdf(y_obs, y_landmark, std_landmark[1]);
			}
		} 
		else {  // no landmarks around
			std::cout << "no landmarks aroud the particle " << n << std::endl;
			prob = 0.0;
		}

		// Update weight
		particles[n].weight = prob;
		// Add the weight to the temporary weights vector
		weights_temp.push_back(prob);
	}

	// Update the weights vector of particle filter
	weights = weights_temp;
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// create generator and discrete distribution
	std::default_random_engine gen;
	std::discrete_distribution<int> distribution(weights.begin(), weights.end());

	// resample particles
	std::vector<Particle> resampled_particles(num_particles);
	for (int i = 0; i < num_particles; ++i) {
		int index = distribution(gen);
		resampled_particles[i] = particles[index];
	}

	// assign resampled_particles to particles
	particles = resampled_particles;
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

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
