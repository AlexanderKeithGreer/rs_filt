use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use statrs::distribution;
use statrs::distribution::Continuous;
use crate::systemdyn::SystemDyn;
use crate::systemdyn::SystemDynGamma;

#[derive(Clone)]
pub struct ParticleSir<'sd> {
    sysdyn: &'sd SystemDyn,

    state: ndarray::Array1<f64>,
    weight: f64,
}

impl<'sd> ParticleSir<'sd> {
    pub fn new(sysdyn: &'sd SystemDyn, n: u64) -> Self {
        let n_states = sysdyn.a_mat.shape()[0];
        ParticleSir {
            sysdyn: sysdyn,
            state: Array::zeros(n_states),
            weight: 1.0/(n as f64) }
    }

    pub fn draw(&mut self) {
        let n_states = self.sysdyn.a_mat.shape()[0];

        let mut u_mat = Array::random((n_states, 1), Normal::new(0., 1.).unwrap());
        u_mat = self.sysdyn.q_cov.dot(&u_mat);

        let mut upd_state =
            self.sysdyn.a_mat.dot(&self.state);

        upd_state += &(&self.sysdyn.b_mat * u_mat).slice(s![..,0]);
        self.state = upd_state;
    }

    pub fn weight(&mut self, meas: f64) {
        // Weight is p(z|x)
        let n_obs    = self.sysdyn.h_mat.shape()[0];
        let part_meas = self.sysdyn.h_mat.dot(&(self.state.t() ))[0];
        let var_meas = self.sysdyn.r_cov[[n_obs-1,n_obs-1]];

        let p_z_given_x = distribution::Normal::new(part_meas, var_meas/0.01).unwrap();
        self.weight = p_z_given_x.pdf(meas);
    }

    pub fn give_phase_weight(&self) -> (f64, f64) {
        (self.state[1], self.weight)
    }

    pub fn give_phase(&self) -> f64 {
        self.state[1]
    }

    pub fn give_weight(&self) -> f64 {
        self.weight
    }

    pub fn normalise_weight(&mut self, total: f64) {
        self.weight = self.weight/total;
    }
}

#[derive(Clone)]
pub struct ParticleSirGamma<'sd> {
    sysdyn: &'sd SystemDynGamma,

    state: ndarray::Array1<f64>,
    weight: f64,
    gamma_var: f64,
    gamma: f64,
}

impl<'sd> ParticleSirGamma<'sd> {
    pub fn new(sysdyn: &'sd SystemDynGamma, n: u64) -> Self {
        let n_states = sysdyn.a_mat.shape()[0];
        let mut initial_state = Array::zeros(n_states);
        let jitter = -4.0 + sysdyn.r_cov[[0,0]] *
                     Array::random(1, Normal::new(0., 1.0).unwrap())[0];
        initial_state[n_states-1 as usize] = jitter;
        Self {
            sysdyn: sysdyn,
            state: initial_state,
            weight: 1.0/(n as f64),
            gamma_var: 0.06,
            gamma: 2.}
    }

    pub fn draw(&mut self) {
        let n_states = self.sysdyn.a_mat.shape()[0];

        let mut u_mat = Array::random((n_states, 1), Normal::new(0., 1.).unwrap());
        u_mat = self.sysdyn.q_cov.dot(&u_mat);

        let mut upd_state =
            self.sysdyn.a_mat.dot(&self.state);

        upd_state += &(&self.sysdyn.b_mat * u_mat).slice(s![..,0]);
        self.state = upd_state;
        self.gamma += self.gamma_var*Array::random(1, Normal::new(0., 1.0).unwrap())[0];
        if self.gamma < 0.05 {
            self.gamma = 0.05;
        } else if self.gamma > 15.0 {
            self.gamma = 15.0;
        }
    }

    pub fn weight(&mut self, meas: f64) {
        // Weight is p(z|x)
        let n_obs    = self.sysdyn.h_mat.shape()[0];
        let part_meas = self.sysdyn.h_mat.dot(&(self.state.t() ))[0];
        let var_meas = 1./self.sysdyn.r_cov[[0,0]];

        let p_z_given_x = distribution::Gamma::new(self.gamma, var_meas/1.0).unwrap();
        //println!("{} {}", part_meas, meas);
        self.weight = p_z_given_x.pdf(meas-part_meas) + 0.00001;
    }

    pub fn give_phase_weight(&self) -> (f64, f64) {
        (self.state[1], self.weight)
    }

    pub fn give_phase(&self) -> f64 {
        self.state[1]
    }

    pub fn give_weight(&self) -> f64 {
        self.weight
    }

    pub fn give_gamma(&self) -> f64 {
        self.gamma
    }

    pub fn give_w_phase(&self) -> f64 {
        self.state[1]*self.weight
    }

    pub fn give_w_gamma(&self) -> f64 {
        self.gamma*self.weight
    }

    pub fn normalise_weight(&mut self, total: f64) {
        self.weight = self.weight/total;
    }
}
