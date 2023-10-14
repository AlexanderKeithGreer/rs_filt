use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

use statrs::distribution;
use statrs::distribution::Continuous;
use crate::SystemDyn;

#[derive(Clone)]
pub struct ParticleSir {
    sysdyn: SystemDyn,

    state: ndarray::Array1<f64>,
    weight: f64,
}

impl ParticleSir {
    pub fn new(sysdyn: SystemDyn, n: u64) -> Self {
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

        let p_z_given_x = distribution::Normal::new(part_meas, var_meas/10.0).unwrap();
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
