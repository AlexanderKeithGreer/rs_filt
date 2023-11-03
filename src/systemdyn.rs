use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::rand_distr::Gamma;
use ndarray_npy::*;

#[derive(Clone)]
pub struct SystemDyn {
    pub a_mat: ndarray::Array2<f64>,
    pub b_mat: ndarray::Array2<f64>,
    pub h_mat: ndarray::Array2<f64>,
    pub q_cov: ndarray::Array2<f64>, //must be diagonal
    pub r_cov: ndarray::Array2<f64>, //must be diagonal
}

impl SystemDyn {
    pub fn new (a_mat: ndarray::Array2<f64>,
            b_mat: ndarray::Array2<f64>,
            h_mat: ndarray::Array2<f64>,
            q_vec: ndarray::Array1<f64>,
            r_vec: ndarray::Array1<f64>) -> Self {

      if a_mat.shape()[0] != a_mat.shape()[1] ||
          a_mat.shape()[1] != b_mat.shape()[0] ||
          a_mat.shape()[0] != h_mat.shape()[1] ||
          a_mat.shape()[0] != q_vec.shape()[0] ||
          r_vec.shape()[0] != h_mat.shape()[0] {
        println!("Error: Shape Mismatch");
        Self {  a_mat: Array::zeros((2,2)),
                b_mat: Array::zeros((2,1)),
                h_mat: Array::zeros((1,2)),
                q_cov: Array::zeros((2,2)),
                r_cov: Array::zeros((2,2)),
                }
      } else {
        let q_cov = Array2::from_diag(&q_vec);
        let r_cov = Array2::from_diag(&r_vec);
        Self {  a_mat,
                b_mat,
                h_mat,
                q_cov,
                r_cov,
        }
      }
    }

    pub fn run (&self, n_steps: u64) ->
            (ndarray::Array2<f64>, ndarray::Array2<f64>) {
        let n_states = self.a_mat.shape()[0];
        let n_obs    = self.h_mat.shape()[0];

        let mut state = Array2::zeros((n_states, n_steps as usize));
        let mut meas = Array2::zeros((n_obs, n_steps as usize));
        let init_state: Array1<f64> = array![1.0, 0.0];
        state.slice_mut(s![..,0 as usize]).assign(&init_state);

        for i in 1..n_steps {
          //Multiply A by
          //let a     = Array::random((2,        5), Uniform::new(0., 10.));
          let mut u_mat = Array::random((n_states, 1), Normal::new(0., 1.).unwrap());
          u_mat = self.q_cov.dot(&u_mat);

          let mut upd_state = self.a_mat.dot(&state.slice(s![0..,(i-1) as usize]));
          upd_state += &(&self.b_mat * u_mat).slice(s![..,0]);

          let mut upd_meas = self.h_mat.dot(&(upd_state.t()));
          upd_meas += &(self.r_cov.dot(&Array::random((n_obs, 1),
                                       Normal::new(0., 1.).unwrap())))
                        .slice(s![..,0]);

          state.slice_mut(s![..,i as usize]).assign(&upd_state);
          meas.slice_mut(s![..,i as usize]).assign(&upd_meas);

        }

        (meas, state)
    }
}

#[derive(Clone)]
pub struct SystemDynGamma {
    pub a_mat: ndarray::Array2<f64>,
    pub b_mat: ndarray::Array2<f64>,
    pub h_mat: ndarray::Array2<f64>,
    pub q_cov: ndarray::Array2<f64>, //must be diagonal
    pub r_cov: ndarray::Array2<f64>, //must be diagonal
    pub k_set: ndarray::Array2<f64>, //2xN, where R1 is k, and N is time
}

impl SystemDynGamma {
  pub fn new (a_mat: ndarray::Array2<f64>,
          b_mat: ndarray::Array2<f64>,
          h_mat: ndarray::Array2<f64>,
          q_vec: ndarray::Array1<f64>,
          r_vec: ndarray::Array1<f64>,
          k_set: ndarray::Array2<f64>) -> Self {

    if a_mat.shape()[0] != a_mat.shape()[1] ||
        a_mat.shape()[1] != b_mat.shape()[0] ||
        a_mat.shape()[0] != h_mat.shape()[1] ||
        a_mat.shape()[0] != q_vec.shape()[0] ||
        r_vec.shape()[0] != h_mat.shape()[0] ||
        k_set.shape()[1] != 2 {
      println!("Error: Shape Mismatch");
      Self {  a_mat: Array::zeros((2,2)),
              b_mat: Array::zeros((2,1)),
              h_mat: Array::zeros((1,2)),
              q_cov: Array::zeros((2,2)),
              r_cov: Array::zeros((2,2)),
              k_set: Array::zeros((0,0)),
              }
    } else {
      let q_cov = Array2::from_diag(&q_vec);
      let r_cov = Array2::from_diag(&r_vec);
      Self {  a_mat,
              b_mat,
              h_mat,
              q_cov,
              r_cov,
              k_set,
      }
    }
  }

  pub fn run (&self, n_steps: u64) ->
        (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    let n_states = self.a_mat.shape()[0];
    let n_obs    = self.h_mat.shape()[0];

    let mut state = Array2::zeros((n_states, n_steps as usize));
    let mut meas = Array2::zeros((n_obs, n_steps as usize));
    let init_state: Array1<f64> = array![1.0, 0.0];
    state.slice_mut(s![..,0 as usize]).assign(&init_state);

    let n_gamma = self.k_set.shape()[0];
    let mut gamma_next = 0;
    let mut gamma = Gamma::new(1., self.r_cov[[0,0]].sqrt()).unwrap();

    for i in 1..n_steps {
      if self.k_set[[0,gamma_next]] < i as f64 + 0.5 &&
         self.k_set[[0,gamma_next]] > i as f64 - 0.5 {
        gamma = Gamma::new(self.k_set[[1,gamma_next]],
                           self.r_cov[[0,0]]).unwrap();
        if gamma_next+1 < n_gamma {
          gamma_next += 1;
        }
      }

      //Multiply A by
      //let a     = Array::random((2,        5), Uniform::new(0., 10.));
      let mut u_mat = Array::random((n_states, 1), Normal::new(0., 1.).unwrap());
      u_mat = self.q_cov.dot(&u_mat);

      let mut upd_state = self.a_mat.dot(&state.slice(s![0..,(i-1) as usize]));
      upd_state += &(&self.b_mat * u_mat).slice(s![..,0]);

      let mut upd_meas = self.h_mat.dot(&(upd_state.t()));
      upd_meas += &Array::random((n_obs, 1), gamma)
                    .slice(s![..,0]);

      state.slice_mut(s![..,i as usize]).assign(&upd_state);
      meas.slice_mut(s![..,i as usize]).assign(&upd_meas);
    }

    (meas, state)
  }
}
