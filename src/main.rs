use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_npy::*;


struct SystemDyn {
    a_mat: ndarray::Array2<f64>,
    b_mat: ndarray::Array2<f64>,
    h_mat: ndarray::Array2<f64>,
    q_cov: ndarray::Array2<f64>, //must be diagonal
    r_cov: ndarray::Array2<f64>, //must be diagonal
}

impl SystemDyn {
    fn new (a_mat: ndarray::Array2<f64>,
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

    fn run (&self, n_steps: u64) ->
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

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let sysdyn = SystemDyn::new(
                    array![[1.,   0.],
                           [0.01, 1.]], //Assumes a 100Hz f_s
                    array![[1.],
                           [0.]],
                    array![[0., 1.]],
                    array![4., 0.],
                    array![30.] );
   let (meas, state) = sysdyn.run(1000);

   write_npy("data/meas.npy", &meas)?;
   write_npy("data/state.npy", &state)?;

    Ok(())
}
