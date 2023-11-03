use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::rand_distr::Uniform;
use ndarray_npy::*;

mod particle;
mod systemdyn;

const NITER: usize = 1000;
const NPART: usize = 320;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let sysdyn = systemdyn::SystemDynGamma::new(
                            array![[1.,   0.],
                                  [0.01, 1.]], //Assumes a 100Hz f_s
                            array![[0.1],
                                  [1.]],
                            array![[0., 1.]],
                            array![4., 0.],
                            array![10.] ,
                            array![[1., 500.],
                                   [2., 3.5]],);
   let (meas, state) = sysdyn.run(NITER as u64);

   let mut pars = vec![particle::ParticleSirGamma::new(&sysdyn, NPART as u64); NPART];
   let mut pars_copy = vec![particle::ParticleSirGamma::new(&sysdyn,
                                                            NPART as u64); NPART];
   let mut pars_res: ndarray::Array2<f64> = Array::zeros((2, NPART));
   let mut pars_csw: ndarray::Array1<f64> = Array::zeros(NPART);
   let mut pars_gma: ndarray::Array2<f64> = Array::zeros((NITER, NPART));

   let mut pars_ev: ndarray::Array2<f64> = Array::zeros((NITER, NPART));

   let mut mean_res: ndarray::Array1<f64> = Array::zeros(NITER);
   let mut mean_gma: ndarray::Array1<f64> = Array::zeros(NITER);


   for iter in 0..NITER {
    let mut total_weight: f64 = 0.0;
    //Draw and Weight
    for par in 0..NPART {
      pars[par].draw();
      pars[par].weight(meas[[0,iter]]);

      total_weight += pars[par].give_weight();
      pars_csw[par] = total_weight;
    }
    //Normalise
    for par in 0..NPART {
      pars[par].normalise_weight(total_weight);
      (pars_res[[0,par]], pars_res[[1,par]]) = pars[par].give_phase_weight();
      pars_csw[par] = pars_csw[par]/total_weight;
    }
    pars_copy = pars.clone();

    //Resample
    if iter > 1 {
      let comp_start = Array::random(1, Uniform::new(0., 1./(NPART as f64) ))[0];
      for par in 0..NPART {
        let comp_csw = comp_start + 1./(NPART as f64) * (par as f64);
        let mut comp_par = 0;
        while comp_csw > pars_csw[comp_par] {
          comp_par += 1;
        }

        pars[par] = pars_copy[comp_par].clone();
      }
    }

    for par in 0..NPART {
      pars_ev[[iter,par]] = pars[par].give_phase();
      pars_gma[[iter,par]] = pars[par].give_gamma();
      mean_res[iter] += pars[par].give_w_phase();
      mean_gma[iter] += pars[par].give_w_gamma();
    }
    if mean_res[iter] > 1e6 {
      mean_res[iter] = 1e6
    }
    if mean_gma[iter] > 1e6 {
      mean_gma[iter] = 1e6
    }
  }


   write_npy("data/meas.npy", &meas)?;
   write_npy("data/state.npy", &state)?;
   write_npy("data/par_res.npy", &pars_res)?;
   write_npy("data/pars_ev.npy", &pars_ev)?;
   write_npy("data/pars_gma.npy", &pars_gma)?;
   write_npy("data/mean_res.npy", &mean_res)?;
   write_npy("data/mean_gma.npy", &mean_gma)?;

    Ok(())
}
