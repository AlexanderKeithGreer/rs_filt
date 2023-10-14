import numpy as np
import matplotlib.pyplot as plt

def test_scatter():
    data = np.array([[0.5, 0.0,-0.5],
                     [1.0, 0.5, 0.0],
                     [1.5, 1.0, 0.5],
                     [2.0, 1.5, 1.0]])

    axis = np.floor(np.linspace(0,np.shape(data)[0]-1e-6,data.size))
    plt.scatter(axis, data.flatten())
    print(np.linspace(0,np.shape(data)[0]-1e-6,data.size))
    print(axis)
    print(data.flatten())
    plt.show()

def main():
    meas = np.load('data/meas.npy')
    state = np.load('data/state.npy')
    pars_res = np.load('data/par_res.npy')
    pars_ev = np.load('data/pars_ev.npy')

    print(pars_res)
    print("State:")
    print(pars_res[0,:])
    print("Weight:")
    print(pars_res[1,:])

    plt.figure()
    plt.plot(meas[0,:])
    plt.plot(state[1,:])

    #Use to print the weights for given states
    plt.figure()
    sortorder = pars_res[0,:].argsort()
    plt.plot(pars_res[0,sortorder],pars_res[1,sortorder],marker='+')

    plt.figure()
    axis = np.floor(np.linspace(0,np.shape(pars_ev)[0]-1e-6,pars_ev.size))
    plt.scatter(axis, pars_ev.flatten())
    plt.plot(state[1,:],color='r')


    plt.show()

#test_scatter()
main()
