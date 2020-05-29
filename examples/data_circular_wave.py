import os, time
import numpy as np


def wave_func(wave_num=10, center=100, amp_low=5, amp_up=20, 
              velo_low=0.01, velo_up=0.1, seed=1):
    np.random.seed(seed)
    phases = np.random.uniform(0, 2*3.141562, size=wave_num)
    amps = np.random.uniform(amp_low, amp_up, size=wave_num)
    velos = np.random.uniform(velo_low, velo_up, size=wave_num)
    def f(t):
        return center + np.sum(amps*np.sin(np.outer(t, velos) + phases), axis=1)
    return f


def generate_ccw_data(func, sink=np.zeros(2), Nx=30, Ny=30, T=1000, sd=20, v=1, seed=1):
    np.random.seed(seed)
    coord = np.transpose(np.meshgrid(np.arange(Ny), np.arange(Nx)), (2,1,0))
    dist = np.sqrt(np.sum((coord - sink)**2, axis=-1))
    true = np.zeros((T, Nx, Ny))
    
    for t in range(T):
        true[t] = func(v*(t - dist).reshape(-1)).reshape(Nx, Ny)
    
    obs = np.maximum(true + np.random.normal(0, sd, size=(T, Nx, Ny)), 0)
    return true, obs


def main():
    data_root = "../data/concentric_circle_wave"

    wave_num=10; center=120
    amp_low=5; amp_up=20
    velo_low=0.1; velo_up=0.15
    T=1000; Nx=30; Ny=30; sd=0.1
    sink=np.array([-15, 14])

    f = wave_func(wave_num, center, amp_low, amp_up, velo_low, velo_up)
    true, obs = generate_ccw_data(f, sink, Nx, Ny, T, sd)

    np.save(os.path.join(data_root, "obs.npy"), obs)
    np.save(os.path.join(data_root, "true.npy"), true)


if __name__ == "__main__":
    main()