# Locally and Spatially Uniform Linear Operator Construction with Kalman Filter
- This repository w.r.t. locally and spatially uniform linear operator construction with Kalman filter (LSLOCK), which enable us to estimate states and state transition of linear Gaussian state space model in real-time.
- This repository includes the method in `src` directory, examples of application in `examples` directory.

# Examples
- We provide two examples of this method.
    1. global flow data: objects flows each direcation in each interval
        - generation process is seen at `examples/data_global_flow.py`
        - application to our method is checke at `examples/global_flow.py`
    2. concentric circle wave data: a wave propagates concentrically from a source point
        - generation process is seen at `examples/data_concentric_circle_wave.py`
        - application to our method is checke at `examples/concentric_circle_wave.py`

# Packages
- Our codes use `Python3.7.3` and `numpy 1.17`