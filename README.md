# Locally and Spatially Uniform Linear Operator Construction with the Kalman Filter
- This repository w.r.t. locally and spatially uniform linear operator construction with the Kalman filter (LSLOCK), which enable us to estimate states and state transition of linear Gaussian state space model in real-time.
- This repository includes the method in `src` directory, examples of application in `examples` directory.
- For ConvLSTM, we started [this implementation](https://github.com/spacejake/convLSTM.pytorch) and changed structure in order to compare w/ the prposed method.

# Examples
- We provide two examples of this method at `example` directory.
    1. global flow data: objects flow each direction in each interval
        - the generation process is condeucted by `python data_global_flow.py`
        - application of the proposed method is conducted by `python global_flow.py`
        - application of ConvLSTM is conducted by `python run_convlstm.py`
    2. circular wave data: a wave propagates concentrically from a source point
        - the generation process is conducted by `python data_circular_wave.py`
        - application of our method is conducted by `python circular_wave.py`
        - application of ConvLSTM is conducted by `python run_convlstm.py`

# Packages
- The proposed method uses `Python3.7`
- ConvLSTM uses `Python3.7` and `pytorch 1.2.0`
    - If you do not conduct `run_convlstm.py`, not need to install `pytorch`
