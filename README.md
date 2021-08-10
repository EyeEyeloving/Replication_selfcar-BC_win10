# Background

The self-driving car trained by behavior cloning was provided on https://github.com/udacity/CarND-Behavioral-Cloning-P3. Provided that some open source projects were contributed by researchers and lack of maintaining, several mistakes always happen. In this project, I share my approach of making the self-driving car project from Udacity work on my laptop.

# Device

LEGION y7000p - GTX 1660 ti - Windows10

By the way, I had download *CUDA_v9.2* because that I had met the problem https://github.com/iperov/DeepFaceLab/issues/5242 and tried to make it flowing the solution provided. As for that why I input `nvidia-smi` and then show me that `CUDA Version: 11.1`, I think it is just because that I have updated the *Display Adapter* in *Device Adapter* after I downloaded *CUDA_v9.2* . Is it necessary? I haven't been sure.

```cmd
>>> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Tue_Jun_12_23:08:12_Central_Daylight_Time_2018
Cuda compilation tools, release 9.2, V9.2.148
>>> nvidia-smi
Mon Aug 09 16:03:09 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 457.49       Driver Version: 457.49       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 166... WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   47C    P8     4W /  N/A |    609MiB /  6144MiB |     11%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

```

# The Project from Github

- **the simulator from Udacity on Windows**
  https://github.com/udacity/self-driving-car-sim 
  *term1* is enough. 
- **CarND-Term1-Starter-Kit link** 
  https://github.com/udacity/CarND-Term1-Starter-Kit  
  I used its `environment-gpu.yml` to configure the virtual environment. But this project lacked files  `drive.py` and `model.h5`.
- **P4-BehavioralCloning link**
  https://github.com/SakshayMahna/P4-BehavioralCloning  
  I used its `drive.py` and `model.h5`, and it also provided `train.py` which it's lack in many other projects.

# Configuration Instructions

I utilized Anaconda3 for creating the virtual environment which Miniconda could do.

In China, the *Channel* I used provided as follow:

- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

then

```cmd
conda env create -f environment_EyeEyeloving.yml
```

# Operating Instructions

Activate the virtual environment in Anaconda Prompt, and then

```cmd
python drive.py model.h5
```

While the system threw a start signal, open `beta_simulator.exe`, then chose a `scene` and play `automatic mode`. Then, it works.

> scene 1 play well

# Troubleshooting

This is the procession when I searched for the approaches for environment configuration.

```cmd
conda env create -f environment-gpu.yml
```

The file `environments-gpu yml` is from *CarND-Term1-Starter-Kit*, however, I met the same trouble as follow https://github.com/udacity/CarND-Term1-Starter-Kit/issues/116

In this *Issue*, I follow one of the solutions https://github.com/udacity/CarND-Term1-Starter-Kit/issues/116#issuecomment-762464909 to rewrite `environment-gpu.yml` which is `environment-gpu_ver1.yml`

```cmd
conda env create -f environment-gpu_ver1.yml
```

and then

```cmd
python drive.py model.h5
```

However, it didn't work.

```cmd
ERROR: File "D:\Anaconda3\envs\carnd-term1\lib\site-packages\keras\engine\saving.py", line 273, in _deserialize_model
    model_config = json.loads(model_config.decode('utf-8'))
AttributeError: 'str' object has no attribute 'decode'
```

Latter, I also met the problem `Python: SystemError: Unknown opcode` which the reason I had forgotten, I followed the answer https://blog.csdn.net/Sarah_LZ/article/details/86526968 to try to solve it which said that python3.5 could make it.

So

```cmd
conda env create -f environment-gpu_ver2.yml
```

and *ERROR*

```cmd
ERROR: "GET /socket.io/?EIO=4&transport=websocket HTTP/1.1" 400 195 0.000576
```

There is a *Discussion* provided https://github.com/llSourcell/How_to_simulate_a_self_driving_car/issues/34 and I follow the solution https://github.com/llSourcell/How_to_simulate_a_self_driving_car/issues/34#issuecomment-741324365 .

```cmd
pip install -r requirements.txt
```

But that is not enough. During debug, system threw that *xxx==x.x.x didn't match*. During my project, the system warned that `opencv-python`, `matplolib`, `scikit-learn` and `scikit-image` met this *ERROR*. So, I just deleted the version provided in `requirements.txt`, like `matplotlib~=3.3.3` to `matplotlib`.

```cmd
pip install -r requirements_ver1.txt
```

threw *ERROR*

```cmd
'Keras requires TensorFlow 2.2 or higher. 'ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow`
```

and then

```cmd
pip uninstall tensorflowpip uninstall tensorflow-gpu
```

```cmd
pip install tensorflow
```

Unfortunately, I met the *ERROR*

```cmd
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.mkl-random 1.0.1 requires cython, which is not installed.
```

I chose to ignore it, and then

```cmd
pip install tensorflow-gpu
```

then

```cmd
python drive.py model.h5
```

and that work.

# Project Replication

```cmd
conda env create -f environment-gpu_ver2.yml
```

There was nothing wrong during the pkg collecting. However, after the Anaconda Prompt refreshed itself, there was a '\' keeping cycling and did not tell me whether the env created or not, but env file was already created. 

> 'someone could tell me why will be very appreciated'

I chose to close the Anaconda Prompt, and opened it again. Activate the virtual environment in Anaconda Prompt, and then

```cmd
pip install -r requirements_ver1.txt
```

and then

```cmd
pip uninstall tensorflowpip uninstall tensorflow-gpupip install tensorflowpip install tensorflow-gpu
```

and ignore the *ERROR*, then

```cmd
python drive.py model.h5
```

and that work.

# Copyright and Licensing Information

None
