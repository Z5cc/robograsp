# Robotic Grasping via Reinforcement Learning

2025-11-24 by David Nicklaser  

The project simulates robotic grasping in pybullet. The setup consists of a UR5 robot arm combined with the Robotiq 2F-85 gripper, along with a depth camera mounted on the gripper. Reinforcement learning is carried out using DQN.  

One key finding is that a very low image resolution, such as 16×16, is sufficient for vision-based grasping. This suggests that sim-to-real may not be necessary in this case, and that a real robot could potentially learn the policy directly. Deployment in the real world can be organized as a two-stage pipeline. In the first stage, an object is detected using a method such as YOLO, and the gripper moves closer to it. After moving closer to the object, the second stage employs the policy learned in this project to grasp at an advantageous location of the object.  

## Installation

Clone the project and move into the directory:
```bash
git clone https://github.com/Z5cc/robograsp.git  
cd robograsp
```

Make sure you have *Python 3.9* installed and activated. You can check with:
```bash
python3 --version
```

Create a virtual environment, activate it and install the requirements:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the *train.py* file for reinforcement learning:
```bash
python3 train.py
```

Run the *user_control.py* file for being able to manually control the robot for testing purposes:
```bash
python3 user_control.py
```

To adjust parameters, modify the *constants.py* file. Setting *N_ACTIONS=7* disables rotational movements of the TCP and allows for much faster training. To run *user_control.py*, you need to set *VIS=True*.


## Environment

**State Space**

$S=\\{s \in \mathbb{R}^{C \times H \times W}\\}$  
The state space uses tensors of the form C×H×W.  
H×W corresponds to the size of the depth image and is set to 16×16.  
C represents the number of stacked depth images and is set to 4. After each step, a new observation in the form of a depth image of size H×W is returned. The state is then updated by shifting the stack by –1 and inserting the new observation in the last position. The stack is initialized by copying the first observation C times.  

**Action Space**

$A=\\{grasp,-x,+x,-y,+y,-z,+z,-roll,+roll,-pitch,+pitch,-yaw,+yaw\\}$  
The action space consists of 13 discrete actions, which can be grouped into grasp and seek actions.  
The first action, grasp, makes the gripper move forward until it hits something. Then the gripper closes. If the gripper detects that it is holding something while closing, it lifts as long as it continues to grip something. If not, the gripper reopens, retreats and continues with the next action.  
The remaining twelve seek actions move the TCP of the gripper in all translational directions in steps of 15 mm, and in all rotational directions in steps of 0.05 rad. All movements are defined relative to the TCP coordinate frame, not the world frame.  

**Rewards**

![equation](https://latex.codecogs.com/svg.image?$r(s)=%5Cbegin%7Bcases%7D100,&%5Ctext%7Bif%20object.z%7D%3E%5Ctext%7Bthreshold%7D%5C%5C0,&%5Ctext%7Botherwise%7D%5Cend%7Bcases%7D$)  
For the reward function a height threshold is set. If during a step the object reaches that threshold, a reward of 100 is given. Otherwise, the reward is 0. In addition to that, other reward functions involving distance and offset calculations were tested. However they did not mark any improvements. Also when incorporating potential based reward shaping according to *Andrew Y. Ng*, no improvement could be determined  for these new reward functions.  

## Algorithm

DQN is used for the algorithm because of its simplicity. The neural network for the policy and the target network is mainly built from convolutional layers, based on the following idea. Kernels with a high–low–high pattern can detect far–close–far structures in the image, which correspond to good grasp locations. The policy network, which takes a state $s \in \mathcal{S}$ as input and outputs an action $a \in \mathcal{A}$, is designed as follows:  

4x16x16 -> **conv(3)** -> 8x16x16 -> **pool(2)** -> 8x8x8 -> **conv(3)** -> 16x8x8 -> **conv(3)** -> 16x8x8 -> **Flatten** -> 1024 -> **FC** -> 13

## Demo

The demo shows the training process at around episode 700. Although the RGB camera view is displayed, only the depth camera is actually used.  

![gif](demo.gif)

![png](demo.png)


## Credits

**This project includes code licensed as follows:**

https://github.com/ElectronicElephant/pybullet_ur5_robotiq  
Original code: Copyright (c) 2021 ElectronicElephant, released under the BSD 2-Clause License.  

https://github.com/pytorch/tutorials/blob/main/intermediate_source/reinforcement_q_learning.py  
Original code: Copyright (c) 2017-2022 Pytorch contributors, released under the BSD 3-Clause License.

**If you use my project and this code in any form, please cite the following:**  

```bibtex
@misc{nicklaser2025robograsp,
    title={Robograsp: Robotic Grasping via Reinforcement Learning},
    author={Nicklaser, David},
    year={2025},
    url={https://github.com/Z5cc/robograsp}
}
```
