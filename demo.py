from Env import Env



def user_control_demo():

    env = Env()
    while True:
        debug_parameter = env.read_debug_parameter()
        delta, gr_delta = debug_parameter[0:6], debug_parameter[-1]

        obs = env.step_demo(delta,gr_delta)

        # print('\n\n\n','obsbeginn',obs,'obsend')
        # print('\n\n\n','shoulder_torque: ',env.robot.get_shoulder_torque())
        print('\n\n\n','gripper_torque: ',env.robot.gripper.get_torque())









user_control_demo()
