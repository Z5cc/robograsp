from Env import make_env



def user_control_demo():

    env = make_env()
    while True:
        debug_parameter = env.read_debug_parameter()
        delta, gr_delta = debug_parameter[0:6], debug_parameter[-1]

        obs = env.step_move(delta, gr_delta)








user_control_demo()
