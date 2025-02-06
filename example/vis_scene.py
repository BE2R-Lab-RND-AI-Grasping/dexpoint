import os
from time import time

import numpy as np
import open3d as o3d

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.env.rl_env.relocate_env_iiwa import UhvatRelocateRLEnv
from dexpoint.real_world import task_setting
from sapien.utils import Viewer

if __name__ == '__main__':
    def create_env_fn():
        object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
        object_name = "mustard_bottle" #np.random.choice(object_names)
        rotation_reward_weight = 0  # whether to match the orientation of the goal pose
        use_visual_obs = True
        env_params = dict(object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                          randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=True,
                          no_rgb=True, frame_skip=10)

        # If a computing device is provided, designate the rendering device.
        # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
        # based on "CUDA_VISIBLE_DEVICES".
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_params["device"] = "cuda"
        environment = UhvatRelocateRLEnv(**env_params)
        # environment = AllegroRelocateRLEnv(**env_params)

        # Create camera
        environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

        # Specify observation
        environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

        # Specify imagination
        environment.setup_imagination_config(task_setting.IIWA_IMG_CONFIG["relocate_goal_robot"])
        return environment


    env = create_env_fn()
    base_env = env
    print("Observation space:")
    print(env.observation_space)
    print("Action space:")
    print(env.action_space)

    obs = env.reset()
    print("For state task, observation is a numpy array. For visual tasks, observation is a python dict.")

    print("Observation keys")
    print(obs.keys())
    
    viewer = Viewer(base_env.renderer)
    viewer.set_scene(base_env.scene)
    base_env.viewer = viewer

    viewer.toggle_pause(True)
    
    tic = time()
    rl_steps = 1000
    for _ in range(rl_steps):
        action = np.zeros(env.action_space.shape)
        action[0] = 0.000  # Moving forward ee link in x-axis
        obs, reward, done, info = env.step(action)
    while not viewer.closed:
        env.render()
    elapsed_time = time() - tic

    pc = obs["relocate-point_cloud"]
    print('pc shape',pc.shape)
    # The name of the key in observation is "CAMERA_NAME"-"MODALITY_NAME".
    # While CAMERA_NAME is defined in task_setting.CAMERA_CONFIG["relocate"], name is point_cloud.
    # See example_use_multi_camera_visual_env.py for more modalities.

    simulation_steps = rl_steps * env.frame_skip
    print(f"Single process for point-cloud environment with {rl_steps} RL steps "
          f"(= {simulation_steps} simulation steps) takes {elapsed_time}s.")
    print("Keep in mind that using multiple processes during RL training can significantly increase the speed.")
    env.scene = None

    # Note that in the DexPoint paper, we never use "imagination_goal" but only "imagination_robot"
    goal_pc = obs["imagination_goal"]
    goal_robot = obs["imagination_robot"]
    imagination_goal_cloud = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(goal_pc))
    imagination_goal_cloud.paint_uniform_color(np.array([0, 1, 0]))
    imagination_robot_cloud = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(goal_robot))
    imagination_robot_cloud.paint_uniform_color(np.array([0, 0, 1]))

    obs_cloud = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pc))
    obs_cloud.paint_uniform_color(np.array([1, 0, 0]))
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([imagination_goal_cloud, imagination_robot_cloud, coordinate, obs_cloud])
    print('obs shape:', pc.shape)

    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     width=640,
    #     height=480,
    #     fx=525.0,  # Focal length in x
    #     fy=525.0,  # Focal length in y
    #     cx=319.5,  # Optical center x
    #     cy=239.5   # Optical center y
    # )

    # # Project the point cloud to a depth image
    # # depth_image = obs_cloud.project_to_depth_image(intrinsic)
    # colors = np.asarray(imagination_goal_cloud.colors) * 255  # Scale colors to [0,255]
    # image = np.zeros((480, 640, 3), dtype=np.uint8)  # Create an empty image

    # for i in range(len(imagination_goal_cloud.points)):
    #     x = int(imagination_goal_cloud.points[i][0])  # Assuming x corresponds to pixel columns
    #     y = int(imagination_goal_cloud.points[i][1])  # Assuming y corresponds to pixel rows
    #     if (0 <= x < 640) and (0 <= y < 480):
    #         image[y, x] = colors[i]

    # # Save or visualize the image
    # o3d.io.write_image("output_image.png", o3d.geometry.Image(image))
    env.scene = None
