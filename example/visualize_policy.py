import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dexpoint.env.rl_env.relocate_env import AllegroRelocateRLEnv
from dexpoint.env.rl_env.relocate_env_iiwa import UhvatRelocateRLEnv
from dexpoint.real_world import task_setting

import argparse
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP

def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "relocate-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in task_setting.IIWA_IMG_CONFIG['relocate_robot_only'].keys()],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs

def create_env_fn():
    object_names = ["mustard_bottle", "tomato_soup_can", "potted_meat_can"]
    object_name = np.random.choice(object_names)
    rotation_reward_weight = 0  # whether to match the orientation of the goal pose
    use_visual_obs = True
    object_name='potted_meat_can'
    # object_category="02876657"
    env_params = dict(robot_name="uhvat_prl_hand_iiwa14", object_name=object_name, rotation_reward_weight=rotation_reward_weight,
                        randomness_scale=1, use_visual_obs=use_visual_obs, use_gui=True,
                        no_rgb=False)

    # If a computing device is provided, designate the rendering device.
    # On a multi-GPU machine, this sets the rendering GPU and RL training GPU to be the same,
    # based on "CUDA_VISIBLE_DEVICES".
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    environment = UhvatRelocateRLEnv(**env_params)

    # Create camera
    environment.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

    # Specify observation
    environment.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])

    # Specify imagination
    environment.setup_imagination_config(task_setting.IIWA_IMG_CONFIG["relocate_robot_only"])
    return environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task_name', type=str, required=True)
    # parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--use_test_set', dest='use_test_set', action='store_true', default=False)
    args = parser.parse_args()
    # task_name = args.task_name
    use_test_set = args.use_test_set
    checkpoint_path = "assets/checkpoints/20250127-2138_jaw/model_5000.zip"#args.checkpoint_path

    env = create_env_fn()

    policy = PPO.load(checkpoint_path, env, 'cuda:0',
                      policy_kwargs=get_3d_policy_kwargs(extractor_name='mediumpn'),
                      check_obs_space=False, force_load=True)

    while True:
        obs = env.reset()

        for j in range(env.horizon):
            if isinstance(obs, dict):
                for key, value in obs.items():
                    obs[key] = value[np.newaxis, :]
            else:
                obs = obs[np.newaxis, :]
            action = policy.predict(observation=obs,    deterministic=True)[0][0]
            obs, reward, done, _ = env.step(action)
            print(f"reward = {reward}")
            env.render()
            if done:
                break
