import numpy as np
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import time
from dexpoint.utils.ycb_object_utils import YCB_ORIENTATION, load_ycb_object


def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(0.005)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.25])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=0, y=-0.5, z=0.5)
    viewer.set_camera_rpy(r=0, p=-0.3, y=-np.pi/2)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    robot: sapien.Articulation = loader.load("assets/robot/iiwa_barret.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    obj_name = "potted_meat_can"
    obj = load_ycb_object(scene, obj_name)
    pos = [0.12, 0.0, 0.2]
    orientation = YCB_ORIENTATION[obj_name]
    obj.set_pose(sapien.Pose(pos, orientation))
    # Set initial joint positions
    arm_init_qpos = [2.585925  ,  1.1749223 , -2.9670596 ,  1.9096962 ,  1.0489284 ,
        1.5962807 , -3.0543242]
    # arm_init_qpos = [0, 0, 0, 0, 0, 0]
    gripper_init_qpos = [0, 0, 0] * 2 + [0, 0]
    init_qpos = arm_init_qpos + gripper_init_qpos
    robot.set_qpos(init_qpos)

    targets = [[0.0, 0.0, 0.12, 0.0, 0.0, 0, 0, 0],
               [0.0, 0.0, 0.12, 0.0, 0.0, np.pi/2, 0, 0],
                [0.0, 0.0, 0.05, 0.0, 0.0, np.pi/2, 0, 0],
                [0.0, 0.0, 0.05, 0.0, 0.0, np.pi/2, 0.06, -0.0],
                [0.0, 0.0, 0.2, 0.0, 0.0, np.pi/2, 0.06, -0.06]]
    
    active_joints = robot.get_active_joints()
    for joint in active_joints:
        joint.set_drive_property(stiffness=200, damping=50)

    last_time = time.time()
    goal_idx = 0
    target_pos = np.zeros(8)
    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            # if balance_passive_force:
            #     qf = robot.compute_passive_force(
            #         gravity=True, 
            #         coriolis_and_centrifugal=True, 
            #     )
            #     robot.set_qf(qf)
            # print(robot.get_qpos())
            curr_qpos = robot.get_qpos()
            curr_time = time.time()
            # if curr_time - last_time > 2 and goal_idx < len(targets):
            #     target_pos = targets[goal_idx]
            #     last_time = curr_time
            #     goal_idx +=1
            # for joint_idx, joint in enumerate(active_joints):
            #     joint.set_drive_target(target_pos[joint_idx])    
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    demo(fix_root_link=True,
         balance_passive_force=True)


if __name__ == '__main__':
    main()