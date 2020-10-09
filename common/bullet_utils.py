import datetime
import functools
import inspect
import os
import sys
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

from matplotlib import cm as mpl_color
import numpy as np
import pybullet
import torch
import torch.nn.functional as F

from common.bullet_objects import VSphere, VCapsule


DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None, use_ffmpeg=False, fps=60):
        """Creates a Bullet client and connects to a simulation.

        Args:
          connection_mode:
            `None` connects to an existing simulation or, if fails, creates a
              new headless simulation,
            `pybullet.GUI` creates a new simulation with a GUI,
            `pybullet.DIRECT` creates a headless simulation,
            `pybullet.SHARED_MEMORY` connects to an existing simulation.
        """
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT

        options = (
            "--background_color_red=1.0 "
            "--background_color_green=1.0 "
            "--background_color_blue=1.0 "
            "--width=1280 --height=720 "
        )
        if use_ffmpeg:
            from datetime import datetime

            now_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            options += f'--mp4="{now_str}.mp4" --mp4fps={fps} '

        self._client = pybullet.connect(connection_mode, options=options)

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            if name not in [
                "invertTransform",
                "multiplyTransforms",
                "getMatrixFromQuaternion",
                "getEulerFromQuaternion",
                "computeViewMatrixFromYawPitchRoll",
                "computeProjectionMatrixFOV",
                "getQuaternionFromEuler",
            ]:  # A temporary hack for now.
                attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class Scene:
    "A base class for single- and multiplayer scenes"
    multiplayer = False

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

        self.test_window_still_open = True
        self.human_render_detected = False

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer:
            return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def set_physics_parameters(self):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.set_physics_parameters()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step()


class World:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.set_physics_parameters()

    def set_physics_parameters(self):
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.numSolverIterations,
            numSubSteps=self.frame_skip,
        )

    def step(self):
        self._p.stepSimulation()


class StadiumScene(Scene):

    stadium_halflen = 105 * 0.25
    stadium_halfwidth = 50 * 0.25

    def initialize(self, remove_ground=False):
        current_dir = os.path.dirname(__file__)

        if not remove_ground:
            filename = os.path.join(current_dir, "data", "misc", "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)

    def set_friction(self, lateral_friction):
        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=lateral_friction)


class Camera:
    def __init__(self, bc, fps=60, dist=2.5, yaw=0, pitch=-5, use_egl=False):

        self._p = bc
        self._cam_dist = dist
        self._cam_yaw = yaw
        self._cam_pitch = pitch
        self._coef = np.array([1.0, 1.0, 0.1])

        self.use_egl = use_egl

        self._fps = fps
        self._target_period = 1 / fps
        self._last_frame_time = time.perf_counter()

    def track(self, pos, smooth_coef=None):

        self.wait()

        smooth_coef = self._coef if smooth_coef is None else smooth_coef
        assert (smooth_coef <= 1).all(), "Invalid camera smoothing parameters"

        yaw, pitch, dist, lookat_ = self._p.getDebugVisualizerCamera()[-4:]
        lookat = (1 - smooth_coef) * lookat_ + smooth_coef * pos
        self._cam_target = lookat

        self._p.resetDebugVisualizerCamera(dist, yaw, pitch, lookat)

        # Remember camera for reset
        self._cam_yaw, self._cam_pitch, self._cam_dist = yaw, pitch, dist

    def lookat(self, pos):
        self._cam_target = pos
        self._p.resetDebugVisualizerCamera(
            self._cam_dist, self._cam_yaw, self._cam_pitch, pos
        )

    def dump_rgb_array(self):

        if self.use_egl:
            # use_egl
            width, height = 1920, 1080
            view = self._p.computeViewMatrixFromYawPitchRoll(
                self._cam_target, distance=4, yaw=0, pitch=-20, roll=0, upAxisIndex=2
            )
            proj = self._p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=100.0
            )
        else:
            # is_rendered
            width, height, view, proj = self._p.getDebugVisualizerCamera()[0:4]

        (_, _, rgb_array, _, _) = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
            flags=self._p.ER_NO_SEGMENTATION_MASK,
        )

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def wait(self):
        if self.use_egl:
            return

        time_spent = time.perf_counter() - self._last_frame_time
        time.sleep(max(self._target_period - time_spent, 0))
        self._last_frame_time = time.perf_counter()

        # Need this otherwise mouse control will be laggy when rendered
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)


class MocapViewer:

    Y_AXIS_UP = 1

    def __init__(
        self,
        env,
        num_characters=1,
        target_fps=30,
        use_params=True,
        camera_tracking=False,
        render_character_links=False,
        device="cpu",
    ):
        self.env = env
        self.num_characters = num_characters
        self.use_params = use_params

        self.device = device
        self.character_index = 0
        self.controller_autonomy = 1.0
        self.debug = False
        self.gui = False

        self.camera_tracking = camera_tracking
        # use 1.5 for close up, 3 for normal, 6 with GUI
        self.camera_distance = 6 if self.camera_tracking else 12
        self.camera_smooth = np.array([1, 1, 1])

        connection_mode = pybullet.GUI
        self._p = BulletClient(connection_mode=connection_mode, fps=target_fps)

        pVisualizer = lambda *args: self._p.configureDebugVisualizer(*args)
        pVisualizer(pybullet.COV_ENABLE_Y_AXIS_UP, self.Y_AXIS_UP)
        pVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pVisualizer(pybullet.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        pVisualizer(pybullet.COV_ENABLE_MOUSE_PICKING, 0)
        pVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        pVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pVisualizer(pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # Disable rendering during creation
        pVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self.camera = Camera(
            self._p, fps=target_fps, dist=self.camera_distance, pitch=-10, yaw=45
        )

        target_period = 1 / target_fps
        scene = StadiumScene(self._p, gravity=9.8, timestep=target_period, frame_skip=1)
        scene.initialize()

        if self.Y_AXIS_UP:
            ground_id = scene.ground_plane_mjcf[0]
            self._p.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [-1, 0, 0, 1])

        cmap = mpl_color.get_cmap("coolwarm")
        self.colours = cmap(np.linspace(0, 1, self.num_characters))

        if num_characters == 1:
            self.colours[0] = (0.376, 0.490, 0.545, 1)

        # here order is important for some reason ?
        # self.targets = MultiTargets(self._p, num_characters, self.colours)
        self.characters = MultiMocapCharacters(
            self._p, num_characters, self.colours, render_character_links
        )

        # Re-enable rendering
        pVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.state_id = self._p.saveState()

        if self.use_params:
            self._setup_debug_parameters()

    def reset(self):
        # self._p.restoreState(self.state_id)
        self.env.reset()

    def duplicate_character(self):
        characters = self.characters
        colours = self.colours
        num_characters = self.num_characters
        bc = self._p

        bc.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        if self.characters.has_links:
            for index, colour in zip(range(num_characters), colours):
                faded_colour = colour.copy()
                faded_colour[-1] = 1.0
                characters.heads[index].set_color(faded_colour)

                characters.links[index] = []
                # head = VSphere(bc, radius=0.12)
                # head.set_color(colour)
                # characters.heads[index] = head

        self.characters = MultiMocapCharacters(bc, num_characters, colours)
        bc.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def render(self, joint_coordinates, facings, root_xzs):

        # x, y, z = extract_joints_xyz(joint_coordinates, *self.joint_indices, dim=1)
        # mat = self.env.get_rotation_matrix(facings).to(self.device)
        # rotated_xy = torch.matmul(mat, torch.stack((x, y), dim=1))
        # poses = torch.cat((rotated_xy, z.unsqueeze(dim=1)), dim=1).permute(0, 2, 1)
        # root_xyzs = F.pad(root_xzs, pad=[0, 1])
        #
        # joint_xyzs = ((poses + root_xyzs.unsqueeze(dim=1)) * FOOT2METER).cpu().numpy()
        # self.root_xyzs = (
        #     (F.pad(root_xzs, pad=[0, 1], value=3) * FOOT2METER).cpu().numpy()
        # )
        # self.joint_xyzs = joint_xyzs

        coordinates = joint_coordinates.cpu().numpy()

        for index in range(self.num_characters):
            self.characters.set_joint_positions(coordinates[index], index)

        self._handle_mouse_press()
        self._handle_key_press()
        self._handle_parameter_update()
        if self.camera_tracking:
            character_root = coordinates[self.character_index, 0]
            self.camera.track(character_root, np.ones(3))
        else:
            self.camera.wait()

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    def close(self):
        self._p.disconnect()
        sys.exit(0)

    def _setup_debug_parameters(self):
        max_frame = self.env.mocap_data.shape[0]
        self.parameters = [
            {
                # -1 for random start frame
                "default": -1,
                "args": ("Start Frame", -1, max_frame, -1),
                "dest": (self.env, "debug_frame_index"),
                "func": lambda x: int(x),
                "post": lambda: self.env.reset(),
            },
            {
                "default": self.env.data_fps,
                "args": ("Target FPS", 1, 240, self.env.data_fps),
                "dest": (self.camera, "_target_period"),
                "func": lambda x: 1 / x,
            },
            {
                "default": 1,
                "args": ("Controller Autonomy", 0, 1, 1),
                "dest": (self, "controller_autonomy"),
                "func": lambda x: x,
            },
            {
                "default": 1,
                "args": ("Camera Track Character", 0, 1, int(self.camera_tracking)),
                "dest": (self, "camera_tracking"),
                "func": lambda x: x > 0.5,
            },
        ]

        if self.num_characters > 1:
            self.parameters.append(
                {
                    "default": 1,
                    "args": ("Selected Character", 1, self.num_characters + 0.999, 1),
                    "dest": (self, "character_index"),
                    "func": lambda x: int(x - 1.001),
                }
            )

        # setup Pybullet parameters
        for param in self.parameters:
            param["id"] = self._p.addUserDebugParameter(*param["args"])

    def _handle_parameter_update(self):
        if not self.use_params:
            return

        for param in self.parameters:
            func = param["func"]
            value = func(self._p.readUserDebugParameter(param["id"]))
            cur_value = getattr(*param["dest"], param["default"])
            if cur_value != value:
                setattr(*param["dest"], value)
                if "post" in param:
                    post_func = param["post"]
                    post_func()

    def _handle_mouse_press(self):
        events = self._p.getMouseEvents()
        for ev in events:
            if ev[0] == 2 and ev[3] == 0 and ev[4] == self._p.KEY_WAS_RELEASED:
                # (is mouse click) and (is left click)

                (
                    width,
                    height,
                    _,
                    proj,
                    _,
                    _,
                    _,
                    _,
                    yaw,
                    pitch,
                    dist,
                    target,
                ) = self._p.getDebugVisualizerCamera()

                pitch *= DEG2RAD
                yaw *= DEG2RAD

                R = np.reshape(
                    self._p.getMatrixFromQuaternion(
                        self._p.getQuaternionFromEuler([pitch, 0, yaw])
                    ),
                    (3, 3),
                )

                # Can't use the ones returned by pybullet API, because they are wrong
                camera_up = np.matmul(R, [0, 0, 1])
                camera_forward = np.matmul(R, [0, 1, 0])
                camera_right = np.matmul(R, [1, 0, 0])

                x = ev[1] / width
                y = ev[2] / height

                # calculate from field of view, which should be constant 90 degrees
                # can also get from projection matrix
                # d = 1 / np.tan(np.pi / 2 / 2)
                d = proj[5]

                A = target - camera_forward * dist
                aspect = height / width

                B = (
                    A
                    + camera_forward * d
                    + (x - 0.5) * 2 * camera_right / aspect
                    - (y - 0.5) * 2 * camera_up
                )

                C = np.array(
                    [
                        (B[2] * A[0] - A[2] * B[0]) / (B[2] - A[2]),
                        (B[2] * A[1] - A[2] * B[1]) / (B[2] - A[2]),
                        0,
                    ]
                )

                if hasattr(self.env, "reset_target"):
                    self.env.reset_target(location=C)

    def _handle_key_press(self, keys=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()

        RELEASED = self._p.KEY_WAS_RELEASED

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = not self.debug
        elif keys.get(ord("g")) == RELEASED:
            self.gui = not self.gui
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, int(self.gui))
        elif keys.get(ord("n")) == RELEASED:
            # doesn't work with pybullet's UserParameter
            self.character_index = (self.character_index + 1) % self.num_characters
            self.camera.lookat(self.root_xyzs[self.character_index])
        elif keys.get(ord("m")) == RELEASED:
            self.camera_tracking = not self.camera_tracking
        elif keys.get(ord("r")) == RELEASED:
            self.reset()
        elif keys.get(65280) == RELEASED:
            # F1
            from imageio import imwrite

            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            imwrite("{}.png".format(now), self.camera.dump_rgb_array())
        elif keys.get(65281) == RELEASED:
            # F2
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self._p.startStateLogging(
                self._p.STATE_LOGGING_VIDEO_MP4, "{}.mp4".format(now)
            )
        elif keys.get(ord(" ")) == RELEASED:
            self._p.configureDebugVisualizer(
                self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 0
            )
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
                elif keys.get(65280) == RELEASED:
                    self._handle_key_press(keys)


class MultiMocapCharacters:
    def __init__(self, bc, num_characters, colours=None, render_links=True):
        self._p = bc
        self.num_joints = 22
        total_parts = num_characters * self.num_joints

        # create all spheres at once using batchPositions
        # self.start_index = self._p.getNumBodies()
        # useMaximalCoordinates=True is faster for things that don't `move`
        joints = VSphere(bc, radius=0.07, max=True, replica=total_parts)
        self.ids = joints.ids
        self.render_links = render_links

        if render_links:
            self.linked_joints = np.array(
                [
                    [3, 4],  # left foot
                    [2, 3],  # left shin
                    [1, 2],  # left leg
                    [7, 8],  # right foot
                    [6, 7],  # right shin
                    [5, 6],  # right leg
                    [9, 0],  # spine 1
                    [10, 9],  # spine 2
                    [11, 10],  # spine 3
                    [12, 11],  # neck
                    [14, 15],  # left shoulder
                    [15, 16],  # left upper arm
                    [16, 17],  # left lower arm
                    [18, 19],  # right shoulder
                    [19, 20],  # right upper arm
                    [20, 21],  # right lower arm
                ]
            )

            self.links = {
                i: [
                    VCapsule(self._p, radius=0.06, height=0.1, rgba=colours[i])
                    for _ in range(self.linked_joints.shape[0])
                ]
                for i in range(num_characters)
            }
            self.z_axes = np.zeros((self.linked_joints.shape[0], 3))
            self.z_axes[:, 2] = 1

            # self.heads = [VSphere(bc, radius=0.12) for _ in range(num_characters)]

        if colours is not None:
            self.colours = colours
            for index, colour in zip(range(num_characters), colours):
                self.set_colour(colour, index)
                # if render_links:
                #     self.heads[index].set_color(colour)

    def set_colour(self, colour, index):
        # start = self.start_index + index * self.num_joints
        start = index * self.num_joints
        joint_ids = self.ids[start : start + self.num_joints]
        for id in joint_ids:
            self._p.changeVisualShape(id, -1, rgbaColor=colour)

    def set_joint_positions(self, xyzs, index):
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joints
        joint_ids = self.ids[start : start + self.num_joints]
        for xyz, id in zip(xyzs, joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))

        if self.render_links:
            rgba = self.colours[index].copy()
            rgba[-1] = 1.0

            deltas = xyzs[self.linked_joints[:, 1]] - xyzs[self.linked_joints[:, 0]]
            heights = np.linalg.norm(deltas, axis=-1)
            positions = xyzs[self.linked_joints].mean(axis=1)

            a = np.cross(deltas, self.z_axes)
            b = np.linalg.norm(deltas, axis=-1) + (deltas * self.z_axes).sum(-1)
            orientations = np.concatenate((a, b[:, None]), axis=-1)
            orientations[:, [0, 1]] *= -1

            for lid, (delta, height, pos, orn, link) in enumerate(
                zip(deltas, heights, positions, orientations, self.links[index])
            ):
                # 0.05 feet is about 1.5 cm
                if abs(link.height - height) > 0.05:
                    self._p.removeBody(link.id[0])
                    link = VCapsule(self._p, radius=0.06, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            # self.heads[index].set_position(0.5 * (xyzs[13] - xyzs[12]) + xyzs[13])

        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
