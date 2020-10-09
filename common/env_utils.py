import datetime
import inspect
import os
import types

import gym
import numpy as np


def make_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    return env


class EpisodeRunner(object):
    def __init__(
        self, env, save=False, use_ffmpeg=False, dir=None, max_steps=None, csv=None
    ):
        self.env = env
        self.save = save
        self.use_ffmpeg = use_ffmpeg
        self.csv = csv

        self.max_steps = max_steps or float("inf")
        self.done = False
        self.step = 0

        base_dir = os.path.dirname(os.path.realpath(inspect.stack()[-1][1]))
        self.dump_dir = dir or os.path.join(base_dir, "dump")
        if self.csv is not None:
            self.csv = os.path.join(base_dir, self.csv)

        self.override_reset()  # add progress bar
        self.override_step()  # save frame and csv data

        if self.csv is not None:
            self.csv_data_buffer = {}

        if self.save:
            self.rgb_buffer = []
            self.max_steps = max_steps or env.max_timestep

            now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.filename = os.path.join(self.dump_dir, f"{now_str}.mp4")
            print("\nRecording... Close to terminate recording.")

        self.progress_bar = None
        if self.max_steps != float("inf"):
            try:
                from tqdm import tqdm

                self.progress_bar = tqdm(total=self.max_steps)
            except ImportError:
                pass

    def override_reset(self):
        old_reset_func = self.env.reset
        runner = self

        def new_reset(self, **kwargs):
            if kwargs.get("reset_runner", True):
                runner.step = 0
                if runner.progress_bar:
                    runner.progress_bar.reset()

            kwargs.pop("reset_runner", None)
            return old_reset_func(**kwargs)

        self.env.reset = types.MethodType(new_reset, self.env)

    def override_step(self):

        old_step_func = self.env.step
        runner = self

        def new_step(self, *args):
            ret = old_step_func(*args)

            if runner.use_ffmpeg:
                pc = self.camera._p
                pc.configureDebugVisualizer(pc.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            else:
                runner.store_current_frame()

            runner.save_csv_render_data()
            runner.step += 1

            if runner.progress_bar is not None:
                runner.progress_bar.update(1)

            if runner.step >= runner.max_steps:
                runner.done = True
                if runner.progress_bar is not None:
                    runner.progress_bar.close()

            return ret

        self.env.step = types.MethodType(new_step, self.env)

    def store_current_frame(self):
        if self.save:
            image = self.env.camera.dump_rgb_array()
            self.rgb_buffer.append(image)

    def save_csv_render_data(self):
        if self.csv is not None:
            render_data_dict = self.env.dump_additional_render_data()
            for file, render_data in render_data_dict.items():
                header = render_data["header"]
                data = render_data["data"].clone().cpu().numpy()
                once = render_data.get("once", False)

                if file not in self.csv_data_buffer:
                    list_data = data if once else []
                    self.csv_data_buffer[file] = {"header": header, "data": list_data}

                if not once:
                    self.csv_data_buffer[file]["data"].append(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.save and len(self.rgb_buffer) >= self.max_steps:
            import moviepy.editor as mp

            if not os.path.exists(self.dump_dir):
                print(f"Creating directory {self.dump_dir}")
                os.makedirs(self.dump_dir)

            clip = mp.ImageSequenceClip(self.rgb_buffer, fps=1 / self.env.control_step)
            clip.write_videofile(self.filename)

        if self.csv is not None:
            for file, data_dict in self.csv_data_buffer.items():
                np.savetxt(
                    os.path.join(self.csv, file),
                    np.asarray(data_dict["data"]),
                    delimiter=",",
                    header=data_dict["header"],
                    comments="",
                )

        self.env.close()
