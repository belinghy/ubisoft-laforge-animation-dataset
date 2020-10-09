import argparse

import numpy as np

from common.env_utils import make_env, EpisodeRunner


def play_mocap(args):
    mocap_data = np.concatenate([np.load(file) for file in args.files])

    env_options = {
        "env_id": "environments:MocapReplayEnv-v0",
        "num_parallel": args.num,
        "mocap_data": mocap_data,
    }

    env = make_env(**env_options)
    if args.render:
        env.create_viewer()

    runner_options = {
        "save": args.save,
        "use_ffmpeg": args.ffmpeg,
        "max_steps": args.len,
    }

    with EpisodeRunner(env, **runner_options) as runner:

        env.reset()

        while not runner.done:
            env.step(None)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Examples:\n"
            "   python play_mocap.py --files <PARSED_BVH_FILES>\n"
            "   (Remote) python play_mocap.py --files <PARSED_BVH_FILES> --len 1000 --render 0 --save 1\n"
            "   (Faster) python play_mocap.py --files <PARSED_BVH_FILES> --len 1000 --save 1 --ffmpeg 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--num", type=int, default=1)
    parser.add_argument("--len", type=int, default=float("inf"))
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--ffmpeg", type=int, default=0)
    args = parser.parse_args()

    play_mocap(args)
