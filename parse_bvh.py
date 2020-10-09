import argparse
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np

from common.bvh_utils import load


def parse_bvh_files(args):
    raw_mocap_clips = [(*load(f), f) for f in args.files]

    def traverse_nodes(node, coordinates=None):
        if coordinates is not None:
            coordinates.append(node.coordinates.squeeze())
        for child in node.children:
            traverse_nodes(child, coordinates)

    progress_bar = None
    try:
        from tqdm import tqdm

        num_frames = np.sum([len(clip[1]) for clip in raw_mocap_clips])
        progress_bar = tqdm(total=num_frames)
    except ImportError:
        pass

    mocap_data = []
    for root, frames, period, file in raw_mocap_clips:
        print(f"Parsing: {file}")

        clip_joint_coordinates = []
        for frame in frames:
            root.load_frame(frame)
            root.apply_transformation()
            frame_joint_coords = []
            traverse_nodes(root, frame_joint_coords)
            clip_joint_coordinates.append(frame_joint_coords)
            progress_bar.update(1)

        clip_joint_coordinates = np.array(clip_joint_coordinates, dtype=np.float32)
        mocap_data.append(clip_joint_coordinates)

    # Convert to meters
    mocap_data = np.concatenate(mocap_data, axis=0) / 100

    if args.output is not None:
        np.save(args.output, mocap_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Examples:\n\tpython parse_bvh.py --files <BVH_FILES>\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--files", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    parse_bvh_files(args)
