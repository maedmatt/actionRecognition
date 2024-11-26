import os
import os.path as osp
import re
import numpy as np
from typing import Optional, Tuple
from types import SimpleNamespace

from mmengine import Config

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer


def get_output(
    video_path: str,
    out_filename: str,
    data_sample: str,
    labels: list,
    fps: int = 30,
    font_scale: Optional[float] = None,
    font_color: str = 'white',
    target_resolution: Optional[Tuple[int, int]] = None,
) -> None:
    """Get demo output using ``moviepy``.

    This function will generate a video or gif file from raw video or
    frames, using ``moviepy``. For more information on some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path.
        out_filename (str): Output filename for the generated file.
        data_sample (str): Predicted label of the generated file.
        labels (list): Label list of the current dataset.
        fps (int): Number of frames to read per second. Defaults to 30.
        font_scale (float): Font scale of the text. Defaults to None.
        font_color (str): Font color of the text. Defaults to 'white'.
        target_resolution (Tuple[int, int], optional): Set to
            (desired_width, desired_height) to resize frames. If
            either dimension is None, frames are resized while keeping
            the existing aspect ratio. Defaults to None.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    # Initialize visualizer
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)

  
    text_cfg = {
        'colors': font_color,
        'positions': np.array([10, 10]),  # Convert to NumPy array
        'font_sizes': 30 if font_scale is None else font_scale,
        'font_families': 'monospace',
    }
    if font_scale is not None:
        text_cfg.update({'font_sizes': font_scale})

    visualizer.add_datasample(
        out_filename,
        video_path,
        data_sample,
        draw_pred=True,
        draw_gt=False,
        text_cfg=text_cfg,
        fps=fps,
        out_type=out_type,
        out_path=osp.join('output', out_filename),
        target_resolution=target_resolution,
    )


def main():
    # Remove parse_args(), and define args directly
    args = SimpleNamespace()

    # Set your arguments here
    args.config = 'configs/config/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'  # Update this path
    args.checkpoint = 'configs/checkpoint/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'  # Update this path
    args.video = 'input/deadlifting.mp4'  # Update this path
    args.label = 'configs/label/label_map_k400.txt'  # Update this path

    # Optional arguments
    args.cfg_options = None
    args.device = 'cpu'  # Change to 'cuda:0' if you have a GPU
    args.fps = 30
    args.font_scale = None
    args.font_color = 'white'
    args.target_resolution = (720, -1)  # e.g., (width, height)
    args.out_filename = None  # Set to None initially

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    pred_result = inference_recognizer(model, args.video)

    # Extract the highest confidence level prediction
    pred_label = pred_result.pred_label.item()
    pred_score = pred_result.pred_score[pred_label].item()

    with open(args.label, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    result = (labels[pred_label], pred_score)

    print('The top label with corresponding score is:')
    print(f'{result[0]}: {result[1]}')

    # Ensure the label is safe to use as a filename
    safe_label = re.sub(r'[^\w\-_. ]', '_', labels[pred_label]).strip()
    safe_label = safe_label.replace(' ', '_')
    args.out_filename = f'{safe_label}.mp4'

    # Create the output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.out_filename is not None:
        if args.target_resolution is not None:
            width, height = args.target_resolution
            if width == -1:
                assert height > 0
            if height == -1:
                assert width > 0
            args.target_resolution = (width, height)

        get_output(
            args.video,
            args.out_filename,
            pred_result,
            labels,
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution,
        )


if __name__ == '__main__':
    main()