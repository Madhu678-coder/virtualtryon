#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted from IDM-VTON gradio_demo/apply_net.py for DensePose pose generation.

import argparse
import glob
import logging
import os
import sys
from typing import Any, ClassVar, Dict, List
import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

DOC = """Apply Net - a tool to print / visualize DensePose results"""
LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)
_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action:
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v", "--verbosity", action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument(
            "--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[], nargs=argparse.REMAINDER,
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace, human_img):
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        predictor = DefaultPredictor(cfg)
        context = cls.create_context(args, cfg)
        with torch.no_grad():
            outputs = predictor(human_img)["instances"]
            out_pose = cls.execute_on_outputs(context, {"image": human_img}, outputs)
        cls.postexecute(context)
        return out_pose

    @classmethod
    def setup_config(cls, config_fpath, model_fpath, args, opts):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


@register_action
class DumpAction(InferenceAction):
    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls, parser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument("--output", metavar="<dump_file>", default="results.pkl",
                            help="File name to save dump to")

    @classmethod
    def execute_on_outputs(cls, context, entry, outputs):
        result = {"file_name": entry.get("file_name", "")}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls, args, cfg):
        return {"results": [], "out_fname": args.output}

    @classmethod
    def postexecute(cls, context):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            torch.save(context["results"], hFile)


@register_action
class ShowAction(InferenceAction):
    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls, parser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument("visualizations", metavar="<visualizations>",
                            help="Comma separated list of visualizations")
        parser.add_argument("--min_score", metavar="<score>", default=0.8, type=float)
        parser.add_argument("--nms_thresh", metavar="<threshold>", default=None, type=float)
        parser.add_argument("--texture_atlas", metavar="<texture_atlas>", default=None)
        parser.add_argument("--texture_atlases_map", metavar="<texture_atlases_map>", default=None)
        parser.add_argument("--output", metavar="<image_file>", default="outputres.png")

    @classmethod
    def setup_config(cls, config_fpath, model_fpath, args, opts):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(cls, context, entry, outputs):
        import cv2
        import numpy as np
        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        return image_vis

    @classmethod
    def postexecute(cls, context):
        pass

    @classmethod
    def _get_out_fname(cls, entry_idx, fname_base):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls, args, cfg):
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg, texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        return {
            "extractor": extractor, "visualizer": visualizer,
            "out_fname": args.output, "entry_idx": 0,
        }


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser
