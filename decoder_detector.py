import torch
import torchvision
import multiprocessing
import fire
import cv2
from typing import Any, Dict, List, Tuple, Union
import sys, os, time
from gevent import monkey
import traceback
import PyNvCodec as nvc

from fvcore.common.timer import Timer
import numpy as np
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient import utils
from configs import get_cfg, get_logger
import uuid
import shared_numpy as snp

from worker_decoder import Worker_decoder
from worker_detector import Worker_detector

logger = get_logger()
# init default config and merge from base.yaml
# default values configs/__init__.py
cfg = get_cfg("configs/DECODER.yaml")
# monkey.patch_all()
model_w = 640
model_h = 384
model_c = 3
input_w = 1280
input_h = 720


# model_c = 1

def start_work(verbose=False):
    logger.debug("===============")
    for k, v in cfg.items():
        logger.debug(f"k is:{k}, v is :{v}")
    logger.debug("===============")

    # os._exit(0)
    cameralist = cfg.CAMERA_LIST
    available_gpus = cfg.GPU_ASSIGNED_LIST
    detector_model_name = cfg.DETECTOR_MODEL_NAME
    faceid_model_name = cfg.FACE_ID_MODEL_NAME
    server_url = cfg.SERVER_URL
    outputdim = cfg.OUTPUT_DIM
    model_w, model_h, model_c = cfg.MODEL_W, cfg.MODEL_H, cfg.MODEL_C
    inputtensorname, outputtensorname = cfg.INPUT_TENSOR_NAME, cfg.OUTPUT_TENSOR_NAME
    verbose_output_filename = cfg.OUTPUT_MEDIA_FILENAME
    assert len(cameralist) > 0
    assert len(detector_model_name) > 0
    assert len(faceid_model_name) > 0
    assert len(available_gpus) > 0

    processes = list()

    q = snp.Queue(maxsize=25 * 30)  # Build a shared memory queue per camera ... this is a test so one queue
    det_q = snp.Queue(maxsize=9999)
    # for i in range(len(cameralist)):
    for i in range(0, 1):
        p = Worker_decoder(queue=q, idx=i, rtsp=cameralist[i], gpu_id=available_gpus[i], verbose=verbose)
        pp = Worker_detector(queue=q, det_q=det_q, idx=i, model_name=detector_model_name, server_url=server_url,
                             verbose=verbose, outputdim=outputdim, model_w=model_w, model_h=model_h,
                             outputtensorname=outputtensorname, inputtensorname=inputtensorname,
                             verbose_output_filename=verbose_output_filename, model_c=model_c)
        # pp=multiprocessing.Process(target=test_con,args=(q,))
        p.start()
        pp.start()
        processes.append(p)
        processes.append(pp)

        # while 1:
        #     time.sleep(99)
    [proc.join() for proc in processes]
    # p.join()


if __name__ == "__main__":
    fire.Fire()
