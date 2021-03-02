import multiprocessing
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch, torchvision
from configs import get_cfg, get_logger
import tritonclient.http as httpclient
import tritonclient.utils.cuda_shared_memory as cudashm
import time
from fvcore.common.timer import Timer
import shared_numpy as snp
import cv2
import sys
from tritonclient import utils
import traceback

logger = get_logger()
# init default config and merge from base.yaml
# default values configs/__init__.py
cfg = get_cfg("configs/DECODER.yaml")
model_w = 640
model_h = 384
model_c = 3
input_w = 1280
input_h = 720


class Worker_detector(multiprocessing.Process):
    def __init__(self, queue: snp.Queue, det_q: snp.Queue, idx, **kwargs):
        super(Worker_detector, self).__init__()
        self.queue = queue
        self.det_q = det_q
        self.idx = idx
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name: str, value: Any) -> None:
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def get(self, name: str) -> Any:
        return self._fields[name]

    def preprocess(self, img, c, h, w):
        frame = np.copy(img)
        # frame.resize(c, h, w)
        frame = frame.reshape((c, h, w))
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = np.transpose(frame, (2,1,0))

        # frame = frame[::-1, :, :]
        # surface_tensor.resize_(model_c, model_h, model_w)
        frame = np.ascontiguousarray(frame / 255., dtype=np.float32)

        # return torch.as_tensor((img / 255.), dtype=torch.float32 if dtype == "FP32" else torch.int8)
        return frame

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        intput_w = cfg.BUSI_TYPE_LIST.BUSI_A.INPUT_W
        input_h = cfg.BUSI_TYPE_LIST.BUSI_A.INPUT_H
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        r_w = intput_w / origin_w
        r_h = input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (intput_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (intput_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # print(pred.shape)
        # to a torch Tensor
        # pred = torch.Tensor(pred).cuda()
        pred = torch.Tensor(pred).cpu()
        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4]
        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        conf_thresh = cfg.BUSI_TYPE_LIST.BUSI_A.CONF_THRESH
        iou_thresh = cfg.BUSI_TYPE_LIST.BUSI_A.IOU_THRESHOLD
        si = scores > conf_thresh
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)
        # Do nms
        # logger.debug(boxes)
        # logger.debug(boxes.shape) # torch.Size([82, 4])
        # logger.debug(scores)
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thresh).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        return result_boxes, result_scores, result_classid

    def run(self):
        model_name = self.get("model_name")
        server_url = self.get("server_url")
        verbose = self.get("verbose")
        model_w = self.get("model_w")
        model_h = self.get("model_h")
        model_c = self.get("model_c")
        outputdim = self.get("outputdim")
        verbose_output_filename = self.get("verbose_output_filename")
        outputtensorname = self.get("outputtensorname")
        inputtensorname = self.get("inputtensorname")

        # logger.debug(f"process {self.name} cam_id {cam_id} camera_address {encFile} gpu_id {gpuId} ")
        logger.debug(f"model_name {model_name} server_url {server_url} inputtensorname {inputtensorname}")
        logger.debug(f"outputtensorname {outputtensorname}")

        # init triton client
        triton_client = httpclient.InferenceServerClient(url=server_url, verbose=False)
        triton_client.unregister_system_shared_memory()
        triton_client.unregister_cuda_shared_memory()

        input_byte_size = 4 * model_c * model_h * model_w
        output_byte_size = outputdim * 4
        shm_ip0_handle = cudashm.create_shared_memory_region("input0_data", input_byte_size, 0)
        shm_op0_handle = cudashm.create_shared_memory_region("output0_data", output_byte_size, 0)
        triton_client.register_cuda_shared_memory("output0_data", cudashm.get_raw_handle(shm_op0_handle), 0,
                                                  output_byte_size)
        triton_client.register_cuda_shared_memory("input0_data", cudashm.get_raw_handle(shm_ip0_handle), 0,
                                                  input_byte_size)
        inputs, outputs = [], []
        inputs.append(httpclient.InferInput(inputtensorname, [1, model_c, model_h, model_w], "FP32"))
        inputs[-1].set_shared_memory("input0_data", input_byte_size)
        outputs.append(httpclient.InferRequestedOutput(outputtensorname, binary_data=True))
        outputs[-1].set_shared_memory("output0_data", output_byte_size)

        # init variables
        num_frame = 1
        if verbose:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(verbose_output_filename, fourcc, 25.0, (model_w, model_h), True)
        # begin read frames
        tt = Timer()
        while self.queue.empty():
            logger.debug(f"sleeping .......................")
            time.sleep(1)
        while True:
            try:
                frame_decoded: snp.ndarray = self.queue.get()
                if frame_decoded is None: continue
                assert frame_decoded.size == model_w * model_h * model_c * 2
                if verbose:
                    # frame = np.copy(frame_decoded)
                    frame = np.ascontiguousarray(frame_decoded[:model_w * model_h * model_c]).flatten()
                    frame = frame.reshape(model_h, model_w, model_c)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # frame = np.copy(frame_decoded)
                    # frame.resize(model_h, model_w,model_c)
                    # frame = frame[:, :, ::-1]

                input0_data = self.preprocess(frame_decoded[model_w * model_h * model_c:], model_c, model_h, model_w)

                frame_decoded.close()
                frame_decoded.unlink()

                cudashm.set_shared_memory_region(shm_ip0_handle, [input0_data])  # cong cpu -> gpu
                results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
                output0 = results.get_output(outputtensorname)

                if output0 is not None:
                    output0_data = cudashm.get_contents_as_numpy(shm_op0_handle,
                                                                 utils.triton_to_np_dtype(output0['datatype']),
                                                                 output0['shape'])
                else:
                    logger.critical("OUTPUT0 is missing in the response.")
                    sys.exit(1)

                if 0 == num_frame % 100:
                    num_frame = 1
                    logger.info(
                        f"the 100 times loop total time is : {tt.seconds()}, so fps is :{1 / tt.seconds() * 100} FPS")
                    logger.debug(
                        f"the output is: {output0},output type is:{type(output0)},output0_data type is :{type(output0_data)} output_data0 shape is :{output0_data.shape}")
                    tt.reset()
                else:
                    num_frame += 1

                self.det_q.put(output0_data[0])
                logger.debug(f"put self.det_q.put(output0_data[0]) {self.det_q.qsize()}")

                # # result_boxes, result_scores, result_classid = self.post_process(output0_data[0], model_h, model_w)
                # result_boxes, result_scores, result_classid = self.post_process(output0_data[0], input_h, input_w)
                # if verbose:
                #     cv2.rectangle(frame, (track[0], track[1]), (track[0] + track[2], track[1] + track[3]),
                #                   (255, 0, 0), 2)
                #     cv2.putText(frame, str(track[4]), (int(track[0]), int(track[1])), 0, 5e-3 * 100, (0, 255, 0), 2)
                #     out.write(frame)
                #     # out.write(cv2.flip(frame, 1))


            except:
                try:
                    frame_decoded.close()
                    frame_decoded.unlink()
                except:
                    pass
                traceback.print_exception(*sys.exc_info())
                # raise RuntimeError() from exec
