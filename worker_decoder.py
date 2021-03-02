import multiprocessing
from typing import Any, Dict, List, Tuple, Union
import PyNvCodec as nvc
import shared_numpy as snp
import traceback
from configs import get_logger
import numpy as np
import sys
import shared_numpy as snp
import time

model_w = 640
model_h = 384
model_c = 3
input_w = 1280
input_h = 720

logger = get_logger()


class Worker_decoder(multiprocessing.Process):
    def __init__(self, queue: snp.Queue, idx, **kwargs):
        super(Worker_decoder, self).__init__()
        self.queue = queue
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

    def run(self):
        gpuId = self.get("gpu_id")
        encFile = self.get("rtsp")
        verbose = self.get("verbose")
        gpuId = 0 if gpuId < 0 else gpuId
        try:
            nvDec = nvc.PyNvDecoder(encFile, gpuId,
                                    {'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'})
            nvCvt = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvDec.Format(), nvc.PixelFormat.YUV420, gpuId)
            nvRes = nvc.PySurfaceResizer(model_w, model_h, nvCvt.Format(), gpuId)
            to_rgb = nvc.PySurfaceConverter(model_w, model_h, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, gpuId)
            to_planar = nvc.PySurfaceConverter(model_w, model_h, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuId)
            nvDwn = nvc.PySurfaceDownloader(model_w, model_h, to_rgb.Format(), 0)
            nvDwna = nvc.PySurfaceDownloader(model_w, model_h, to_planar.Format(), 0)
            while 1:
                try:
                    rawSurface = nvDec.DecodeSingleSurface()
                    if (rawSurface.Empty()):
                        nvDec = nvc.PyNvDecoder(encFile, gpuId,
                                                {'rtsp_transport': 'tcp', 'max_delay': '5000000', 'bufsize': '30000k'})
                        time.sleep(.5)
                        # raise RuntimeError("rawSurface empty") from exec
                except:
                    traceback.print_exception(*sys.exc_info())
                    continue

                yuvSurface = nvCvt.Execute(rawSurface)
                resSurface = nvRes.Execute(yuvSurface)
                rgb_byte = to_rgb.Execute(resSurface)
                rgb_planar = to_planar.Execute(rgb_byte)
                frameRGB = snp.ndarray((2 * model_w * model_h * model_c), dtype=np.uint8)
                # frameBGR = np.ndarray((3000, 3000), dtype=np.float32)
                # success = nvDwn.DownloadSingleSurface(rgb_planar, frameRGB)
                success = nvDwn.DownloadSingleSurface(rgb_byte, frameRGB[:model_w * model_h * model_c])
                if not (success):
                    raise RuntimeError("DecodeSingleFrame error ...") from exec
                success = nvDwna.DownloadSingleSurface(rgb_planar, frameRGB[model_w * model_h * model_c:])
                if not (success):
                    raise RuntimeError("DecodeSingleFrame error ...") from exec

                # if not self.queue.full():
                self.queue.put(frameRGB)
                # logger.debug(f"    self.queue put frame shape: {frameRGB.shape}")
                frameRGB.close()
                # print(f"put ...............")
                # time.sleep(.001)
            # self.queue.put(None)
        except:
            try:
                frameRGB.close()
                frameRGB.unlink()
            except:
                pass
            traceback.print_exception(*sys.exc_info())
            raise RuntimeError() from exec
