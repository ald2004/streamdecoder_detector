import os
import numpy as np
from PIL import Image
from tritonclient import utils

def preprocess(img, format, dtype, c, h, w, scaling, protocol):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    # np.set_printoptions(threshold='nan')
    if c == 1:
        sample_img = img.convert('L')
    else:
        sample_img = img.convert('RGB')
    resized_img = sample_img.resize((w, h), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    npdtype = utils.triton_to_np_dtype(dtype)
    typed = resized.astype(npdtype)
    if scaling == 'INCEPTION':
        scaled = (typed / 128) - 1
    elif scaling == 'VGG':
        if c == 1:
            scaled = typed - np.asarray((128,), dtype=npdtype)
        else:
            scaled = typed - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = typed
    # Swap to CHW if necessary
    if protocol == "grpc":
        if format == mc.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
    else:
        if format == "FORMAT_NCHW":
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled
    # Channels are in RGB order. Currently model configuration data
    # doesn't provide any information as to other channel orderings
    # (like BGR) so we just assume RGB.
    return ordered


def Postprocess(output0_data: np.ndarray, label_filename: str, topK: int):
    if os.path.exists(label_filename):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)  # only difference
        with open(label_filename) as fid:
            labels = list(map(str.strip, fid.readlines()))
            assert output0_data.size == len(labels)
            labelsarray = np.asarray(labels, dtype='<U20')
            inds = output0_data.argsort()[-topK:][::-1]
            return (labelsarray[inds].tolist(),softmax(output0_data[inds]).tolist())