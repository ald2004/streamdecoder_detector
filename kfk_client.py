from nvjpeg import NvJpeg
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
from configs import get_cfg, get_logger
import numpy as np
import os, sys
import uuid
from datetime import date
import random

logger = get_logger()


# init default config and merge from base.yaml
# default values configs/__init__.py

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            # because I wanted to initialise my class
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class ProducerWarp(metaclass=Singleton):
    __connection = None

    def __init__(self, server, username, password):
        self.__connection = KafkaProducer(bootstrap_servers=server, security_protocol="SASL_PLAINTEXT",
                                          sasl_mechanism='PLAIN',
                                          sasl_plain_username=username, sasl_plain_password=password)

    # 成功回调
    @staticmethod
    def on_send_success(record_metadata):
        # logger.debug(
        #     f"succeed send to topic :{record_metadata.topic}, partation: {record_metadata.partition}, and offset: {record_metadata.offset}")
        pass

    # 错误回调
    @staticmethod
    def on_send_error(excp):
        logger.debug(f"faild send : {excp}")
        # logger.debug('I am an errback', exc_info=excp)

    def produce_business(self, data, topic):
        self.__connection.send(topic, value=data).add_callback(ProducerWarp.on_send_success).add_errback(ProducerWarp.on_send_error)

    def close(self):
        self.__connection.close(timeout=1)


def persis_image_sendkafka(img: np.ndarray, args: dict, cfg=None):
    # logger.debug("===============")
    # for k, v in cfg.items():
    #     logger.debug(f"k is:{k}, v is :{v}")
    # logger.debug("===============")
    # encode jpeg frame and write to disk ...
    if img is None: return
    assert img.ndim > 2

    if img.ndim == 3:
        imgs = np.expand_dims(img, axis=0)
    else:
        imgs = img
    if cfg is None: cfg = get_cfg("configs/DECODER.yaml")
    server = cfg.KFK_SERVER_LIST
    username = cfg.KFK_CONSUMER_USER
    password = cfg.KFK_CONSUMER_PWD
    topic = cfg.KFK_TOPIC_CUSTOMERFLOW
    moutpoint = cfg.PICTURE_MOUNT_POINT  # "/dev/shm/PICTURE_MOUNT_POINT"
    localhostip = cfg.LOCAL_HOST_IP  # 10.10.117.131
    today = date.today().strftime('%Y%m%d')  # "20210303"
    try:
        busitype = args['busitype']
    except:
        busitype = random.choice(
            ["dustbin", "car", "bus", "track", "bicycle", "motocycle", "tricycle", "person", "face"])
    nj = NvJpeg()
    # producer = KafkaProducer(bootstrap_servers=server, security_protocol="SASL_PLAINTEXT", sasl_mechanism='PLAIN',
    #                          sasl_plain_username=username, sasl_plain_password=password)
    producer = ProducerWarp(server, username, password)
    data = None
    with open('test.json', "rb") as f:
        data = f.read()
    djson = json.loads(data)

    # merge djson from args
    for k, v in args.items():
        try:
            djson[k] = v
        except:
            pass

    # merge
    os.makedirs(os.path.join(moutpoint, busitype, today), exist_ok=True)
    for frame in imgs:
        assert frame.ndim == 3
        uudi_tmp = uuid.uuid4().hex
        with open(os.path.join(moutpoint, busitype, today, f"{uudi_tmp}.jpg"), "wb") as fid:
            frame_jpg = nj.encode(frame)
            fid.write(frame_jpg)
        djson['image_url'] = os.path.join(localhostip, busitype, today, f"{uudi_tmp}.jpg")
        dd = json.dumps(djson).encode('utf-8')
        # producer.send(topic, value=dd).add_callback(on_send_success).add_errback(on_send_error)
        producer.produce_business(dd, topic)
        print('.', end='')
    sys.stdout.flush()
    # logger.debug(dd)
    # producer.close()


if __name__ == '__main__':
    cfg = get_cfg("configs/DECODER.yaml")
    logger.debug("===============")
    for k, v in cfg.items():
        logger.debug(f"k is:{k}, v is :{v}")
    logger.debug("===============")
    import cv2

    cap = cv2.VideoCapture("/dev/shm/output.avi")
    args = {}
    for i in range(10):
        f, img = cap.read()
        # img = cv2.imread('test.jpg')
        if not f:
            continue
        persis_image_sendkafka(img, args)

    # docker run -it --rm -d -p 80:80 --name web -v /dev/shm/PICTURE_MOUNT_POINT:/usr/share/nginx/html:Z nginx
