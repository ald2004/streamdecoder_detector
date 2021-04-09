import multiprocessing

from daemonize import Daemon
from cfgapiutils.config import get_cfg
from fvcore.common.timer import Timer
from cfgapiutils.logger import setup_logger
import json, os, time
import hashlib

from triton_test_kfk import temporal_restrict as temporal_restrict_main

# from triton_test_kfk_bk import xxx as temporal_restrict_main
logger = setup_logger(name="temporal_restrict", output="temporal_restrict_log")
receivedjson = "./receivedjson.json"
baseconfigyamlfile = "./configs/BASE.yaml"


class temporal_restrict(Daemon):
    def __init__(self, pidfile):
        super(temporal_restrict, self).__init__(pidfile=pidfile)
        self.tt = Timer()
        self.event_type = str(self.__class__.__name__)
        self.cfgs = getattr(get_cfg(), self.event_type)

    def gen_yaml(self, file: str, yamlfile: str):
        with open(file, 'r') as fid:
            for line in fid.readlines():
                if self.event_type in line:
                    try:
                        dcts = json.loads(line)
                        for dct in dcts:
                            for k, v in dct.items():
                                # logger.debug(f"{k} ---- {v}")
                                getattr(self.cfgs, str(k)).append(v)
                    except:
                        # pass
                        logger.debug(f"{k} ---- {v}")
                        raise

        try:
            os.makedirs(os.path.dirname(yamlfile), exist_ok=True)
            self.cfgs.CAMERA_LIST = [x for x in self.cfgs.flow_link]
            # logger.critical(self.cfgs.zone)
            # self.cfgs.FENCES = [float(x) for x in str(self.cfgs.zone[0]).split(',')]
            self.cfgs.FENCES = self.cfgs.zone
            # if len(self.cfgs.zone) > 1:
            #     self.cfgs.FENCES = [x for x in self.cfgs.zone]
            self.cfgs.flow_link = []
            self.cfgs.zone = []
            # logger.debug(self.cfgs.dump())
            with open(yamlfile, 'w') as fid:
                fid.write(self.cfgs.dump())
        except:
            logger.critical(self.cfgs.zone)
            raise

    def run(self):
        pid = str(os.getpid())
        open(self.pidfile, 'w+').write("%s\n" % pid)

        temporal_restrict_main()
        # while 1:
        #     print(f'1111111111111    {os.getpid()}')
        #     time.sleep(1)
        # configmodified = True
        # while 1:
        #     self.gen_yaml(receivedjson, baseconfigyamlfile)
        #     if configmodified:
        #         self.restart()
        #         # self.killallchild()
        #         # temporal_restrict_main()
        #         # configmodified = self.updateconfigmodified()
        #     time.sleep(10)


class keeper():
    def __init__(self, ):
        super(keeper, self).__init__()
        self.pre_config_md5 = hashlib.md5(''.join(open(receivedjson, 'r').readlines()).encode('utf-8')).hexdigest()
        self.cur_config_md5 = self.pre_config_md5

    def updateconfigmodified(self):
        # logger.debug('aaaaaaaaaaaaaaaa')
        # logger.debug(self.pre_config_md5)
        self.cur_config_md5 = hashlib.md5(''.join(open(receivedjson, 'r').readlines()).encode('utf-8')).hexdigest()
        xxxx = self.pre_config_md5
        self.pre_config_md5 = self.cur_config_md5
        # logger.debug('bbbbbbbbbbbb')
        # logger.debug(self.pre_config_md5)
        return self.cur_config_md5 != xxxx


def start_do_the_work():
    p = temporal_restrict("temporal_restrict.pid")
    p.gen_yaml(receivedjson, baseconfigyamlfile)
    p.run()


def stop_do_the_work():
    p = temporal_restrict("temporal_restrict.pid")
    p.gen_yaml(receivedjson, baseconfigyamlfile)
    p.stop()


if __name__ == '__main__':
    keeper = keeper()
    configmodified = False
    p = multiprocessing.Process(target=start_do_the_work)
    p.start()
    while 1:
        configmodified = keeper.updateconfigmodified()
        if configmodified:
            logger.debug(f"processing restarting ...")
            p = multiprocessing.Process(target=stop_do_the_work)
            p.start()
            p = multiprocessing.Process(target=start_do_the_work)
            p.start()
        time.sleep(2)
