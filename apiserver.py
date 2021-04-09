from flask import Flask, request, Response
from daemonize import Daemon
import fire
import traceback, json
import os


class daemonapiserver(Daemon):
    def __init__(self, pidfile):
        super(daemonapiserver, self).__init__(pidfile=pidfile)
        self.app = Flask(__name__)

    def run(self):
        @self.app.route('/aiconfig', methods=['POST'])
        def getAnalysisType():
            try:

                requestdata = request.json  # dict
                # print(requestdata)
                # with open('/dev/shm/nvxx/receivedjson','a') as fid:
                with open(os.path.join(os.getcwd(), 'receivedjson.json'), 'a') as fid:
                    fid.writelines(json.dumps(requestdata))
                    fid.write('\n')

            except Exception as exc:
                return Response(json.dumps(
                    {
                        "status_code": 1,  # 状态码，成功=200，其它值错误
                        "status_message": str(traceback.format_exc()),  # 错误信息
                    }
                    , ensure_ascii=False), mimetype='application/json')
            return Response(json.dumps(
                {
                    "status_code": 200,  # 状态码，成功=200，其它值错误
                    "status_message": "ok",  # 错误信息
                }
                , ensure_ascii=False), mimetype='application/json')

        self.app.run(debug=True, port=5123, host='0.0.0.0')


def dothework():
    p = daemonapiserver(os.path.join(os.getcwd(), 'apiserver.pid'))
    # p.daemonize()
    p.run()


if __name__ == "__main__":
    fire.Fire()
