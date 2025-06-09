import time

import paramiko
import os
import warnings
import traceback
import time
import datetime
warnings.filterwarnings("ignore")


class TransferServerLocal():
    def __init__(self, ip='139.196.57.140', port=22, username='jumper', password='jump2cfi888'):
        self.transport = paramiko.Transport(ip, port)
        self.transport.banner_timeout = 30

        while True:
            try:
                self.transport.connect(username=username, password=password)
                break
            except:
                time.sleep(1)
                print(f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}({username}@{ip}:{port}) 连接出错！！！")
                print(traceback.format_exc())

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client._transport = self.transport

        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def exec_command(self, cmd):
        if isinstance(cmd, str):
            stdin, stdout, stderr = self.client.exec_command(cmd)
            print(str(stdout.read(), encoding='utf-8'))
        elif isinstance(cmd, list):
            for c in cmd:
                stdin, stout, stderr = self.client.exec_command(c)
        else:
            assert False, "'str' and 'list' are expected!!"

    def upload_file(self, local_path, server_path):
        # server_file_path = '/'.join(server_path.split('/')[:-1])
        # if not os.path.exists(f'{server_file_path}/'):
        #     self.exec_command(f'mkdir {server_file_path}')
        self.sftp.put(local_path, server_path)

    def upload_directory(self, local_path, server_path):
        if not os.path.exists(f'{server_path}'):
            self.exec_command(f'mkdir {server_path}')

        for filename in os.listdir(local_path):
            if os.path.isdir(local_path + filename):
                if not os.path.exists(f'{server_path}{filename}/'):
                    self.exec_command(f'mkdir {server_path}{filename}/')
                self.upload_directory(f'{local_path}{filename}/', f'{server_path}{filename}/')
            else:
                self.sftp.put(local_path + filename, server_path + filename)

    def download_file(self, server_path, local_path):
        self.sftp.get(server_path, local_path)

    def download_directory(self, server_path, local_path):
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        for filename in self.sftp.listdir_attr(server_path):
            if filename.asbytes().decode()[0] == 'd':
                if not os.path.exists(f'{local_path}{filename.filename}/'):
                    os.makedirs(f'{local_path}{filename.filename}/')
                self.download_directory(f'{server_path}{filename.filename}/', f'{local_path}{filename.filename}/')
            elif filename.asbytes().decode()[0] == '-':
                self.sftp.get(server_path + filename.filename, local_path + filename.filename)
            else:
                pass
