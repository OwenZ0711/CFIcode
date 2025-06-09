# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:20:24 2021

@author: LuoMiao
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders

from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from pretty_html_table import build_table


class AutoSendEmail():
    def __init__(
            self,
            curdate, content_dict, subject, receivers, receivers_cc=None,
            imagedir=None, attchmentdir=None, sender=None, sender_pwd=None):
        if sender is None:
            self.sender = 'ritchguo@centuryfrontier.com'
        else:
            self.sender = sender
        if sender_pwd is None:
            self.sender_pwd = 'GR19981201sjqy'
        else:
            self.sender_pwd = sender_pwd

        self.subject = subject
        self.curdate = curdate
        self.receivers_all = ['luom@centuryfrontier.com',
                              '1026688756@qq.com',
                              'zhzhou@centuryfrontier.com',
                              'charleschen@centuryfrontier.com',
                              'hyanak@centuryfrontier.com',
                              'yang@centuryfrontier.com']
        self.receivers = receivers
        self.receivers_cc = receivers_cc
        if imagedir is not None:
            self.imagedir = [id for id in imagedir if os.path.exists(id)]
        else:
            self.imagedir = []

        if attchmentdir is not None:
            self.attchmentdir = [ad for ad in attchmentdir if os.path.exists(ad)]
        else:
            self.attchmentdir = []

        self.content_dict = content_dict

    def get_content(self, content='', blank='<br><br>'):
        for key in self.content_dict.keys():
            html_table = build_table(self.content_dict[key], 'blue_light')
            content += key + ':' + blank + html_table + blank
        return content

    def send_email(self):
        # msg = MIMEMultipart('related')
        if (not self.content_dict) and (not self.attchmentdir):
            return None

        msg = MIMEMultipart('mixed')

        msg['Subject'] = self.subject
        msg['From'] = self.sender
        msg['To'] = ','.join(self.receivers)
        if self.receivers_cc is not None:
            msg['Cc'] = ','.join(self.receivers_cc)

        msg.attach(MIMEText(self.get_content(), "html", "utf-8"))

        if self.imagedir:
            for img_i, img_path in enumerate(self.imagedir):
                msg_txt = MIMEText(
                    f"""
                    <hr><br>
                    <h1><font size="100">{img_path.split('/')[-1]}</font></h1><br>
                    <table>
                        <tr>
                            <td><img src="cid:{img_path.split('/')[-1]}"></td>
                        </tr>
                    </table>
                    <br><br>
                    """,
                    "html", "utf-8")
                msg.attach(msg_txt)
                with open(img_path, "rb") as image:
                    m_img = MIMEImage(image.read())
                    m_img.add_header('Content-ID', img_path.split('/')[-1])
                    m_img.add_header('Content-Disposition', 'attachment', filename=img_path.split('/')[-1])
                    msg.attach(m_img)

        if len(self.attchmentdir) > 0:
            for i in range(len(self.attchmentdir)):
                with open(self.attchmentdir[i], 'rb') as f:
                    att = MIMEApplication(f.read(), 'utf-8')  # 读入需要发送的excel文件
                    att.add_header('Content-Disposition', 'attachment', filename=self.attchmentdir[i].split('/')[-1])
                    # 添加附加 并且命名为.xlsx后缀文件  名字可以自取  但是后缀必须和源文件后缀相同
                    msg.attach(att)

        server = smtplib.SMTP_SSL('smtp.exmail.qq.com', 465, timeout=3000)
        server.login(self.sender, self.sender_pwd)  # log in to the server
        server.send_message(msg)
        server.quit()


class AutoSendEmailTxt():
    def __init__(self, curdate, content_txt, subject, receivers,
                 receivers_cc=None, imagedir=None, attchmentdir=None, sender=None, sender_pwd=None):
        if sender is None:
            self.sender = 'ritchguo@centuryfrontier.com'
        else:
            self.sender = sender
        if sender_pwd is None:
            self.sender_pwd = 'GR19981201sjqy'
        else:
            self.sender_pwd = sender_pwd

        self.subject = subject
        self.curdate = curdate
        self.receivers_all = ['luom@centuryfrontier.com',
                              '1026688756@qq.com',
                              'zhzhou@centuryfrontier.com',
                              'charleschen@centuryfrontier.com',
                              'hyanak@centuryfrontier.com',
                              'yang@centuryfrontier.com']
        self.receivers = receivers
        self.receivers_cc = receivers_cc
        if imagedir is not None:
            self.imagedir = [id for id in imagedir if os.path.exists(id)]
        else:
            self.imagedir = []

        if attchmentdir is not None:
            self.attchmentdir = [ad for ad in attchmentdir if os.path.exists(ad)]
        else:
            self.attchmentdir = []
        self.content_txt = content_txt

    def send_email(self):
        msg = MIMEMultipart('related')
        # msg = MIMEMultipart('mixed')

        msg['Subject'] = self.subject
        msg['From'] = self.sender
        msg['To'] = ','.join(self.receivers)
        if self.receivers_cc is not None:
            msg['Cc'] = ','.join(self.receivers_cc)

        msg.attach(MIMEText(self.content_txt))

        if len(self.imagedir) > 0:
            for i in range(len(self.imagedir)):
                with open(self.imagedir[i], "rb") as image:
                    m_img = MIMEImage(image.read())
                    m_img.add_header('Content-Disposition', 'attachment', filename=self.imagedir[i].split('/')[-1])
                    m_img.add_header('Content-ID', f'image<{i + 1}>')
                    encoders.encode_base64(m_img)
                    msg.attach(m_img)

        if len(self.attchmentdir) > 0:
            for i in range(len(self.attchmentdir)):
                att = MIMEApplication(open(self.attchmentdir[i], 'rb').read(), 'utf-8')  # 读入需要发送的excel文件
                att.add_header('Content-Disposition', 'attachment', filename=self.attchmentdir[i].split('/')[-1])
                # 添加附加 并且命名为.xlsx后缀文件  名字可以自取  但是后缀必须和源文件后缀相同
                msg.attach(att)

        server = smtplib.SMTP_SSL('smtp.exmail.qq.com', 465)
        server.login(self.sender, self.sender_pwd)  # log in to the server
        server.send_message(msg)
        server.quit()


class AutoSendEmailSimple():
    def __init__(self, curdate, content_dict, subject, receivers, imagedir=None, attchmentdir=None):
        self.sender, self.sender_pwd = 'ritchguo@centuryfrontier.com', 'GR19981201sjqy'

        self.subject = subject
        self.curdate = curdate
        self.receivers_all = ['luom@centuryfrontier.com',
                              '1026688756@qq.com',
                              'zhzhou@centuryfrontier.com',
                              'charleschen@centuryfrontier.com',
                              'hyanak@centuryfrontier.com',
                              'yang@centuryfrontier.com']
        self.receivers = receivers
        if imagedir is not None:
            self.imagedir = [id for id in imagedir if os.path.exists(id)]
        else:
            self.imagedir = []

        if attchmentdir is not None:
            self.attchmentdir = [ad for ad in attchmentdir if os.path.exists(ad)]
        else:
            self.attchmentdir = []
        self.content_dict = content_dict

    def get_content(self, content='', blank='<br><br>'):
        for key in self.content_dict.keys():
            html_table = build_table(self.content_dict[key], 'blue_light')
            content += key + ':' + blank + html_table + blank
        return content

    def send_email(self):
        # msg = MIMEMultipart('related')
        msg = MIMEMultipart('mixed')

        msg['Subject'] = self.subject
        msg['From'] = self.sender
        msg['To'] = ','.join(self.receivers)

        msg.attach(MIMEText(self.get_content(), "html", "utf-8"))

        # if len(self.imagedir) > 0:
        #     for i in range(len(self.imagedir)):
        #         msg_txt = MIMEText(
        #             f"""
        #             <hr><br>
        #             <h1><font size="100">{self.imagedir[i][1]}</font></h1><br>
        #             <table>
        #                 <tr>
        #                     <td><img src="cid:{self.imagedir[i][0].split('/')[-1]}"></td>
        #                 </tr>
        #             </table>
        #             <br><br>
        #             """,
        #             "html", "utf-8")
        #         msg.attach(msg_txt)
        #         with open(self.imagedir[i][0], "rb") as image:
        #             m_img = MIMEImage(image.read())
        #             m_img.add_header('Content-ID', self.imagedir[i][0].split('/')[-1])
        #             m_img.add_header('Content-Disposition', 'attachment', filename=self.imagedir[i][0].split('/')[-1])
        #             msg.attach(m_img)
        #     for i in range(len(self.imagedir)):
        #         with open(self.imagedir[i], "rb") as image:
        #             m_img = MIMEImage(image.read())
        #             m_img.add_header('Content-Disposition', 'attachment', filename=self.imagedir[i].split('/')[-1])
        #             m_img.add_header('Content-ID', self.imagedir[i].split('/')[-1])

        if len(self.imagedir) > 0:
            for i in range(len(self.imagedir)):
                with open(self.imagedir[i][0], "rb") as f:
                    m_img = MIMEBase('image', 'jpg')
                    m_img.add_header('Content-Disposition', 'attachment',
                                     filename=self.imagedir[i][0].split('/')[-1])
                    m_img.add_header('Content-ID', f'<{self.imagedir[i][1]}>')  # 设置图片id为0
                    m_img.set_payload(f.read())
                    encoders.encode_base64(m_img)
                    msg.attach(m_img)

        if len(self.attchmentdir) > 0:
            for i in range(len(self.attchmentdir)):
                att = MIMEApplication(open(self.attchmentdir[i], 'rb').read(), 'utf-8')  # 读入需要发送的excel文件
                att.add_header('Content-Disposition', 'attachment', filename=self.attchmentdir[i].split('/')[-1])
                # 添加附加 并且命名为.xlsx后缀文件  名字可以自取  但是后缀必须和源文件后缀相同
                msg.attach(att)

        server = smtplib.SMTP_SSL('smtp.exmail.qq.com', 465)
        server.login(self.sender, self.sender_pwd)  # log in to the server
        server.send_message(msg)
        server.quit()


class AutoSendEmailTxtSimple():
    def __init__(self, curdate, content_txt, subject, receivers, imagedir=None, attchmentdir=None):
        self.sender, self.sender_pwd = 'ritchguo@centuryfrontier.com', 'GR19981201sjqy'

        self.subject = subject
        self.curdate = curdate
        self.receivers_all = ['luom@centuryfrontier.com',
                              '1026688756@qq.com',
                              'zhzhou@centuryfrontier.com',
                              'charleschen@centuryfrontier.com',
                              'hyanak@centuryfrontier.com',
                              'yang@centuryfrontier.com']
        self.receivers = receivers
        if imagedir is not None:
            self.imagedir = [id for id in imagedir if os.path.exists(id)]
        else:
            self.imagedir = []

        if attchmentdir is not None:
            self.attchmentdir = [ad for ad in attchmentdir if os.path.exists(ad)]
        else:
            self.attchmentdir = []
        self.content_txt = content_txt

    def send_email(self):
        msg = MIMEMultipart('related')
        # msg = MIMEMultipart('mixed')

        msg['Subject'] = self.subject
        msg['From'] = self.sender
        msg['To'] = ','.join(self.receivers)

        msg.attach(MIMEText(self.content_txt))

        if len(self.imagedir) > 0:
            for i in range(len(self.imagedir)):
                with open(self.imagedir[i], "rb") as image:
                    m_img = MIMEImage(image.read())
                    m_img.add_header('Content-Disposition', 'attachment', filename=self.imagedir[i].split('/')[-1])
                    m_img.add_header('Content-ID', f'image<{i + 1}>')
                    encoders.encode_base64(m_img)
                    msg.attach(m_img)

        if len(self.attchmentdir) > 0:
            for i in range(len(self.attchmentdir)):
                att = MIMEApplication(open(self.attchmentdir[i], 'rb').read(), 'utf-8')  # 读入需要发送的excel文件
                att.add_header('Content-Disposition', 'attachment', filename=self.attchmentdir[i].split('/')[-1])
                # 添加附加 并且命名为.xlsx后缀文件  名字可以自取  但是后缀必须和源文件后缀相同
                msg.attach(att)

        server = smtplib.SMTP_SSL('smtp.exmail.qq.com', 465)
        server.login(self.sender, self.sender_pwd)  # log in to the server
        server.send_message(msg)
        server.quit()