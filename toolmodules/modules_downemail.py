import poplib
import email
import time
import os
import datetime
import traceback

import numpy as np
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr


class AutoDownEmail():
    def __init__(self,
                 curdate,
                 host='pop.centuryfrontier.com',
                 username='ritchguo@centuryfrontier.com',
                 password='GR19981201sjqy',
                 priority='new',
                 enddate=None,
                 enddate_break=False,
                 default_save_path=None):
        self.curdate = curdate
        self.host = host
        self.username = username
        self.password = password

        if enddate is None:
            self.enddate = curdate
        else:
            self.enddate = enddate
        self.enddate_break = enddate_break

        self.server = poplib.POP3(self.host, timeout=180)
        self.server.user(self.username)
        self.server.pass_(self.password)
        self.priority = priority

        self.default_save_path = default_save_path
        if self.default_save_path is not None:
            if not os.path.exists(self.default_save_path): os.makedirs(self.default_save_path)

    def decode_str(self, string):
        value, charset = decode_header(string)[0]
        if charset:
            value = value.decode(charset)
        return value

    def get_email_headers(self, content):
        headers = {}  # 邮件的From, To, Subject存在于根对象上:
        for header in ['From', 'To', 'Subject', 'Date']:
            value = content.get(header, '')
            if value:
                if header == 'Date':
                    headers['Date'] = value
                if header == 'Subject':
                    subject = self.decode_str(value)   # 需要解码Subject字符串:
                    headers['Subject'] = subject
                else:
                    hdr, addr = parseaddr(value)   # 需要解码Email地址:
                    name = self.decode_str(hdr)
                    value = u'%s <%s>' % (name, addr)
                    if header == 'From':
                        from_address = value
                        headers['From'] = from_address
                    else:
                        to_address = value
                        headers['To'] = to_address
        return headers

    def get_attachments(self, msg, attach_list=None, filepath=None):
        if not os.path.exists(filepath): os.makedirs(filepath)
        for part in msg.walk():
            file_name = part.get_filename()  # 获取附件名称类型
            if file_name:
                h = email.header.Header(file_name)
                dh = email.header.decode_header(h)  # 对附件名称进行解码

                filename = self.decode_str(str(dh[0][0], dh[0][1])) if dh[0][1] else dh[0][0]
                filename = filename.replace('\r', '').replace('\n', '')
                try:
                    data = part.get_payload(decode=True)  # 下载附件
                    att_file = open(filepath + filename, 'wb')  # 在指定目录下创建文件，注意二进制文件需要用wb模式打开
                    att_file.write(data)  # 保存附件
                    att_file.close()
                    if attach_list is not None: attach_list.append(filename)
                except not PermissionError:
                    if attach_list is not None: attach_list.append(f'****** Error download: {filename}')
                    print('****** Error download:', filename)
                    print(traceback.format_exc())
        return attach_list

    def down_email(self, multi_match_infor: list):
        """
        multi_match_infor=[
            [('matched_1', 'matched_2'), ('filtered_1', 'filtered_1'), 'save_path'],
            [(), (), '']
        ]

        print('Messages: %s. Size: %s' % server.stat()) #stat()返回邮件数量和占用空间:
        可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
        """
        resp, mails, octets = self.server.list()
        index, attach_list = len(mails), []

        for i in range(index, 0, -1):        # 倒序遍历邮件
            resp, lines, octets = self.server.retr(i)       # lines存储了邮件的原始文本的每一行
            content_origin = b'\r\n'.join(lines).decode('utf-8', "ignore")        # 邮件的原始文本
            content = Parser().parsestr(content_origin)        # 解析邮件

            headers = self.get_email_headers(content)
            if headers.get('Subject') is None: continue
            if headers.get('From') is None: continue
            if self.username in headers.get('From'): continue

            subject = headers['Subject']
            date_email = time.strftime("%Y%m%d", time.strptime(headers["Date"][0:24], '%a, %d %b %Y %H:%M:%S'))
            try:
                print(f'[{date_email}]: {i}: {subject}  // content_type: {content.get_content_type()}')
            except:
                pass

            if date_email < self.enddate:
                break

            multi_match_infor_left = []
            for match_infor in multi_match_infor:
                if len(match_infor) == 3:
                    keyword_tuple, filtered_typle, save_path = match_infor
                    priority_email = self.priority
                else:
                    keyword_tuple, filtered_typle, save_path, priority_email = match_infor

                is_matched = bool(int(np.mean(
                    [kt in subject for kt in keyword_tuple] + [ft not in subject for ft in filtered_typle])))
                if is_matched:
                    attach_list = self.get_attachments(content, attach_list, save_path)
                    if priority_email != 'new':
                        multi_match_infor_left.append(match_infor)
                else:
                    multi_match_infor_left.append(match_infor)

            multi_match_infor = multi_match_infor_left

            if not self.enddate_break:
                if not multi_match_infor:
                    break

        self.server.quit()
        return attach_list

    def get_attachments_and_rename(self, msg, attach_list, match_attch_infor):
        for part in msg.walk():
            file_name_type = part.get_filename()  # 获取附件名称类型
            if file_name_type:
                h = email.header.Header(file_name_type)
                dh = email.header.decode_header(h)  # 对附件名称进行解码

                filename = self.decode_str(str(dh[0][0], dh[0][1])) if dh[0][1] else dh[0][0]
                filename = filename.replace('\r', '').replace('\n', '')
                try:
                    filename_target = ''
                    for keyword_tuple, filtered_typle, save_filename in match_attch_infor:
                        if bool(int(np.mean([kt in filename for kt in keyword_tuple] +
                                            [ft not in filename for ft in filtered_typle]))):
                            filename_target = save_filename
                            break
                    if len(filename_target) == 0:
                        continue
                    filepath = '/'.join(filename_target.split('/')[:-1])
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    data = part.get_payload(decode=True)  # 下载附件
                    att_file = open(filename_target, 'wb')  # 在指定目录下创建文件，注意二进制文件需要用wb模式打开
                    att_file.write(data)  # 保存附件
                    att_file.close()
                    attach_list.append(filename)
                except not PermissionError:
                    attach_list.append(f'****** Error download: {filename}')
                    print('****** Error download:', filename)
                    print(traceback.format_exc())
        return attach_list

    def down_email_rename_attch(self, multi_match_infor: list):
        """
        match_infor=[
            [('matched_1', 'matched_2'), ('filtered_1', 'filtered_1'), 'save_path', 'match_attch_infor'],
            [(), (), '', 'match_attch_infor']
        ]

        match_attch_infor=[
            [('matched_1', 'matched_2'), ('filtered_1', 'filtered_1'), 'save_name'],
            [(), (), '']
        ]
        print('Messages: %s. Size: %s' % server.stat()) #stat()返回邮件数量和占用空间:
        可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
        """
        resp, mails, octets = self.server.list()
        index, attach_list = len(mails), []

        for i in range(index, 0, -1):        # 倒序遍历邮件
            try:
                resp, lines, octets = self.server.retr(i)       # lines存储了邮件的原始文本的每一行
            except poplib.error_proto as e:
                print(f"发生 POP3 协议错误: {e}")
                continue

            content_origin = b'\r\n'.join(lines).decode('utf-8', "ignore")        # 邮件的原始文本
            content = Parser().parsestr(content_origin)        # 解析邮件

            try:
                headers = self.get_email_headers(content)
            except:
                continue

            if headers.get('Subject') is None: continue
            if headers.get('From') is None: continue
            if self.username in headers.get('From'): continue

            subject = headers['Subject']
            date_email = time.strftime("%Y%m%d", time.strptime(headers["Date"][0:24], '%a, %d %b %Y %H:%M:%S'))
            try: print(f'[{date_email}]: {i}: {subject}  // content_type: {content.get_content_type()}')
            except: pass

            if date_email < self.enddate: break

            multi_match_infor_left = []
            global_matched = False
            for match_infor in multi_match_infor:
                if len(match_infor) == 4:
                    keyword_tuple, filtered_typle, save_path, match_attch_infor = match_infor
                    priority_email = self.priority
                else:
                    keyword_tuple, filtered_typle, save_path, match_attch_infor, priority_email = match_infor

                is_matched = bool(int(np.mean([kt in subject for kt in keyword_tuple] + [ft not in subject for ft in filtered_typle])))
                if is_matched:
                    if match_attch_infor: attach_list = self.get_attachments_and_rename(content, attach_list, match_attch_infor)
                    else: attach_list = self.get_attachments(content, attach_list, save_path)

                    if priority_email != 'new': multi_match_infor_left.append(match_infor)
                    global_matched = True
                else:
                    multi_match_infor_left.append(match_infor)
            if (not global_matched) and (self.default_save_path is not None) and ('centuryfrontier.com' not in headers.get('From')):
                self.get_attachments(content, None, self.default_save_path)

            multi_match_infor = multi_match_infor_left

            if not multi_match_infor: break
        self.server.quit()

        return attach_list

    def down_email_rename_attch_temp(self, save_path, start_i=None):
        """
        match_infor=[
            [('matched_1', 'matched_2'), ('filtered_1', 'filtered_1'), 'save_path', 'match_attch_infor'],
            [(), (), '', 'match_attch_infor']
        ]

        match_attch_infor=[
            [('matched_1', 'matched_2'), ('filtered_1', 'filtered_1'), 'save_name'],
            [(), (), '']
        ]
        print('Messages: %s. Size: %s' % server.stat()) #stat()返回邮件数量和占用空间:
        可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
        """
        resp, mails, octets = self.server.list()
        index, attach_list = len(mails), []
        if start_i is None: start_i = index
        for i in range(start_i, 0, -1):        # 倒序遍历邮件
            try:
                resp, lines, octets = self.server.retr(i)       # lines存储了邮件的原始文本的每一行
            except:
                return i

            content_origin = b'\r\n'.join(lines).decode('utf-8', "ignore")        # 邮件的原始文本
            content = Parser().parsestr(content_origin)        # 解析邮件

            try:
                headers = self.get_email_headers(content)
            except:
                continue

            if headers.get('Subject') is None: continue
            if headers.get('From') is None: continue

            subject = headers['Subject']
            print(subject)
            if ('EQPB-世纪前沿鲸涛1号-DMA' not in subject) and ('Recall' not in subject): continue

            date_email = time.strftime("%Y%m%d", time.strptime(headers["Date"][0:24], '%a, %d %b %Y %H:%M:%S'))
            try: print(f'[{date_email}]: {i}: {subject}  // content_type: {content.get_content_type()}')
            except: pass

            self.get_attachments(content, None, save_path)

        self.server.quit()

        return attach_list


def down_email_all_temp(curdate, end_date='20240501', save_path=None):
    ade = AutoDownEmail(
        curdate,
        host='pop.centuryfrontier.com',
        username='ritchguo@centuryfrontier.com',
        password='GR19981201sjqy',
        priority='new',
        enddate=end_date,
        default_save_path=None
    )
    index = ade.down_email_rename_attch_temp(save_path=save_path)
    while True:
        ade = AutoDownEmail(
            curdate,
            host='pop.centuryfrontier.com',
            username='ritchguo@centuryfrontier.com',
            password='GR19981201sjqy',
            priority='new',
            enddate=end_date,
            default_save_path=None
        )
        index = ade.down_email_rename_attch_temp(save_path=save_path, start_i=index)
        if isinstance(index, int):
            continue
        break


if __name__ == '__main__':
    curdate = datetime.datetime.now().strftime('%Y%m%d')

    ade = AutoDownEmail(curdate)
    ade.down_email(
        multi_match_infor=[
            [
                ('matched1', ), ('filtered1', ), 'save_path'
            ]
        ]
    )
    ade.down_email_rename_attch(
        multi_match_infor=[
            [
                ('matched1',), ('filtered1',), 'save_path',
                [
                    [(), (), '']
                ]
            ],
            [
                ('matched1',), ('filtered1',), 'save_path',
                [
                    [(), (), '']
                ]
            ]
        ]
    )