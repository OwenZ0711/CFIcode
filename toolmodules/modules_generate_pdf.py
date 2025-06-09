from fpdf import FPDF
from PIL import Image
import os
import requests
import datetime


def wechat_bot_msg_check(msg, mention=None):
    if mention is None:
        mention = []
    url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=7a3ca5b1-c492-4aa7-993c-232ebe238f87'
    # 企业微信建群后在群里创建机器人，将页面的hook地址复制过来
    data = {
        "msgtype": "text",
        "text": {
            "content": msg,
            "mentioned_list": mention,
        }
    }
    a = requests.post(url, json=data)
    print(a)


class GeneratePDF():
    def __init__(self, curdate, pdf_path, max_hw=0.618):
        self.curdate = curdate
        self.pdf_path = pdf_path
        self.max_hw = max_hw
        if not os.path.exists(self.pdf_path):
            os.makedirs(self.pdf_path)

    def generate_pdf(self, pdf_filename, imgs_list):
        imgs_list_exist = []
        imgs_list_exist_not = []
        for il in imgs_list:
            if os.path.exists(il):
                imgs_list_exist.append(il)
            else:
                try:
                    imgs_list_exist_not.append(il.split('/')[-1])
                except:
                    pass
        try:
            msg = f'{pdf_filename}-[{len(imgs_list_exist)}]/[{len(imgs_list)}], ' \
                  f'不存在如下({datetime.datetime.now().strftime("%H%M%S")})：\n\t' + '\n\t'.join(imgs_list_exist_not)

            wechat_bot_msg_check(msg)
        except:
            pass

        width, height, hw_ratio, img_size_dict = 0, 0, 0, {}
        for img in imgs_list_exist:
            with Image.open(img) as img_file:
                w, h = img_file.size
            width, height = max(w, width), max(h, height)
            hw_ratio = min(max(h / w, hw_ratio), self.max_hw)
            img_size_dict[img] = [w, h, h / w]

        if height / width < hw_ratio:
            width, height = width, int(width * hw_ratio)
        else:
            width, height = int(height / hw_ratio), height

        hw_ratio_pdf = height / width
        width += 10
        height += 10

        pdf = FPDF(unit="pt", format=[width, height])
        for page in imgs_list_exist:
            pdf.add_page()

            w, h, hw_ratio = img_size_dict[page]
            if hw_ratio <= hw_ratio_pdf:
                w, h = width - 10, int(h * (width - 10) / w)
            else:
                w, h = int(w * (height - 10) / h), height - 10

            x = int((width - w) / 2)
            y = int((height - h) / 2)
            pdf.image(page, x, y, w, h)

        pdf.output(self.pdf_path + pdf_filename, "F")


