# """ 读取图片 """
# import json
#
# from AipImageClassify import client
#
#
# def get_file_content(filePath):
#     with open(filePath, 'rb') as fp:
#         return fp.read()
#
# image = get_file_content('C:\\Users\\Administrator\\Desktop\\1.PNG')
#
# """ 调用通用物体识别 """
# a=client.advancedGeneral(image);
# print(json.dumps(a, indent=1, ensure_ascii=False))
#
# """ 如果有可选参数 """
# options = {}
# options["baike_num"] = 5
#
# """ 带参数调用通用物体识别 """
# b=client.advancedGeneral(image, options)
# print(b)

""" 读取图片 """
from AipImageClassify import client


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('C:\\Users\\Administrator\\Desktop\\4.png')

""" 调用菜品识别 """
aa=client.dishDetect(image);
print(aa)
""" 如果有可选参数 """
options = {}
options["top_num"] = 3
options["filter_threshold"] = "0.7"
options["baike_num"] = 5

""" 带参数调用菜品识别 """
client.dishDetect(image, options)