import glob
import os
import cv2
import numpy as np
import random
from skimage.draw import polygon


def crop_image(image, x, y, width, height):
    """
    裁剪图片指定区域的图像
    起点坐标：x,y  宽高：width, height
    """
    cropped_image = image[y:y + height, x:x + width]
    return cropped_image


def convert_to_absolute(label, image_width, image_height):
    """
    将yolo格式的标注信息[class_id, x_center, y_center, relative_width, relative_height]
    转换为[class_id 左 上 右 下]的形式
    """
    class_id, relative_x_center, relative_y_center, relative_width, relative_height = label

    # 计算边界框的绝对坐标
    absolute_x_center = relative_x_center * image_width
    absolute_y_center = relative_y_center * image_height
    absolute_width = relative_width * image_width
    absolute_height = relative_height * image_height

    # 计算边界框的左上角和右下角坐标
    left = absolute_x_center - absolute_width / 2
    top = absolute_y_center - absolute_height / 2
    right = absolute_x_center + absolute_width / 2
    bottom = absolute_y_center + absolute_height / 2

    # 返回绝对坐标形式的边界框
    return [class_id, left, top, right, bottom]


def convert_to_yolo_format(class_id, left, top, right, bottom, image_width, image_height):
    """
    将[class_id 左 上 右 下]的标注信息
    转换为yolo格式[class_id, x_center, y_center, relative_width, relative_height]
    """
    # 计算目标框的中心点坐标和宽高
    x = (left + right) / 2
    y = (top + bottom) / 2
    width = right - left
    height = bottom - top

    # 将坐标和尺寸归一化到[0, 1]之间
    x /= image_width
    y /= image_height
    width /= image_width
    height /= image_height

    # 返回Yolo格式的标注
    return f"{class_id} {x} {y} {width} {height}"


def is_coincide(polygon_1, polygon_2):
    '''
    判断2个bondbox是否重合
    param polygon_1: [class_id， x1, y1, x2, y2]
    param polygon_2: [class_id， x3, y3, x4, y4]
    return:  True表示重合
    '''
    # 获取第一个矩形的左上角和右下角坐标
    x1 = min(polygon_1[1], polygon_1[3])
    y1 = max(polygon_1[2], polygon_1[4])

    # 获取第二个矩形的左上角和右下角坐标
    x2 = min(polygon_2[1], polygon_2[3])
    y2 = max(polygon_2[2], polygon_2[4])

    if (x1 > x2 and y1 < y2) or (y1 > y2 and x1 < x2):
        return True
    else:
        return False


def get_src_location_map(txtdir, imagewidth, imageheight):
    """
    输入yolo格式的标注txt文件，得到[class_id 左 上 右 下]的信息
    """
    with open(txtdir) as F:
        for linestr in F:
            info = linestr.strip().split(" ")
            Label = [int(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])]
            Class_id, Left, Top, Right, Bottom = convert_to_absolute(Label, imagewidth, imageheight)
            src_location_map.append([Class_id, Left, Top, Right, Bottom])
    return src_location_map


if __name__ == "__main__":
    '''
    指定输入地址：
    img_dir-----原始图片存放地址,程序指定图片格式为jpg，其他格式需要修改
    txt_dir-----原始图片yolo格式标注存放地址
    
    指定输出地址：
    cp_img_save-----copy-paste后图片保存地址
    cp_txt_save-----copy-paste后图片标注保存地址
    '''
    img_dir = r''  # 原始图片存放地址, img_dir形式应为"img save dir/*.jpg"
    txt_dir = r''   # 原始图片标注存放地址
    cp_img_save = r''  # copy-paste后图片保存地址
    cp_txt_save = r''  # copy-paste后图片标注保存地址
    os.makedirs(cp_img_save, exist_ok=True)
    os.makedirs(cp_txt_save, exist_ok=True)

    img_dir_list = glob.glob(img_dir)
    for img_path in img_dir_list:
        img_name = os.path.basename(img_path)
        img_txt = img_path.replace("img", "txt").replace(".jpg", ".txt")  # 找到对应的txt地址
        # print(img_name)
        # print(img_path)
        # print(img_txt)
        image_a = cv2.imread(img_path)
        image_height, image_width, _ = image_a.shape

        cp_img_dir = os.path.join(cp_img_save, img_name)
        cp_img_txt_dir = os.path.join(cp_txt_save, img_name.replace(".jpg", ".txt"))

        src_location_map = []
        cp_location_map = []
        src_location_map = get_src_location_map(img_txt, image_width, image_height)
        for row in src_location_map:
            class_id, left, top, right, bottom = row
            cp_location_map.append([class_id, left, top, right, bottom])

        image_b = cv2.imread(img_path)  # 复制原始图片
        res_list = []
        rescale_ratio = np.random.uniform(0.7, 1)   # 图像缩放比例
        # print(rescale_ratio)

        # 打开源文件和目标文件，并以追加模式打开
        with open(img_txt, 'r') as source, open(cp_img_txt_dir, 'a') as target:
            # 读取源文件的内容
            content = source.read()
            # 将内容写入目标文件
            target.write(content)

        for row in src_location_map:
            class_id, left, top, right, bottom = row
            if left or top or right or bottom:
                try:
                    # 目标可以出现在空白图片的任何位置,只要没有超过限制即可
                    x = int(left)  # 指定区域的起始横坐标
                    y = int(top)  # 指定区域的起始纵坐标
                    width = int(right - left)  # 指定区域的宽度
                    height = int(bottom - top)  # 指定区域的高度
                    cropped_image_a = crop_image(image_a, int(x), int(y), int(width), int(height))

                    # 计算新的宽度和高度
                    new_width = int(width * rescale_ratio)
                    new_height = int(height * rescale_ratio)
                    # print(new_width)
                    # print(new_height)
                    # 对裁剪下来的图像进行缩放
                    resized_cropped_image_a = cv2.resize(cropped_image_a, (new_width, new_height))

                    image_b_height, image_b_width, _ = image_b.shape

                    label = True
                    while label:
                        b_x = random.randint(0, int(image_b_width - width - 5))
                        b_y = random.randint(0, int(image_b_height - height - 5))
                        res_polygon = [class_id, b_x, b_y, b_x + new_width, b_y + new_height]

                        label = False
                        for or_polygon in src_location_map:
                            if is_coincide(res_polygon, or_polygon):
                                label = True
                                break

                    image_b[b_y:b_y + new_height, b_x:b_x + new_width] = resized_cropped_image_a
                    res = convert_to_yolo_format(class_id, b_x, b_y, b_x + new_width, b_y + new_height, image_b_width,
                                                 image_b_height)
                    cp_location_map.append([class_id, b_x, b_y, b_x + new_width, b_y + new_height])
                    with open(cp_img_txt_dir, "a") as f:
                        f.write(res + '\n')
                    cv2.imwrite(cp_img_dir, image_b)
                    # break
                except:
                    print("error")
                    break
