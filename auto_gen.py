import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd

categories = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'person'},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'car'},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'airplane'},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'bus'},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'train'},
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'truck'},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'boat'},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'traffic light'},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'fire hydrant'},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 12, 'name': 'stop sign'},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 13, 'name': 'parking meter'},
    {'color': [250, 0, 30], 'isthing': 1, 'id': 14, 'name': 'bench'},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 15, 'name': 'bird'},
    {'color': [255, 77, 255], 'isthing': 1, 'id': 16, 'name': 'cat'},
    {'color': [0, 226, 252], 'isthing': 1, 'id': 17, 'name': 'dog'},
    {'color': [182, 182, 255], 'isthing': 1, 'id': 18, 'name': 'horse'},
    {'color': [0, 82, 0], 'isthing': 1, 'id': 19, 'name': 'sheep'},
    {'color': [120, 166, 157], 'isthing': 1, 'id': 20, 'name': 'cow'},
    {'color': [110, 76, 0], 'isthing': 1, 'id': 21, 'name': 'elephant'},
    {'color': [174, 57, 255], 'isthing': 1, 'id': 22, 'name': 'bear'},
    {'color': [199, 100, 0], 'isthing': 1, 'id': 23, 'name': 'zebra'},
    {'color': [72, 0, 118], 'isthing': 1, 'id': 24, 'name': 'giraffe'},
    {'color': [255, 179, 240], 'isthing': 1, 'id': 25, 'name': 'backpack'},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 26, 'name': 'umbrella'},
    {'color': [209, 0, 151], 'isthing': 1, 'id': 27, 'name': 'handbag'},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 28, 'name': 'tie'},
    {'color': [0, 220, 176], 'isthing': 1, 'id': 29, 'name': 'suitcase'},
    {'color': [255, 99, 164], 'isthing': 1, 'id': 30, 'name': 'frisbee'},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 31, 'name': 'skis'},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 32, 'name': 'snowboard'},
    {'color': [78, 180, 255], 'isthing': 1, 'id': 33, 'name': 'sports ball'},
    {'color': [0, 228, 0], 'isthing': 1, 'id': 34, 'name': 'kite'},
    {'color': [174, 255, 243], 'isthing': 1, 'id': 35, 'name': 'baseball bat'},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 36, 'name': 'baseball glove'},
    {'color': [134, 134, 103], 'isthing': 1, 'id': 37, 'name': 'skateboard'},
    {'color': [145, 148, 174], 'isthing': 1, 'id': 38, 'name': 'surfboard'},
    {'color': [255, 208, 186], 'isthing': 1, 'id': 39, 'name': 'tennis racket'},
    {'color': [197, 226, 255], 'isthing': 1, 'id': 40, 'name': 'bottle'},
    {'color': [171, 134, 1], 'isthing': 1, 'id': 41, 'name': 'wine glass'},
    {'color': [109, 63, 54], 'isthing': 1, 'id': 42, 'name': 'cup'},
    {'color': [207, 138, 255], 'isthing': 1, 'id': 43, 'name': 'fork'},
    {'color': [151, 0, 95], 'isthing': 1, 'id': 44, 'name': 'knife'},
    {'color': [9, 80, 61], 'isthing': 1, 'id': 45, 'name': 'spoon'},
    {'color': [84, 105, 51], 'isthing': 1, 'id': 46, 'name': 'bowl'},
    {'color': [74, 65, 105], 'isthing': 1, 'id': 47, 'name': 'banana'},
    {'color': [166, 196, 102], 'isthing': 1, 'id': 48, 'name': 'apple'},
    {'color': [208, 195, 210], 'isthing': 1, 'id': 49, 'name': 'sandwich'},
    {'color': [255, 109, 65], 'isthing': 1, 'id': 50, 'name': 'orange'},
    {'color': [0, 143, 149], 'isthing': 1, 'id': 51, 'name': 'broccoli'},
    {'color': [179, 0, 194], 'isthing': 1, 'id': 52, 'name': 'carrot'},
    {'color': [209, 99, 106], 'isthing': 1, 'id': 53, 'name': 'hot dog'},
    {'color': [5, 121, 0], 'isthing': 1, 'id': 54, 'name': 'pizza'},
    {'color': [227, 255, 205], 'isthing': 1, 'id': 55, 'name': 'donut'},
    {'color': [147, 186, 208], 'isthing': 1, 'id': 56, 'name': 'cake'},
    {'color': [153, 69, 1], 'isthing': 1, 'id': 57, 'name': 'chair'},
    {'color': [3, 95, 161], 'isthing': 1, 'id': 58, 'name': 'couch'},
    {'color': [163, 255, 0], 'isthing': 1, 'id': 59, 'name': 'potted plant'},
    {'color': [119, 0, 170], 'isthing': 1, 'id': 60, 'name': 'bed'},
    {'color': [0, 182, 199], 'isthing': 1, 'id': 61, 'name': 'dining table'},
    {'color': [0, 165, 120], 'isthing': 1, 'id': 62, 'name': 'toilet'},
    {'color': [183, 130, 88], 'isthing': 1, 'id': 63, 'name': 'tv'},
    {'color': [95, 32, 0], 'isthing': 1, 'id': 64, 'name': 'laptop'},
    {'color': [130, 114, 135], 'isthing': 1, 'id': 65, 'name': 'mouse'},
    {'color': [110, 129, 133], 'isthing': 1, 'id': 66, 'name': 'remote'},
    {'color': [166, 74, 118], 'isthing': 1, 'id': 67, 'name': 'keyboard'},
    {'color': [219, 142, 185], 'isthing': 1, 'id': 68, 'name': 'cell phone'},
    {'color': [79, 210, 114], 'isthing': 1, 'id': 69, 'name': 'microwave'},
    {'color': [178, 90, 62], 'isthing': 1, 'id': 70, 'name': 'oven'},
    {'color': [65, 70, 15], 'isthing': 1, 'id': 71, 'name': 'toaster'},
    {'color': [127, 167, 115], 'isthing': 1, 'id': 72, 'name': 'sink'},
    {'color': [59, 105, 106], 'isthing': 1, 'id': 73, 'name': 'refrigerator'},
    {'color': [142, 108, 45], 'isthing': 1, 'id': 74, 'name': 'book'},
    {'color': [196, 172, 0], 'isthing': 1, 'id': 75, 'name': 'clock'},
    {'color': [95, 54, 80], 'isthing': 1, 'id': 76, 'name': 'vase'},
    {'color': [128, 76, 255], 'isthing': 1, 'id': 77, 'name': 'scissors'},
    {'color': [201, 57, 1], 'isthing': 1, 'id': 78, 'name': 'teddy bear'},
    {'color': [246, 0, 122], 'isthing': 1, 'id': 79, 'name': 'hair drier'},
    {'color': [191, 162, 208], 'isthing': 1, 'id': 80, 'name': 'toothbrush'}
]
categories_index = {}
for x in categories:
    categories_index.update({x['name']: x['id']})


def show_anns(anns, scores):
    if len(anns) == 0:
        return
    m = anns[0]['segmentation']  # 【x,x】
    #     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img2 = np.zeros((m.shape[0], m.shape[1]))  # 【x,x】全白255
    for i, ann in enumerate(anns):
        m = ann['segmentation']  # 【x,x】
        color_mask = categories_index[scores[i]['class']]
        img2[m[:, :] == 1] = color_mask
    #         img2[m[:,:]==True] = color_mask
    return img2


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        # texts.append(template.format(cls_split))  # 用classname和template构成完全的所有的句子
                        texts = template.format(cls_split)  # 用classname和template构成完全的所有的句子，单个的
            else:
                # texts = [template.format(classname) for template in templates]  # format with class
                texts = templates.format(classname)

            zeroshot_weights.append(texts)
    return zeroshot_weights


# 求所有相邻的区域
def get_nerbor(masks2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    nerbor = {}
    for i, x in enumerate(masks2):
        m = x['segmentation']
        m = m.astype('uint8')
        m = cv2.dilate(m, kernel)
        tem = []
        for j, y in enumerate(masks2[i + 1:]):
            m2 = y['segmentation']

            re = m & m2  # 与运算，看膨胀后有没有交在一起的

            if (len(re[re[:, :] == 1]) != 0):
                tem.append(i + j + 1)

        if len(tem) != 0:
            nerbor.update({i: tem})
    return nerbor


# nerbor

import numpy as np
import os
import json


def gen_crop(masks):
    crop_list = []
    for i in range(len(masks)):
        mask = masks[i]['segmentation']
        mask = np.array(mask, dtype=np.int32) * 255
        #     print(tem*255)
        person = image.copy()
        person_ori = person.copy()
        mask_ = mask / 255.0
        person[:, :, 0] = person[:, :, 0] * mask_
        person[:, :, 1] = person[:, :, 1] * mask_
        person[:, :, 2] = person[:, :, 2] * mask_
        # 这里做个相加就可以实现合并
        # result = cv2.add(back,person)
        diffImg2 = cv2.subtract(person, person_ori)
        (cnts, _) = cv2.findContours(np.uint8(mask.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # draw a bounding box arounded the detected barcode and display the image
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs) if min(Xs) > 0 else 0
        x2 = max(Xs) if max(Xs) > 0 else 0
        y1 = min(Ys) if min(Ys) > 0 else 0
        y2 = max(Ys) if max(Ys) > 0 else 0
        hight = y2 - y1
        if hight <= 3:
            hight = 4
        width = x2 - x1
        if width <= 3:
            width = 4
        cropImg = person[y1:y1 + hight, x1:x1 + width]
        crop_list.append(cropImg)

    return crop_list


# 送入clip得分
# 加载clip
def actr(or_image, logits_per_image, texts, model, device, start_layer=-1):
    or_image = np.array(Image.fromarray(or_image).resize((224, 224)))
    mask = np.zeros((np.array(or_image).shape[0], np.array(or_image).shape[1]))
    mask[:, :] = or_image[:, :, 0]
    mask[mask[:, :] != 0] = 1
    batch_size = texts.shape[0]
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * logits_per_image)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    dim = int(image_relevance[0].numel() ** 0.5)
    image_relevance = image_relevance[0].reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')

    image_relevance = image_relevance.reshape(224, 224).cpu().data.cpu().numpy()

    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())

    image_relevance = np.array(image_relevance)

    com1 = image_relevance[mask[:, :] == 1]
    com2 = image_relevance[mask[:, :] == 0]
    if np.sum(com1[com1[:] > 0.3]) / float(len(com1[com1[:] > 0])) > np.sum(com2[com2[:] > 0.3]) / float(
            len(com2[com2[:] > 0])):
        return True, image_relevance
    else:
        return False, image_relevance


def get_clip_score(text_prompts, crop_list, model1, model2, processor1, processor2, masks, exc):
    images1 = []
    images2 = []
    for image in crop_list:
        images1.append(processor1(Image.fromarray(image).resize((224, 224))))
        images2.append(processor2(Image.fromarray(image).resize((224, 224))))
    image_input1 = torch.tensor(np.stack(images1)).to(device)
    image_input2 = torch.tensor(np.stack(images2)).to(device2)
    text_prompts1 = clip.tokenize(text_prompts).to(device)
    text_prompts2 = clip.tokenize(text_prompts).to(device2)
    logits_per_image1, logits_per_text1 = model1(image_input1, text_prompts1)
    logits_per_image2, logits_per_text2 = model2(image_input2, text_prompts2)
    probs1 = logits_per_image1.softmax(dim=1)
    probs2 = logits_per_image2.softmax(dim=1)
    scores1, index1 = probs1.topk(1, dim=1)
    scores2, index2 = probs2.topk(1, dim=1)
    c = []
    tem = []
    for i in range(len(crop_list)):
        if exc:
            img1 = images1[i].reshape((1, 3, 224, 224)).to(device)
            img2 = images2[i].reshape((1, 3, 224, 224)).to(device2)
            texts1 = text_prompts1[int(index1[i])].reshape((1, -1))
            texts2 = text_prompts2[int(index1[i])].reshape((1, -1))
            batch_size = texts1.shape[0]
            img1 = img1.repeat(batch_size, 1, 1, 1).to(device)
            logits_per_image3, _ = model1(img1, texts1)
            logits_per_image4, _ = model2(img2, texts2)


            sym1, image_relevance1 = actr(crop_list[i], logits_per_image3, texts1, model1, device)
            sym2, image_relevance2 = actr(crop_list[i], logits_per_image4, texts2, model2, device2)

            c.append(image_relevance1)

            if index1[i].cpu() == index2[i].cpu():

                if sym1 or sym2:
                    tem.append({'score': float(max(scores1[i].cpu(), scores2[i].cpu())),
                                "class": train_class_indexes[int(index1[i])]})

                else:

                    tem.append({'score': 0, "class": train_class_indexes[int(index1[i])]})
                #                 del img1
            else:
                if sym1 and not sym2:
                    tem.append({'score': float(scores1[i]), "class": train_class_indexes[int(index1[i])]})
                if not sym1 and sym2:
                    tem.append({'score': float(scores2[i]), "class": train_class_indexes[int(index2[i])]})
                if sym1 and sym2:
                    sc = scores1[i] if scores1[i].cpu() > scores2[i].cpu() else scores2[i]
                    class_name = train_class_indexes[int(index1[i])] if scores1[i].cpu() > scores2[i].cpu() else \
                    train_class_indexes[int(index2[i])]
                    tem.append({'score': float(sc), "class": class_name})
                if not sym1 and not sym2:
                    tem.append({'score': 0, "class": train_class_indexes[int(index1[i])]})
        else:
            sc = scores1[i] if scores1[i].cpu() > scores2[i].cpu() else scores2[i]
            class_name = train_class_indexes[int(index1[i])] if scores1[i].cpu() > scores2[i].cpu() else \
            train_class_indexes[int(index2[i])]
            tem.append({'score': float(sc), "class": class_name})
    del text_prompts, image_input1, image_input2
    return tem



# 计算所有crop之间的相似性
from scipy import spatial
class CosineSimilarityTest(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityTest, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)
        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
        final = x.mul(1 / x_frobenins)
        return final


def sim_crop(crop_list,cos_model,model):
    images=[]
    for image in crop_list:
        images.append(processor1(Image.fromarray(image)))
    similar=np.zeros((len(images),len(images)))
    image_input = torch.tensor(np.stack(images))
    image_input = image_input.to(device)
    image_features = model.encode_image(image_input).float()
    #     img_emb = model.get_image_features(**image_features)  #图像特征，大小一直(1,512)
    img_emb = image_features.detach()
    similar = cos_model(img_emb, img_emb)
    similar.fill_diagonal_(0)
    similar = similar.cpu().numpy()
    return similar


# 去重(预处理)
def del_overlap(masks):
    pre_del=[]
    for i,x in enumerate(masks[:len(masks)-1]):
        m=x['segmentation']
        m=m*1
        for j,y in enumerate(masks[i+1:]):
            m2 = y['segmentation']
            min=len(m[m[:,:]==1]) if len(m[m[:,:]==1])<len(m2[m2[:,:]==1]) else len(m2[m2[:,:]==1])
            if min==0:
                continue
            y['segmentation']=y['segmentation']*1
            re = m & m2
            if len(re[re[:,:]==1])/float(min)>=0.8:

                _,masks,m = com_mask(i,i+j+1,masks)
                m=m[0]['segmentation']
                pre_del.append(i+j+1)
    masks=[i for num,i in enumerate(masks) if num not in pre_del] #枚举法删除，del和remove容易出错
    pre_del=[]
    for i,m in enumerate(masks):
        if m['area']<300:
            pre_del.append(i)
    masks=[i for num,i in enumerate(masks) if num not in pre_del] #枚举法删除，del和remove容易出错
    return masks


# 合并相邻masks
def com_mask(index1,index2,masks):
    new_mask_list=[]
    index=[index1,index2]
    m = masks[index1]['segmentation']
    new_seg=masks[index1]['segmentation'] | masks[index2]['segmentation']
    area = len(new_seg[new_seg[:,:]==1])
    new_mask={'segmentation':new_seg,'area':area}
    zero_mask=copy.deepcopy(masks)
    zero_mask[index1]=new_mask
    zero_mask[index2]['segmentation']=np.zeros((image.shape[0],image.shape[1]),int)
    masks=[i for num,i in enumerate(masks) if num not in index] #枚举法删除，del和remove容易出错
    masks.append(new_mask)
    new_mask_list.append(new_mask)

    return masks,zero_mask,new_mask_list


from PIL import Image
from pylab import *


class LBP:
    # 将图像载入，并转化为灰度图，获取图像灰度图的像素信息
    def describe(self, image):
        image_array = np.array(Image.open(image).convert('L').resize((100, 100)))
        # image_array=cv2.imread(image)
        return image_array

    # 图像的LBP原始特征计算算法：将图像指定位置的像素与周围8个像素比较
    # 比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    def calute_basic_lbp(self, image_array, i, j):
        sum = []
        if image_array[i - 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i - 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j - 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        if image_array[i + 1, j + 1] > image_array[i, j]:
            sum.append(1)
        else:
            sum.append(0)
        return sum

    # 获取值r的二进制中1的位数
    def calc_sum(self, r):
        num = 0
        while (r):
            r &= (r - 1)
            num += 1
        return num

    # 获取图像的LBP原始模式特征
    def lbp_basic(self, image_array):
        basic_array = np.zeros(image_array.shape, np.uint8)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                sum = self.calute_basic_lbp(image_array, i, j)
                bit_num = 0
                result = 0
                for s in sum:
                    result += s << bit_num
                    bit_num += 1
                basic_array[i, j] = result
        return basic_array

    # 获取图像的LBP旋转不变等价模式特征
    def lbp_revolve_uniform(self, image_array):
        uniform_revolve_array = np.zeros(image_array.shape, np.uint8)
        basic_array = self.lbp_basic(image_array)
        width = image_array.shape[0]
        height = image_array.shape[1]
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                k = basic_array[i, j] << 1
                if k > 255:
                    k = k - 255
                xor = basic_array[i, j] ^ k
                num = self.calc_sum(xor)
                if num <= 2:
                    uniform_revolve_array[i, j] = self.calc_sum(basic_array[i, j])
                else:
                    uniform_revolve_array[i, j] = 9
        uniform_revolve_array = uniform_revolve_array.reshape(1, -1)
        uniform_revolve_array = torch.FloatTensor(uniform_revolve_array)
        return uniform_revolve_array


def color_moments(img):
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean()) ** 3)
    s_skewness = np.mean(abs(s - s.mean()) ** 3)
    v_skewness = np.mean(abs(v - v.mean()) ** 3)
    h_thirdMoment = h_skewness ** (1. / 3)
    s_thirdMoment = s_skewness ** (1. / 3)
    v_thirdMoment = v_skewness ** (1. / 3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])
    color_feature = torch.tensor(color_feature)
    color_feature = color_feature.reshape(1, -1)

    return color_feature


import sys
import copy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import CLIP.clip as clip
sys.path.append("..")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device2="cuda:2"

# model_id = "openai/clip-vit-base-patch32"
model_id1="ViT-B/32"
model_id2="ViT-B/16"
# model_id="RN50x4"

# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model1, processor1 = clip.load(model_id1, device=device, jit=False)
model2, processor2 = clip.load(model_id2, device=device2, jit=False)
cos_model = CosineSimilarityTest().to(device)
# 文本
with open(r"/workspace/segment-anything-main/class_data/all_classnames.json", 'r') as f_in:
    train_class_indexes = json.load(f_in)
prompt_templates = 'A  photo of {}'
text_prompts=zeroshot_classifier(train_class_indexes,prompt_templates)

sam_checkpoint = "/data/ZAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)
# mask_generator = SamAutomaticMaskGenerator(
#    model=sam,
#    points_per_side=32,
#    pred_iou_thresh=0.86,
#    stability_score_thresh=0.92,
#    crop_n_layers=1,
#    crop_n_points_downscale_factor=2,
#    min_mask_region_area=100,  # Requires open-cv to run post-processing
# )
print('start')
have_gen = os.listdir(r"/data/gen_coco/")
index = os.listdir(r'/data/entity_data/train2017').index('000000141426.jpg')
#print(index)
for x in os.listdir(r'/data/entity_data/train2017')[index:100]:
    print(x)
    x2 = x.split(".")[0] + ".png"
    if x2 in have_gen:
        print("jump")
        continue
    image = cv2.imdecode(np.fromfile(r"/data/entity_data/train2017/{}".format(x), dtype=np.uint8), cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('gen masks')
    masks = mask_generator.generate(image)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print('success')
    masks = del_overlap(masks)  # 预处理

    crop_list = gen_crop(masks)  # 剪裁crop

    nerbor = get_nerbor(masks)  # 相邻区域
    try:
        similar = sim_crop(crop_list, cos_model, model1)  # 余弦相似性
    except:
        file = open("/data/ignore.txt", "a")
        file.seek(0,2)
        file.write("{}\n".format(x))
        file.close()
        continue
    scores = get_clip_score(text_prompts, crop_list, model1, model2, processor1, processor2, masks, False)  # clip得分
    count = 0
    while (np.max(similar) != 0):
        max_sim = np.max(similar)
        i = int(np.argmax(similar) / len(crop_list))  # 行
        j = np.argmax(similar) % len(crop_list)  # 列
        similar[i][j] = 0
        similar[j][i] = 0
        if i in nerbor.keys():
            if j in nerbor[i]:  # 若相邻
                base_score_i = scores[i]['score']
                base_score_j = scores[j]['score']
                if base_score_i > 0.85 or base_score_j > 0.85 or base_score_i == 0 or base_score_j == 0:
                    continue
                _, _, new_seg = com_mask(i, j, masks)
                # 计算颜色和纹理相似性
                h = np.array(Image.fromarray(crop_list[i]).convert('L').resize((100, 100)))
                h2 = np.array(Image.fromarray(crop_list[j]).convert('L').resize((100, 100)))
                lbp = LBP()
                wenli1 = lbp.lbp_revolve_uniform(h)

                wenli2 = lbp.lbp_revolve_uniform(h2)
                cos_sim_wenli = torch.cosine_similarity(wenli1, wenli2)

                color_1 = color_moments(crop_list[i])

                color_2 = color_moments(crop_list[j])
                cos_sim_color = torch.cosine_similarity(color_1, color_2)
                all_sim = (float(cos_sim_wenli) + float(cos_sim_color)) / 2
                #             print(all_sim)
                new_crop = gen_crop(new_seg)
                new_score = get_clip_score(text_prompts, new_crop, model1, model2, processor1, processor2, masks, True)
                if (new_score[0]['score'] - base_score_i >= 0.12 and new_score[0][
                    'score'] - base_score_j >= 0.12 and all_sim > 0.75) or (
                        new_score[0]['score'] - base_score_i >= 0.03 and new_score[0][
                    'score'] - base_score_j >= 0.03 and (
                                new_score[0]['class'] == scores[i]['class'] or new_score[0]['class'] == scores[i][
                            'class']) and all_sim > 0.8):
                    print(base_score_i, base_score_j, all_sim)
                    print(new_score[0]['score'], new_score[0]['class'])
                    count += 1
                    print("有{}个".format(count))
                    masks[i] = new_seg[0]
                    masks.pop(j)
                    crop_list[i] = new_crop[0]
                    crop_list.pop(j)
                    similar = sim_crop(crop_list, cos_model, model1)  # 余弦相似性
                    # 重新计算score矩阵？

                    scores[i] = new_score[0]
                    scores.pop(j)
                    nerbor = get_nerbor(masks)  # 相邻区域
    will_del = []
    for i, mask in enumerate(masks):
        if scores[i]['score'] < 0.45 or masks[i]['area'] < 300:
            will_del.append(i)
    masks = [i for num, i in enumerate(masks) if num not in will_del]  # 枚举法删除，del和remove容易出错
    scores = [i for num, i in enumerate(scores) if num not in will_del]  # 枚举法删除，del和remove容易出错
    crop_list = [i for num, i in enumerate(crop_list) if num not in will_del]  # 枚举法删除，del和remove容易出错
    del will_del
    if len(masks) == 0:
        continue
    re = show_anns(masks, scores)

    plt.figure(figsize=(8, 4), dpi=100)
    plt.axis('off')
    plt.imshow(re)
    plt.savefig("/data/gen_coco/{}.png".format(x.split('.')[0]), bbox_inches='tight', pad_inches=0)
    plt.close()
    #cv2.imwrite("/data/gen_coco/{}.png".format(x.split('.')[0]),re)
    #cv2.imwrite('/data/cjf/data/orgin/{}'.format(x), image2)
