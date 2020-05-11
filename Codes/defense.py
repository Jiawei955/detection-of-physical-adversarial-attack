import torch
import torch.nn.functional as F
from torchvision import transforms as T
import os
from PIL import Image
from math import ceil
import numpy as np
from models.resnet import resnet_face18, resnet18
from torch.nn import DataParallel


def load_net(path):
    """
    load arcface model in pytorch
    :return:
    """
    model = resnet_face18(use_se=False)
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def noisy(img,sigma):
    return img+sigma*torch.randn_like(img)

def l1_distance(model, img, sigma):
    return torch.norm(model(img) - model(noisy(img, sigma)), 1).item()


def detect(adv_dir,model,sigma,num,transform,threshold):
    """

    :param adv_dir: dirctory of adv examples
    :param model: arcface net
    :param sigma: variance of Gaussian noise
    :param num: number of samples for one adv pic
    :param batch_size:
    :return:
    """
    model.eval()
    count = 0
    adv_img = os.listdir(adv_dir)
    total_adv = 0
    for p in adv_img:
        if p[0] == '.':
            continue
        l1_val = []
        for i in range(num):
            adv_img_path = os.path.join(adv_dir,p)
            img = Image.open(adv_img_path)
            img = transform(img)
            # img_noise = noisy(img,sigma)
            distance = l1_distance(model,img,sigma)
            l1_val.append(distance)
        np.asarray(l1_val)
        mean_distance = np.mean(l1_val)
        if mean_distance>threshold:
            count+=1
        total_adv += 1
    tpr = float(count)/float(total_adv)
    return tpr,count # the number of adversary examples being successfully detected


def get_natural_acc(natural_dir,model,sigma,num,transform,threshold):
    """

    :param adv_dir: dirctory of adv examples
    :param model: arcface net
    :param sigma: variance of Gaussian noise
    :param num: number of samples for one adv pic
    :param batch_size:
    :return:
    """
    model.eval()
    count = 0
    nat_img = os.listdir(natural_dir)
    total_imgs = 0
    for p in nat_img:
        if p[0] == '.':
            continue
        l1_val = []
        for i in range(num):
            adv_img_path = os.path.join(natural_dir,p)
            img = Image.open(adv_img_path)
            img = transform(img)
            # img_noise = noisy(img,sigma)
            img = img.unsqueeze(0)
            distance = l1_distance(model,img,sigma)
            l1_val.append(distance)
        np.asarray(l1_val)
        mean_distance = np.mean(l1_val)
        if mean_distance>threshold:
            count+=1
        total_imgs += 1
    fpr = float(count)/float(total_imgs)
    return fpr,count # the number of natural examples being wrongly detected (false positive)


def main():
    sigma = 0.3
    sample_num = 100
    threshold = 7
    modle_path = 'resnet18_110.pth'
    model = resnet_face18(use_se=False)
    model = DataParallel(model)
    model.load_state_dict(torch.load(modle_path,map_location=torch.device('cpu')))
    # model.to(torch.device("cuda"))
    # model = load_net(modle_path) # do not use this function !!!
    transform = T.Compose([
        T.Resize(size=(120, 120)),
        T.Grayscale(),
        T.ToTensor()
    ])
    natural_dir = 'Data_Nat'
    #adv_dir = 'Data_Adv'      # path to the dir of adversarial examples
    fpr, count_nat = get_natural_acc(natural_dir,model,sigma,sample_num,
                                     transform,threshold)
    #tpr, count_adv = detect(adv_dir,model,sigma,sample_num, transform,threshold)
    print(f"At threshold{threshold}, sigma{sigma}, we get natural example: {count_nat}")
    #print(f"At threshold{threshold}, sigma{sigma}, we get natural example: {count_adv}")
    
          # f"tpr: {tpr:.2f},  adversarial example: {count_adv}, fpr: {fpr:.2f}")

    

if __name__ == '__main__':
    main()

