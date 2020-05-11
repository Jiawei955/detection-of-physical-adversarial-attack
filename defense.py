import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image
from math import ceil
import numpy as np
from PIL import Image
import glob


def load_net():
    """
    load arcface model in pytorch
    :return:
    """    print("Model")
    model = model.load_state_dict(torch.load('epochs/' + r100.pb, map_location=lambda storage, loc: storage)) 
    #model = torch.load('C:/Users/Heetika/Documents/AllMyProjects/advhat-master/advhat-master/Demo/r100.pb')
    
    pass

def noisy(img,sigma):
    return img+sigma*torch.randn_like(img)

def l1_distance(model, img, sigma,transform):
    for image in img:
       
        img = Image.open(image)
        
        trans = transforms.ToPILImage()
        trans1 = transforms.ToTensor()
        plt.imshow(trans(trans1(img)))

    
    a = torch.norm(model(transforms(img)) -
        model(transforms(noisy(img, sigma))), 1).item()
  
    
    return a

def detect(adv_dir,model,sigma,num,transform,threshold):
    """
    :param adv_dir: dirctory of adv examples
    :param model: arcface net
    :param sigma: variance of Gaussian noise
    :param num: number of samples for one adv pic
    :param batch_size:
    :return:
    """

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
            distance = l1_distance(model,img,sigma,transform)
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
    model = load_net() #the arcfacce here
    transform = transforms.ToTensor()
    
    #natural_dir = glob.glob("C:/Users/Heetika/Documents/AllMyProjects/advhat-master/advhat-master/Demo/Data_Nat") # path to the dir of natural
    natural_dir = 'C:/Users/Heetika/Documents/AllMyProjects/advhat-master/advhat-master/Demo/Data_Nat'
    adv_dir = glob.glob("C:/Users/Heetika/Documents/AllMyProjects/advhat-master/advhat-master/Demo/Data_Adv"  )    # path to the dir of adversarial 
    fpr, count_nat = get_natural_acc(natural_dir,model,sigma,sample_num,transform,threshold)
    tpr, count_adv = detect(adv_dir,model,sigma,sample_num,transform,threshold)
    print(f"at threshold{threshold}, sigma{sigma}, we get nat: {count_nat}, "
          f"tpr: {tpr:.2f},  nat: {count_adv}, fpr: {fpr:.2f}")


if __name__ == '__main__':
    main() 