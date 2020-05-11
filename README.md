# Detection of Physical Adversarial Attacks

Goal: To create a detection algorithm against physical adversarial attack with adjustable false positive rate.

Method: The base of detection mechanism is based upon the lack of robustness of adversarial examples. \\



Directory:

1. Data Folder: Contains 4 folders.\
Images_FINAL: Original Data\
Images_Adv_FINAL: Data with sticker\
Images_Aligned_FINAL: Aligned Data (600 * 600)\
Images_Adv_Aligned_FINAL: Aligned Adversarial Data (600 * 600)\

2. Models: Load 2 models\
https://drive.google.com/file/d/1fb70KgMRSmaEUF5cJ67BCD_DmTPCR5uJ/view (r100.pb)\
resnet18_110.pth from https://github.com/ronghuaiyang/arcface-pytorch (resnet18_110.pth)\



