
import torchvision.transforms as transforms
import glob
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


# x = Image.open('/home/jb/Documents/data/mimic-crx-images/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014_256.jpg')
# y = Image.open('./data/mimic-crx-images/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')
# a1 = (np.array(x))
# a2= (np.array(y))
# print(a1==a2)
# print((a1==a2).all())
# sys.exit()

x = glob.glob('data/**/*.jpg', recursive=True)
print(len(x))


t = transforms.Compose([
     transforms.Resize(256),
])

for im in tqdm(x):
    head, tail = os.path.split(im)
    head = head.replace('data','/home/jb/Documents/data')
    tail = tail.replace('.jpg', '_256.npy')
    if not os.path.exists(os.path.join(head, tail)):
        os.makedirs(head, exist_ok=True)
        np.save(os.path.join(head, tail), np.array(t(Image.open(im))).astype(np.uint8))