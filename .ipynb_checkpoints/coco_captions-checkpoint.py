
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import json

path = 'F:/$daten/datasets/coco_train2017/'
with open(path + 'captions_train2017.json') as f:
    data = json.load(f)

captions = dict()
for dan in data['annotations']:
    id = dan['image_id']
    cap = dan['caption']
    if not id in captions:
        captions[id] = []
    captions[id].append(cap)

with open('encoded_captions.dat', 'rb') as f:
    encoded_captions = pkl.load(f)

N = len(captions)


def showimg(im):
    if im.shape[0]==3:
        im = im.transpose(1, 2, 0)
    plt.imshow(im, extent=(0,1,0,1))
    plt.axis('off')
    #plt.show()


def getimg(k):
    fn = data['images'][k]['file_name']
    im = plt.imread(path+'128x128/'+fn) / 255
    if len(im.shape)==2:
        im = im[:,:,None]
    return im

def getfeatures(k):
    fn = data['images'][k]['file_name']
    try:
        return np.load(path+'resnet-features/'+fn+'.npy')
    except:
        return np.zeros((2048, 7, 7)) 

def getcaptions(k):
    id = data['images'][k]['id']
    return captions[id]

def getRandomEncodedCaption(k):
    cs = encoded_captions[k]
    i = np.random.randint(len(cs))
    return cs[i]


def randomcrop(x, hw):
    dy = np.random.randint(x.shape[0]-hw) if x.shape[0]>hw else 0
    dx = np.random.randint(x.shape[1]-hw) if x.shape[1]>hw else 0
    x = x[dy:dy+hw, dx:dx+hw, :]
    return x