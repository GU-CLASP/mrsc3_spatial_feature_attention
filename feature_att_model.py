dev = False

## command-line args
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-r', dest='resume', type=int)
parser.add_argument('-e', dest='epochs', type=int, default=10)
parser.add_argument('-g', dest='gpu', type=int, default=0)
parser.add_argument('-b', dest='batch_size', type=int, default=128)
parser.add_argument('-d', dest='development', action='store_true')

args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
dev = args.development

from pathlib import Path

## library
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import numpy as np
import json
from PIL import Image
from collections import Counter
import gc

from matplotlib import pyplot as plt

## load from files
base_url = 'https://cs.stanford.edu/people/rak248/'

images = {
    item['image_id']: {
        'width'  : item['width'],
        'height' : item['height'],
        'path'   : item['url'].replace(base_url, 'visual_genome/data/')
    }
    for item in json.load(open('visual_genome/data/image_data.json'))
}

# read from file
rels_from_file = json.load(open('visual_genome/data/relationships.json'))

# name/names correction for reading content of nodes in the dataset
def name_extract(x):
    if 'names' in x and len(x['names']):
        name = x['names'][0]
    elif 'name' in x:
        name = x['name']
    else:
        name = ''
    return name.strip().lower()

# convert it into a set of (image, subject, predicate, object)
triplets_key_values = {
    (
        rels_in_image['image_id'],
        name_extract(rel['subject']),
        rel['predicate'].lower().strip(),
        name_extract(rel['object']),
    ): (
        rels_in_image['image_id'],
        (name_extract(rel['subject']), rel['subject']['object_id'], (rel['subject']['x'],rel['subject']['y'],rel['subject']['w'],rel['subject']['h'])),
        rel['predicate'].lower().strip(),
        (name_extract(rel['object']), rel['object']['object_id'], (rel['object']['x'], rel['object']['y'], rel['object']['w'], rel['object']['h'])),
    )
    for rels_in_image in rels_from_file
    for rel in rels_in_image['relationships']
}
triplets = list(triplets_key_values.values())
del triplets_key_values

image_ids = list(np.load('visual_genome/data/relationships/image_ids.npy'))

if dev:
    image_ids = image_ids[:32]

chunck_size = 10000
img_visual_features = []
for l in range(0, len(image_ids), chunck_size):
    vfs = np.load('visual_genome/data/relationships/image_resnet50_features_['+str(l)+'].npy')
    img_visual_features += list(zip(image_ids[l:l+chunck_size], vfs))
    del vfs

img_visual_features = dict(img_visual_features)


# clean the data from examples in which there is no saved vector for them
triplets = [item for item in triplets if item[0] in img_visual_features and type(img_visual_features[item[0]]) != int]

object_ids = list(np.load('visual_genome/data/relationships/object_ids.npy'))

chunck_size = 100000
visual_features = []
for l in range(0, len(object_ids), chunck_size):
    vfs = np.load('visual_genome/data/relationships/objects_resnet50_features_['+str(l)+'].npy')
    visual_features += list(zip(object_ids[l:l+chunck_size], vfs))

visual_features = dict(visual_features)

#vocab = Counter([w.strip() for _,(sbj,_,_),pred,(obj,_,_) in triplets for w in ' '.join([sbj,pred,obj]).split(' ')])
#np.save('visual_genome/data/relationships/vocab_caption.npy', vocab)
vocab = np.load('visual_genome/data/relationships/vocab_caption.npy')[None][0]

word2ix = {w:i for i,w in enumerate(['<0>', '<s>']+list(vocab))}
ix2word = {i:w for w,i in word2ix.items()}
word2onehot = lambda w: np.array([0.]*word2ix[w] + [1.] + [0.]*(len(word2ix)-word2ix[w]-1))

max_len = max(len(' '.join([sbj,pred,obj]).split(' ')) for _,(sbj,_,_),pred,(obj,_,_) in triplets)

# keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Flatten, AveragePooling2D
from keras.layers import Dense, LSTM, Embedding, Masking
from keras.layers import Input, Lambda, RepeatVector, Reshape
from keras.layers import TimeDistributed
from keras import backend as K


def item2features(item):
    img_id,(sbj,object_id1,sbj_bbx),pred,(obj,object_id2,obj_bbx) = item

    # visual features
    vf0 = img_visual_features[img_id]
    vf1 = visual_features[object_id1]
    vf2 = visual_features[object_id2]

    # spatial features
    # based on VisKE
    a1 = sbj_bbx[2] * sbj_bbx[3]
    a2 = obj_bbx[2] * obj_bbx[3]
    if obj_bbx[0] <= sbj_bbx[0] <= obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]:
        # overlap
        w = (obj_bbx[0]+obj_bbx[2]) - (sbj_bbx[0])
    elif obj_bbx[0] <= sbj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]:
        # obj contains sbj
        w = sbj_bbx[2]
    elif sbj_bbx[0] <= obj_bbx[0] <= sbj_bbx[0] + sbj_bbx[2] <= obj_bbx[0]+obj_bbx[2]:
        # overlaps
        w = (sbj_bbx[0]+sbj_bbx[2]) - (obj_bbx[0])
    elif sbj_bbx[0] <= obj_bbx[0] <= obj_bbx[0]+obj_bbx[2] <= sbj_bbx[0] + sbj_bbx[2]:
        # subj contains obj
        w = obj_bbx[2]
    else:
        w = 0

    if obj_bbx[1] <= sbj_bbx[1] <= obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]:
        # overlap
        h = (obj_bbx[1]+obj_bbx[3]) - (sbj_bbx[1])
    elif obj_bbx[1] <= sbj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] <= obj_bbx[1]+obj_bbx[3]:
        # obj contains sbj
        h = sbj_bbx[3]
    elif sbj_bbx[1] <= obj_bbx[1] <= sbj_bbx[1] + sbj_bbx[3] <= obj_bbx[1]+obj_bbx[3]:
        # overlaps
        h = (sbj_bbx[1]+sbj_bbx[3]) - (obj_bbx[1])
    elif sbj_bbx[1] <= obj_bbx[1] <= obj_bbx[1]+obj_bbx[3] <= sbj_bbx[1] + sbj_bbx[3]:
        # subj contains obj
        h = obj_bbx[3]
    else:
        h = 0

    overlap_a = w * h

    # dx; dy; ov; ov1; ov2; h1;w1; h2;w2; a1; a2
    sf1 = [
        #obj_bbx[0] - sbj_bbx[0], # dx = x2 - x1 
        #obj_bbx[1] - sbj_bbx[1], # dy = y2 - y1
        obj_bbx[0] - sbj_bbx[0] + (obj_bbx[2] - sbj_bbx[2])/2, # dx = x2 - x1 + (w2 - w1)/2
        obj_bbx[1] - sbj_bbx[1] + (obj_bbx[3] - sbj_bbx[3])/2, # dy = y2 - y1 + (h2 - h1)/2
        0 if (a1+a2) == 0 else overlap_a/(a1+a2), # ov
        0 if a1 == 0 else overlap_a/a1, # ov1
        0 if a2 == 0 else overlap_a/a2, # ov2
        sbj_bbx[3], # h1
        sbj_bbx[2], # w1
        obj_bbx[3], # h2
        obj_bbx[2], # w2
        a1, # a1
        a2, # a2
    ]
    
    # spatial template (two attention masks)
    x1, y1, w1, h1 = sbj_bbx
    x2, y2, w2, h2 = obj_bbx

    mask = np.zeros([7,7,2])
    mask[int(y1*7):int((y1+h1)*7), int(x1*7):int((x1+w1)*7), 0] = 1 # mask bbox 1 
    mask[int(y2*7):int((y2+h2)*7), int(x2*7):int((x2+w2)*7), 0] = 1 # mask bbox 2

    sf2 = mask.flatten()
    
    # sentence encoding
    sent = ' '.join([sbj,pred,obj]).split(' ')
    sent = [word2ix['<s>']]+[word2ix[w] for w in sent]+[word2ix['<0>']]*(1+max_len-len(sent))

    return vf0, sf1, sf2, vf1, vf2, sent
    
def generator_features_description(batch_size=32, split=(0.,1.), all_data = triplets, mode='bbox'):
    while True:
        gc.collect()
        
        # shuffle 
        _all_data = all_data[int(len(all_data)*split[0]):int(len(all_data)*split[1])]
        np.random.shuffle(_all_data)
        
        # start
        X_vfs = []
        X_sfs = []
        X_objs = []
        X_sents = []
        
        for item in _all_data:
            vf0, sf1, sf2, vf1, vf2, sent = item2features(item)
            
            X_vfs.append(vf0)
            X_sents.append(sent)
            
            if mode[:4] == 'bbox' or mode[-4:] == 'bbox':
                X_sfs.append(sf1)
            elif mode[:9] == 'attention':
                X_sfs.append(sf2)
            
            if mode[:4] == 'bbox' or mode[:9] == 'attention' or mode[:8] == 'implicit':
                l = [vf1, vf2]
                if mode[-2:] == '-r':
                    np.random.shuffle(l)
                X_objs.append(l)

            if len(X_sents) == batch_size:
                sents = np.array(X_sents)
                if mode[:4] == 'bbox' or mode[:9] == 'attention':
                    yield ([np.array(X_vfs), np.array(X_sfs), np.array(X_objs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                elif mode[:8] == 'implicit':
                    yield ([np.array(X_vfs), np.array(X_objs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                elif mode == 'spatial_adaptive-bbox':
                    yield ([np.array(X_vfs), np.array(X_sfs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))

                elif mode[:7] == 'no-beta' or mode == 'spatial_adaptive' or mode == 'spatial_adaptive-attention':
                    yield ([np.array(X_vfs), sents[:, :-1]], np.expand_dims(sents[:, 1:], 2))
                    
                X_vfs = []
                X_sfs = []
                X_objs = []
                X_sents = []


def build_model(mode='bbox'):
    print('mode:', mode)
    
    unit_size = 100
    regions_size = 7 * 7
    beta_size = 2 + 1 + 1 # 2 objects + 1 sentential + 1 spatial

    delayed_sent = Input(shape=[max_len+1])
    
    sf_size = 11 # dx; dy; ov; ov1; ov2; h1; w1; h2; w2; a1; a2 (from VisKE)
    beta_feature_size = 2*(beta_size-1)*unit_size
    
    if mode[:9] == 'attention':
        sf_size = 49*2  # attention mask pattern
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode[:4] == 'bbox':
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode[:8] == 'implicit':
        beta_size = 2 + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive':
        beta_size = regions_size + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive-bbox':
        beta_size = regions_size + 1 + 1
        beta_feature_size = 2*(beta_size-1)*unit_size
    elif mode == 'spatial_adaptive-attention':
        sf_size = 49
        beta_size = regions_size + 1 + 1
        beta_feature_size = 2*(beta_size-2)*unit_size
        
    visual_features_in0 = Input(shape=[regions_size, 2048]) # resnet50
    visual_features_objs_in = Input(shape=[2, 2048]) # resnet50
    spatial_features_in = Input(shape=[sf_size]) 

    embeddings = Embedding(len(word2ix), unit_size)(Masking()(delayed_sent))
    
    # fine tune / project features
    mlp_vision = Dense(unit_size, activation='relu')
    mlp_space  = Sequential([
        Dense(unit_size, activation='tanh', input_shape=[sf_size]),
        Dense(unit_size, activation='relu'),
    ])
    mlp_att   = Sequential([
        TimeDistributed(Dense(unit_size, activation='relu'), input_shape=[max_len+1, beta_feature_size]),
        TimeDistributed(Dense(unit_size, activation='tanh')),
        TimeDistributed(Dense(beta_size, activation='softmax')),
    ])
    
    ### global visual features
    visual_features0 = TimeDistributed(mlp_vision)(visual_features_in0) # learn to find
    visual_features0_g = Reshape([7,7,unit_size])(visual_features0)
    visual_features0_g = Flatten()(AveragePooling2D([7,7])(visual_features0_g))
    
    ### objects visual features
    visual_features_objs = TimeDistributed(mlp_vision)(visual_features_objs_in)
    
    spatial_features = mlp_space(spatial_features_in)

    ### adaptive attention: beta (which feature set needs more attention?)
    def feature_fusion(x, regions_size=regions_size, max_len=max_len):
        return K.concatenate([
            x[0],
            K.repeat_elements(K.expand_dims(x[1], 1), max_len+1, 1),
        ], 2)
    
    def beta_features(x, unit_size=unit_size, max_len=max_len, beta_size=beta_size):
        if mode[:8] == 'implicit' or mode == 'spatial_adaptive' or mode=='spatial_adaptive-attention':
            h, vf0 = x
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1) # [sent, 49, unit_size] or [sent, 2, unit_size]
            if mode=='spatial_adaptive-attention':
                h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-2, 2) # [sent, 49, unit_size]
                return K.reshape(K.concatenate([h_, vf0_], 3), [-1, max_len+1, 2*(beta_size-2)*unit_size]) # [sent, 49*b*unit_size]
            else:
                h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-1, 2) # [sent, 49, unit_size]
                return K.reshape(K.concatenate([h_, vf0_], 3), [-1, max_len+1, 2*(beta_size-1)*unit_size]) # [sent, 49*b*unit_size]
        else:
            h, sf, vf0 = x
            sf_  = K.expand_dims(K.repeat_elements(K.expand_dims(sf, 1), max_len+1, 1), 2) # [sent, 1, unit_size]
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1) # [sent, 49, unit_size] or [sent, 2, unit_size]
            vf_sf = K.concatenate([sf_,vf0_],2) # [sent, 49+1, unit_size] or [sent, 2+1, unit_size]
            
            h_   = K.repeat_elements(K.expand_dims(h, 2), beta_size-1, 2) # [sent, 49+1, unit_size] or or [sent, 2+1, unit_size]

            return K.reshape(K.concatenate([h_, vf_sf], 3), [-1, max_len+1, 2*(beta_size-1)*unit_size]) # [sent, 49+1*b*unit_size]

    
    ### use adaptive attention beta

    def adaptation_attention(x, max_len=max_len, regions_size=regions_size, mode=mode):
        if mode[:8] == 'implicit' or mode == 'spatial_adaptive':
            h, vf0, b = x
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1)
            
            return b[:, :, 0:1] * h + K.sum(K.expand_dims(b[:, :, 1:], 3) * vf0_, 2)
        else:
            h, sf, vf0, b = x
            if len(sf.get_shape()) == 2:
                sf_ = K.repeat_elements(K.expand_dims(sf, 1), max_len+1, 1)
            else:
                sf_ = sf
            vf0_ = K.repeat_elements(K.expand_dims(vf0, 1), max_len+1, 1)
            
            return b[:, :, 0:1] * h + b[:, :, 1:2] * sf_ + K.sum(K.expand_dims(b[:, :, 2:], 3) * vf0_, 2)
    
    fused_features = Lambda(feature_fusion)([embeddings, visual_features0_g])
    hidden_a       = LSTM(unit_size, return_sequences=True)(fused_features)
    ling_features  = LSTM(unit_size, return_sequences=True)(hidden_a)

    if mode[:4] == 'bbox' or mode[:9] == 'attention':
        beta_feaures_out = Lambda(beta_features)([hidden_a, spatial_features, visual_features_objs])
        beta = mlp_att(beta_feaures_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features_objs, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, spatial_features_in, visual_features_objs_in, delayed_sent], out)
    elif mode[:8] == 'implicit':
        beta_feaures_out = Lambda(beta_features)([hidden_a, visual_features_objs])
        beta = mlp_att(beta_feaures_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, visual_features_objs, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, visual_features_objs_in, delayed_sent], out)
    elif mode[:7] == 'no-beta':
        out = Dense(len(word2ix), activation='softmax')(fused_features)
        model = Model([visual_features_in0, delayed_sent], out)
    elif mode == 'spatial_adaptive':
        beta_feaures_out = Lambda(beta_features)([hidden_a, visual_features0])
        beta = mlp_att(beta_feaures_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, delayed_sent], out)
    elif mode == 'spatial_adaptive-bbox':
        beta_feaures_out = Lambda(beta_features)([hidden_a, spatial_features, visual_features0])
        beta = mlp_att(beta_feaures_out)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, spatial_features_in, delayed_sent], out)
    elif mode == 'spatial_adaptive-attention':
        beta_feaures_out = Lambda(beta_features)([hidden_a, visual_features0])
        beta = mlp_att(beta_feaures_out)
        beta_spatial = Lambda(lambda x: x[:, :, 2:])(beta)
        spatial_features = mlp_space(beta_spatial)
        adapted_feaures = Lambda(adaptation_attention)([ling_features, spatial_features, visual_features0, beta])
        out = Dense(len(word2ix), activation='softmax')(adapted_feaures)
        model = Model([visual_features_in0, delayed_sent], out)

    model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    return model


dir_path = 'visual_genome/data/relationships/'
if dev:
    dir_path = 'visual_genome/data/relationships/dev_'
    batch_size = 16
    
modes = [
    'no-beta',
    'implicit',
    'spatial_adaptive',
    'bbox',
    'attention',
    'spatial_adaptive-bbox',
    'spatial_adaptive-attention',
    'bbox-r',
    'attention-r',
    'implicit-r',
]

for mode in modes:
    if args.resume is not None and Path(dir_path + 'caption_model_{0}_{1}epochs.h5'.format(mode, args.resume)).is_file() and Path(dir_path + 'caption_model_{0}_{1}epochs_history.npy'.format(mode, args.resume)).is_file() and args.resume <= args.epochs:
        if args.resume == args.epochs:
            continue
        # if file exists load it to resume:
        model = load_model(dir_path + 'caption_model_{0}_{1}epochs.h5'.format(mode, args.resume))
        old_history = np.load(dir_path + 'caption_model_{0}_{1}epochs_history.npy'.format(mode, args.resume))[None][0]
        
        h = model.fit_generator(
            generator=generator_features_description(batch_size=batch_size, split=(0.,0.95), mode=mode), 
            steps_per_epoch=int(len(triplets)*0.95/128) if not dev else 10, 
            validation_data=generator_features_description(batch_size=batch_size, split=(0.95,1.), mode=mode),
            validation_steps=int(len(triplets)*0.05/128) if not dev else 10,
            epochs=args.epochs - args.resume,
        )
        
        model.save(dir_path + 'caption_model_{0}_{1}epochs.h5'.format(mode, args.epochs))
        np.save(dir_path + 'caption_model_{0}_{1}epochs_history.npy'.format(mode, args.epochs), {
            t: [v for v in old_history[t]+h.history[t]]
            for t in old_history
        })
    else:
        # start from scrach:
        model = build_model(mode=mode)
        h = model.fit_generator(
            generator=generator_features_description(batch_size=batch_size, split=(0.,0.95), mode=mode), 
            steps_per_epoch=int(len(triplets)*0.95/128) if not dev else 10, 
            validation_data=generator_features_description(batch_size=batch_size, split=(0.95,1.), mode=mode),
            validation_steps=int(len(triplets)*0.05/128) if not dev else 10,
            epochs=args.epochs,
        )
    
        model.save(dir_path + 'caption_model_{0}_{1}epochs.h5'.format(mode, args.epochs))
        np.save(dir_path + 'caption_model_{0}_{1}epochs_history.npy'.format(mode, args.epochs), h.history)
