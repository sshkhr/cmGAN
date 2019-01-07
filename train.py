from datasets import SYSU_triplet_dataset, SYSU_eval_datasets, Image_dataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

import itertools

from models import FeatureGenerator, IdClassifier, ModalityClassifier
from logger import Logger

from eval import test, evaluate

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

######################## Get Datasets & Dataloaders ###########################

transforms_list = [transforms.CenterCrop(224), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

train_dataset = SYSU_triplet_dataset(transforms_list=transforms_list)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
    num_workers=4, drop_last = True)

eval_train = SYSU_eval_datasets(data_split='train') 
eval_val = SYSU_eval_datasets(data_split='val') 

transform_test = transforms.Compose([
        transforms.Resize((300,100)), 
        transforms.ToTensor(),
    ])

queryloader = DataLoader(
    Image_dataset(eval_val.query, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=4,
    drop_last=False,
)

galleryloader = DataLoader(
    Image_dataset(eval_val.gallery, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=4,
    drop_last=False,
)

queryloader_train = DataLoader(
    Image_dataset(eval_train.query, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=0,
    drop_last=False,
)

galleryloader_train = DataLoader(
    Image_dataset(eval_train.gallery, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=0,
    drop_last=False,
)

##################################### Import models ###########################

feature_generator = FeatureGenerator()
id_classifier = IdClassifier()
mode_classifier = ModalityClassifier()

if torch.cuda.is_available():
   feature_generator.cuda()
   id_classifier.cuda()
   mode_classifier.cuda()

############################# Get Losses & Optimizers #########################

criterion_triplet = torch.nn.TripletMarginLoss(margin = 1.4)
criterion_identity = torch.nn.CrossEntropyLoss()
criterion_modality = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD([
                {'params': feature_generator.parameters(), 'lr':1e-4},
                {'params': id_classifier.parameters(), 'lr':1e-4},
                {'params': mode_classifier.parameters(), 'lr': 1e-3}
            ], momentum=0.9)

#optimizer_G = torch.optim.Adam(itertools.chain(feature_generator.parameters(), 
#    feature_projector.parameters()), lr = 0.001, betas=(0.5, 0.999))

#optimizer_D = torch.optim.Adam(mode_classifier.parameters(), 
#    lr = 0.0001, betas=(0.5, 0.999))

############################# Hyper-parameters ################################

alpha = 1.0
beta = 1.0
gamma = 0.05
K = 5
nu = 1

################################  Train  ######################################


# Loss plot
logger = Logger(2000, len(train_dataloader))

test_ranks, test_mAP = test(feature_generator, queryloader, galleryloader)
train_ranks, train_mAP = test(feature_generator, queryloader_train, galleryloader_train)

for epoch in range(0, 2000):
    
    print("Epoch ---------------", epoch+1)
    for i, batch in enumerate(train_dataloader):

        #print("Batch number ",i)

        anchor_rgb, positive_rgb, negative_rgb, anchor_ir, positive_ir, \
        negative_ir, anchor_label, modality_rgb, modality_ir = batch
        
        if torch.cuda.is_available():
            anchor_rgb = anchor_rgb.cuda()
            positive_rgb = positive_rgb.cuda()
            negative_rgb = negative_rgb.cuda()
            anchor_ir = anchor_ir.cuda()
            positive_ir = positive_ir.cuda()
            negative_ir = negative_ir.cuda()
            anchor_label = anchor_label.cuda()
            modality_rgb = modality_rgb.cuda()
            modality_ir = modality_ir.cuda()

        optimizer.zero_grad() 

        # Generator
        
        anchor_rgb_features = feature_generator(anchor_rgb)
        positive_rgb_features = feature_generator(positive_rgb)
        negative_rgb_features = feature_generator(negative_rgb)

        anchor_ir_features = feature_generator(anchor_ir)
        positive_ir_features = feature_generator(positive_ir)
        negative_ir_features = feature_generator(negative_ir)

        triplet_loss_rgb = criterion_triplet(anchor_rgb_features, 
            positive_ir_features, negative_ir_features)
        
        triplet_loss_ir = criterion_triplet(anchor_ir_features, 
            positive_rgb_features, negative_rgb_features)

        triplet_loss = triplet_loss_rgb + triplet_loss_ir 
        
        predicted_id_rgb = id_classifier(anchor_rgb_features)
        predicted_id_ir = id_classifier(anchor_ir_features)

        identity_loss = criterion_identity(predicted_id_rgb, anchor_label) + \
                        criterion_identity(predicted_id_ir, anchor_label)

        loss_G = alpha*triplet_loss + beta*identity_loss 

        # Discriminator

        predicted_rgb_modality = mode_classifier(anchor_rgb_features)
        predicted_ir_modality = mode_classifier(anchor_ir_features)
        
        loss_D = criterion_modality(predicted_rgb_modality, modality_rgb) + \
                 criterion_modality(predicted_ir_modality, modality_ir)

        if epoch%K:
            loss_total = loss_G - gamma*loss_D
        else:
            loss_total = nu*(gamma*loss_D - loss_G)

        loss_total.backward()
        optimizer.step()

        
        logger.log(## Add losses
                   {'loss_G': loss_G,  'loss_D': loss_D, 'loss_triplet_rgb': \
                    triplet_loss_rgb, 'triplet_loss_ir': triplet_loss_ir, \
                    'loss_identity': identity_loss, 'loss_total': loss_total, \
                   },     
                   ## Add metrics
                   {'test_mAP_percentage': test_mAP*100.0, \
                    'test_rank-1_accuracy_percentage':test_ranks[0]*100.0, \
                    'train_mAP_percentage': train_mAP*100.0, \
                    'train_rank-1_accuracy_percentage':train_ranks[0]*100.0
                   })

    test_ranks, test_mAP = test(feature_generator, queryloader, galleryloader)
    train_ranks, train_mAP = test(feature_generator, queryloader_train, galleryloader_train)


    if(epoch % 5 == 0):
        torch.save(feature_generator.state_dict(), 'saved/feature_generator_'+str(epoch)+'.pth')

        with open(os.path.join('saved','stats_ep' + str(logger.epoch) +'.txt'), 'w') as file:
            file.write("mAP" + " = %s\n" % test_mAP)   
            for index, item in enumerate(test_ranks):
                file.write("Rank-" + str(index+1)+" = %s\n" % item)