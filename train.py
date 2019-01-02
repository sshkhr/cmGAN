from datasets import SYSU_triplet_dataset, SYSU_eval_datasets

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

import itertools

from models_acmr import FeatureGenerator, FeatureProjector, Discriminator
from triplet_loss import TripletLoss
from logger import Logger

######################## Get Datasets & Dataloaders ###########################

transforms_list = [transforms.CenterCrop(224), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

train_dataset = SYSU_triplet_dataset(transforms_list=transforms_list)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
    num_workers=4, drop_last = True)

eval_train = SYSU_eval(data_split='train') #Query, Gallery Images from train IDs
eval_val = SYSU_eval(data_split='val') # Query, Gallery Images from val IDs

transform_test = transforms.Compose([
        transforms.Resize((300,100)), 
        transforms.ToTensor(),
    ])

queryloader = DataLoader(
    ImageDataset(eval_val.query, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=4,
    drop_last=False,
)

galleryloader = DataLoader(
    ImageDataset(eval_val.gallery, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=4,
    drop_last=False,
)

queryloader_train = DataLoader(
    ImageDataset(eval_train.query, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=0,
    drop_last=False,
)

galleryloader_train = DataLoader(
    ImageDataset(eval_train.gallery, transform=transform_test),
    batch_size=32, shuffle=False, num_workers=0,
    drop_last=False,
)

##################################### Import models ###########################

feature_generator = FeatureGenerator()
feature_projector = FeatureProjector()
mode_classifier = Discriminator()

if torch.cuda.is_available():
   feature_generator.cuda()
   feature_projector.cuda()
   mode_classifier.cuda()

############################# Get Losses & Optimizers #########################

criterion_triplet = TripletLoss(margin = 1.0)
criterion_identity = torch.nn.CrossEntropyLoss()
criterion_modality = torch.nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(itertools.chain(feature_generator.parameters(), 
    feature_projector.parameters()), lr = 0.001, betas=(0.5, 0.999))

optimizer_D = torch.optim.Adam(mode_classifier.parameters(), 
    lr = 0.0001, betas=(0.5, 0.999))

###############################################################################


# Loss plot
logger = Logger(202, len(dataloader))

for epoch in range(0, 202):
    
    print("Epoch ---------------", epoch+1)
    for i, batch in enumerate(dataloader):

        #print("Batch number ",i)

        anchor, positive, negative, label, modality = batch
        
        #anchor = Variable(anchor_tensor.copy_(batch[0]))
        #positive = Variable(positive_tensor.copy_(batch[1]))
        #negative = Variable(negative_tensor.copy_(batch[2]))
        #label = Variable(label_tensor.copy_(batch[3]))
        #modality = Variable(modality_tensor.copy_(batch[4]))


        if torch.cuda.is_available():
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            label = label.cuda()
            modality = modality.cuda()


        # Generator
        
        optimizer_G.zero_grad()    
            
        anchor_features = feature_generator(anchor)
        positive_features = feature_generator(positive)
        negative_features = feature_generator(negative)

        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        predicted_id = feature_projector(anchor_features)
        identity_loss = criterion_identity(predicted_id, label)

        loss_G = identity_loss + triplet_loss 

        loss_G.backward()
        optimizer_G.step()

        # Discriminator

        optimizer_D.zero_grad()

        anchor_features = feature_generator(anchor)

        predicted_modality = mode_classifier(anchor_features)
        discriminative_loss = criterion_modality(predicted_modality, modality)

        loss_D = discriminative_loss

        loss_D.backward()
        optimizer_D.step()

        logger.log({'loss_G': loss_G,  'loss_D': loss_D, 'loss_triplet': triplet_loss, 'loss_identity': identity_loss})

    if(epoch % 5 == 0):
        torch.save(feature_generator.state_dict(), 'saved/feature_generator_'+str(epoch)+'.pth')
        torch.save(feature_projector.state_dict(), 'saved/feature_projector_'+str(epoch)+'.pth')
        torch.save(mode_classifier.state_dict(), 'saved/discriminator_'+str(epoch)+'.pth')
        