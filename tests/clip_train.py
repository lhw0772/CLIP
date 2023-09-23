import torch
import clip
from PIL import Image
import os

import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm  # tqdm 라이브러리 임포트
import argparse
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

label_dict = {"chest": ['pleural_effusion','nodule','pneumonia','cardiomegaly','hilar_enlargement',
              'fracture_old','fibrosis','aortic_calcification','tortuous_aorta','thickened_pleura' ,'TB','pneumothorax',
              'emphysema','atelectasis','calcification','pulmonary_edema','increased_lung_markings',
              'elevated_diaphragm','consolidation'], "endo":['ulcer','erosion','polyp','tumor']}

#chest_prior_disc =[['pleural_effusion','pleura','small']]
chest_prior_disc = [["Fluid Accumulation", "Chest", "Opacity", "Gray", "Swelling", "Thoracic Cavity", "Localized", "Bilateral", "Unilateral", "Effusion"],
["Round", "Small", "Lung", "Lobe", "Gray", "Density", "Opacity", "Solitary", "Multiple", "Calcified", "Peripheral", "Central"],
["Infection", "Inflammation", "Consolidation", "Airspace", "Lung", "Bacterial", "Viral", "Alveoli", "Opacity", "Gray"],
["Enlarged", "Heart", "Cardiac", "Silhouette", "Cardiomegaly", "Hypertrophic", "Dilated", "Left Atrium", "Right Atrium", "Aortic Arch"],
["Hilar", "Enlargement", "Bilateral", "Unilateral", "Left Hilum", "Right Hilum", "Opacity", "Gray", "Lung", "Shadow"],
["Fracture", "Old", "Bone", "Healed", "Callus", "Cortical", "Trabecular", "Radiopaque", "Irregular", "Prior Injury"],
["Fibrosis", "Scar", "Lung", "Tissue", "Collagen", "Thickening", "Fibrous", "Opacity", "Gray", "Pulmonary"],
["Aortic", "Calcification", "Calcified", "Vessel", "Artery", "Opacity", "White", "Cardiovascular", "Thoracic", "Aorta"],
["Tortuous", "Aorta", "Curved", "Vessel", "Artery", "Winding", "Abnormal", "Opacity", "Gray", "Cardiovascular"],
["Thickened", "Pleura", "Membrane", "Gray", "Opacity", "Lung", "Chest", "Abnormal", "Fibrous", "Adhesion"],
["Tuberculosis", "Infection", "Granuloma", "Opacity", "Lung", "Cavitary", "Consolidation", "Pulmonary", "Gray", "Bacterial"],
["Collapsed", "Lung", "Air", "Opacity", "Chest", "Pulmonary", "Transparent", "Gray", "Radiolucency", "Breathlessness"],
["Destruction", "Airspace", "Opacity", "Gray", "Pulmonary", "Alveoli", "Hyperinflation", "Bullae", "Breathing Difficulty", "COPD"],
["Collapsed", "Lung", "Opacity", "Pulmonary", "Gray", "Alveoli", "Volume Loss", "Breathing Difficulty", "Consolidation", "Aeration"],
["Tissue", "Hardening", "Opacity", "Gray", "Mineralization", "Calcified", "Vascular", "Artery", "Organ", "Pathology"],
["Fluid Accumulation", "Chest", "Opacity", "Gray", "Swelling", "Thoracic Cavity", "Localized", "Bilateral", "Unilateral", "Effusion"],
["Markings", "Lung", "Opacity", "Gray", "Pattern", "Bronchovascular", "Interstitial", "Radiodensity", "Diffuse", "Airspace"],
["Diaphragm", "Elevation", "Opacity", "Gray", "Chest", "Thoracic", "Hemidiaphragm", "Breathing Difficulty", "Radiodensity", "Asymmetry"],
["Lung", "Tissue", "Opacity", "Gray", "Pulmonary", "Infiltrate", "Solidification", "Alveoli", "Airspace", "Density"]]

filtered_chest_prior_disc = [['Fluid Accumulation', 'Thoracic Cavity', 'Opacity', 'Unilateral', 'Localized', 'Effusion'],
['Lung', 'Round', 'Solitary', 'Central', 'Lobe', 'Multiple', 'Peripheral', 'Density'],
['Alveoli', 'Infection', 'Inflammation', 'Bacterial', 'Airspace'],
['Left Atrium', 'Right Atrium', 'Heart', 'Cardiac', 'Aortic Arch', 'Hypertrophic', 'Enlarged', 'Cardiomegaly', 'Dilated'],
['Shadow', 'Bilateral', 'Enlargement', 'Right Hilum', 'Hilar', 'Opacity'],
['Old'],
['Lung', 'Pulmonary', 'Fibrosis', 'Tissue', 'Collagen', 'Fibrous', 'Opacity'],
['Calcified', 'Calcification'],
['Opacity'],
['Fibrous'],
['Gray', 'Tuberculosis', 'Lung', 'Consolidation', 'Infection', 'Bacterial', 'Cavitary'],
['Breathlessness'],
['Breathing Difficulty', 'Hyperinflation', 'Pulmonary'],
['Gray'],
['Pathology'],
['Thoracic Cavity', 'Effusion', 'Bilateral', 'Chest', 'Unilateral', 'Fluid Accumulation', 'Opacity', 'Gray', 'Localized', 'Swelling'],
['Bronchovascular'],
['Breathing Difficulty'],
['Gray', 'Airspace', 'Alveoli', 'Solidification', 'Density', 'Opacity', 'Tissue']]

class CustomDataset(Dataset):
    def __init__(self, data_root, file_list,task,list_txt, preprocess):
        self.data_root = data_root
        self.task = task
        self.file_list = file_list
        self.title = list_txt
        self.preprocess = preprocess

        self.get_label_list()
    def get_label_list(self):
        file_list = open(self.file_list, 'r').readlines()

        label_list = []
        image_list = []
        for file in file_list:

            if self.task == 'chest':
                img_name, label = file.split(" ")
                label = label.split(',')
                label = [int(item) for item in label]
            elif self.task == 'endo':
                img_name, label = file.split(".png")
                img_name = img_name + ".png"
                label = label.split(' ')[1:]
                label = [int(item) for item in label]
            else:
                img_name, label = file.split(" ")
                label = label.split(',')
                label = [int(item) for item in label]

            label_list.append(label)

            full_img_name = os.path.join(self.data_root, img_name)
            image_list.append(full_img_name)


        self.label_list = label_list
        self.image_list = image_list


    def __len__(self):
        file_list = open(self.file_list, 'r').readlines()
        return len(file_list)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_list[idx]))  # Image from PIL module
        title = clip.tokenize(self.title)
        label = torch.tensor(self.label_list[idx])

        return image, title, label

def get_probs(model, img_name, input_txts):
    image = preprocess(Image.open(img_name)).unsqueeze(0).to(device)
    text = clip.tokenize(input_txts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs = logits_per_image.cpu().numpy()

    return probs

def zeroshot_process(data_loader, model, label_list):

    mean_corr = []
    #predictions = []
    #labels = []

    predictions = np.empty((0,len(label_list)),dtype=np.float32)
    labels = np.empty((0, len(label_list)), dtype=np.float32)

    for batch in tqdm(data_loader, total=len(data_loader)):
        images, texts, label = batch

        images = images.to(device)
        texts = texts[0].to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(images, texts)
            probs = logits_per_image.cpu().numpy()
            predictions = np.concatenate((predictions,probs),axis=0)
            labels = np.concatenate((labels, label), axis=0)

    # 각 클래스별로 AP 계산
    aps = []
    for class_idx in range(predictions.shape[1]):
        ap = average_precision_score(labels[:, class_idx], predictions[:, class_idx])
        aps.append(ap)

    # mAP 계산
    map_score = np.mean(aps)
    print(f'mAP: {map_score:.4f}')
    return map_score

def descriptor_accessment(file_list_name,model,label_list,cls,disc):
    file_list = open(file_list_name,'r').readlines()

    task = file_list_name.split("/")[3]

    pos_cnt = 0
    neg_cnt = 0

    pos_sum = 0
    neg_sum = 0

    for file in file_list:
        if task == 'chest':
            img_name, label = file.split(" ")
            label = label.split(',')
            label = [int(item) for item in label]
        elif task == 'endo':
            img_name, label = file.split(".png")
            img_name = img_name+".png"
            label = label.split(' ')[1:]
            label = [int(item) for item in label]
        else:
            img_name, label = file.split(" ")
            label = label.split(',')
            label = [int(item) for item in label]

        full_img_name = os.path.join(file_prefix,img_name)

        if label[cls]:
            c = 1
            pos_cnt+=1
        else:
            c = -1
            neg_cnt+=1

        lvm_output = get_probs(model, full_img_name, disc)
        score = c * lvm_output[0]

        if score>0:
            pos_sum += score
        else:
            neg_sum += score

    sum_score = pos_sum/pos_cnt + neg_sum/ neg_cnt

    return sum(sum_score)


def get_filtered_disc(data_list, model, label_list, label_class, discs):
    disc_scores = {}  # 각 디스크의 점수를 저장할 딕셔너리

    # 디스크들의 점수를 계산하고 딕셔너리에 저장
    for disc in discs:
        score = descriptor_accessment(data_list, model, label_list, label_class, disc)
        disc_scores[disc] = score

    filtered_disc = []

    # 딕셔너리에서 양수인 점수를 가진 디스크들을 추출하여 filtered_disc에 추가
    for disc, score in disc_scores.items():
        if score > 0:
            filtered_disc.append((disc, score))

    # 양수인 디스크가 없으면 가장 높은 점수를 가진 디스크 하나를 반환
    if not filtered_disc:
        best_disc = max(disc_scores, key=disc_scores.get)
        return [best_disc]

    # 양수인 디스크들을 점수에 따라 내림차순으로 정렬
    filtered_disc.sort(key=lambda x: x[1], reverse=True)

    # 정렬된 디스크들의 리스트 반환
    return [disc[0] for disc in filtered_disc]


def run_get_filtered_disc():
    for label_class in range(0,19):
        filtered_discs = get_filtered_disc(data_list,model,label_list,label_class,chest_prior_disc[label_class])
        print(filtered_discs)

def contrastive_loss(origin_anchor_emb, positive_emb, negative_emb, margin):
    # 유클리드 거리를 계산합니다.

    anchor_emb = origin_anchor_emb.expand(positive_emb.shape[0], -1)
    distance_pos = torch.norm(anchor_emb - positive_emb, p=2, dim=1)

    anchor_emb = origin_anchor_emb.expand(negative_emb.shape[0], -1)
    distance_neg = torch.norm(anchor_emb - negative_emb, p=2, dim=1)

    # Contrastive loss를 계산합니다.
    loss = torch.mean((torch.mean(distance_pos) -torch.mean(distance_neg) + margin).clamp(min=0))

    return loss


def text_emb_learning(model,optimizer, cls_texts, prior_disc,margin,epochs=20,save=False):
    model.eval()
    # Freeze all model parameters
    for name, param in model.named_parameters():
        # print (name)
        if name.find("token_embedding") != -1 or name.find("text_projection") != -1:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_priors = [item for sublist in prior_disc for item in sublist]

    #for name, param in model.named_parameters():
    #    print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")


    for epoch_idx in range(epochs):
        for cls_idx, cls_text in enumerate(cls_texts):
            tokenized_cls = clip.tokenize(cls_text).to(device)
            cls_emb = model.encode_text(tokenized_cls)

            # pos_emb과 neg_emb에 대한 그래디언트 계산을 비활성화
            with torch.no_grad():
                pos_set = set(prior_disc[cls_idx])
                neg_set = set(total_priors) - set(pos_set)
                #print(pos_set, neg_set)

                pos_list = list(pos_set)
                neg_list = list(neg_set)

                pos_txt = clip.tokenize(pos_list).to(device)
                neg_txt = clip.tokenize(neg_list).to(device)

                pos_emb = model.encode_text(pos_txt)
                neg_emb = model.encode_text(neg_txt)

            optimizer.zero_grad()
            # cls_emb에 대한 그래디언트 계산을 활성화
            cls_emb.requires_grad_(True)

            #print(cls_emb.shape, pos_emb.shape, neg_emb.shape)
            loss = contrastive_loss(cls_emb, pos_emb, neg_emb, margin)
            #print('epoch:',epoch_idx, 'loss:',loss)

            # cls_emb에 대한 그래디언트를 사용하여 역전파
            loss.backward()
            optimizer.step()

            # pos_emb과 neg_emb에 대한 그래디언트를 다시 비활성화
            cls_emb = cls_emb.detach()

        test_loss_list = []
        for cls_idx, cls_text in enumerate(cls_texts):
            with torch.no_grad():
                tokenized_cls = clip.tokenize(cls_text).to(device)
                cls_emb = model.encode_text(tokenized_cls)
                pos_set = set(prior_disc[cls_idx])
                neg_set = set(total_priors) - set(pos_set)

                pos_list = list(pos_set)
                neg_list = list(neg_set)

                pos_txt = clip.tokenize(pos_list).to(device)
                neg_txt = clip.tokenize(neg_list).to(device)

                pos_emb = model.encode_text(pos_txt)
                neg_emb = model.encode_text(neg_txt)

                test_loss = contrastive_loss(cls_emb, pos_emb, neg_emb, margin)
                test_loss_list.append(test_loss)

        print (epoch_idx, torch.mean(torch.stack(test_loss_list)))
    if save:
        torch.save(model.state_dict(), "model_weights.pth")



def finetune_model(train_dataloader,optimizer, model,classifier , update_layer, epoch ):

    model.eval()
    # Freeze all model parameters
    for name, param in model.named_parameters():
        param.requires_grad = False
        for name, param in model.named_parameters():

            param.requires_grad = False

            for layer_name in update_layer:
                if name.find(layer_name) != -1:
                    param.requires_grad = True


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

    loss_img = nn.BCEWithLogitsLoss()


    for epoch_iter in range(epoch):
        for batch in train_dataloader:
            images, texts, label = batch
            #print(images.shape, texts.shape)

            images = images.to(device)
            texts = texts[0].to(device)

            #mage_features = model.encode_image(images)
            #text_features = model.encode_text(texts)

            logits_per_image, logits_per_text = model(images, texts)
            #logits_per_text = logits_per_text.T
            with torch.no_grad():
                ground_truth = torch.tensor(label, dtype=torch.float32).to(device).detach()

            if classifier:
                logits_per_image = torch.sigmoid(classifier(logits_per_image.float()))

            #logits_per_text = torch.sigmoid(logits_per_text)

            #print (logits_per_image,logits_per_text )
            optimizer.zero_grad()

            total_loss = loss_img(logits_per_image, ground_truth)

            #print (logits_per_image)
            #print (ground_truth)
            total_loss.backward()
            optimizer.step()
            print (epoch_iter, total_loss)




def main():
    parser = argparse.ArgumentParser(description="test_clip")
    parser.add_argument("-lr", type=float, help="learning_rate",default=1e-5)
    parser.add_argument("-update_mode", type=int, help="update mode",default=0)
    parser.add_argument("-text_emb", type=int, help="text embedding mode", default=0)
    parser.add_argument("-bs", type=int, help="batch size", default=32)
    parser.add_argument("-epochs", type=int, help="epochs", default=30)
    parser.add_argument("-task", type=str, help="task", default="chest")
    parser.add_argument("-shot", type=int, help="shot", default=5)
    parser.add_argument("-exp", type=int, help="exp", default=1)
    parser.add_argument("-usage_clssifier", type=int, help="usage_clssifier", default=0)
    args = parser.parse_args()

    wandb.init(project="clip-medfm", config=args)

    task = args.task
    shot = args.shot
    exp = args.exp

    LOAD = False

    train_data_list = f'/data/MedFMC/{task}/{task}_{shot}-shot_train_exp{exp}.txt'
    val_data_list = f'/data/MedFMC/{task}/{task}_{shot}-shot_val_exp{exp}.txt'
    test_data_list = f'/data/MedFMC/{task}/test_WithoutLabel2.txt'
    file_prefix = f'/data/MedFMC/{task}/images'
    label_list = label_dict[task]

    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)

    if LOAD:
        model.load_state_dict(torch.load("model_weights.pth"))

    if args.usage_clssifier:
        classifier = nn.Linear(len(label_list), len(label_list)).to(device)
        params_optimize = list(model.parameters()) + list(classifier.parameters())
    else:
        classifier = None
        params_optimize = model.parameters()

    optimizer = torch.optim.Adam(params_optimize, lr=args.lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    train_dataset = CustomDataset(file_prefix, train_data_list,task ,label_list, preprocess)
    train_dataloader = DataLoader(train_dataset,batch_size = args.bs)
    val_dataset = CustomDataset(file_prefix, val_data_list,task ,label_list , preprocess)
    val_dataloader = DataLoader(val_dataset,batch_size = args.bs)
    test_dataset = CustomDataset(file_prefix, test_data_list,task ,label_list , preprocess)
    test_dataloader = DataLoader(test_dataset,batch_size = args.bs)

    if args.update_mode == 0:
        update_list = ['logit_scale','visual.proj','visual.conv1']
    elif args.update_mode == 1:
        update_list = ['logit_scale','visual.proj','visual.conv1', 'visual.ln_post']
    elif args.update_mode == 2:
        update_list = ['logit_scale', 'visual.proj', 'visual.conv1', 'visual.ln_post', 'visual.ln_pre']
    elif args.update_mode == 3:
        update_list = ['logit_scale', 'visual']
    elif args.update_mode == 4:
        update_list = ['logit_scale', 'visual.proj', 'visual.conv1', 'visual.ln_post', 'visual.ln_pre',
                       "token_embedding","text_projection"]
    elif args.update_mode == 5:
        update_list = ['logit_scale', 'visual',"token_embedding","text_projection"]


    for stage in range(args.epochs):
        if args.text_emb:
            text_emb_learning(model,optimizer,label_list,chest_prior_disc,margin=1.0,epochs=1)
        finetune_model(train_dataloader,optimizer, model, classifier,update_list, epoch=1)
        score = zeroshot_process(val_dataloader,model,label_list)
        wandb.log({"val_mAP": score})


if __name__ == "__main__":
    main()


