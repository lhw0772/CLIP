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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip

import os
from coop.coop import CustomCLIP
from coop.dualcoop import build_model
from utils.asymmetric_loss import AsymmetricLoss, AsymmetricLoss2, AsymmetricLoss3
import coop.coop_vpt as coop_vpt


#os.environ["WANDB_MODE"]="offline"
WANDB_TITLE = "clip-medfm-1011-domain"

# Set a fixed seed for CPU operations
seed = 100
torch.manual_seed(seed)

# Set a fixed seed for GPU operations (if available)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Additional steps for improved reproducibility (may slow down performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



device = "cuda" if torch.cuda.is_available() else "cpu"

chest_gpt35_combined_label = [
    "pleural_effusion is an irregular-shaped bump, often appearing as white, near the pleural space.",
    "nodule is a round or oval-shaped bump, often in white or light gray, within the lung tissue.",
    "pneumonia is an irregular-shaped area with increased opacity, often in white or gray, within the lung.",
    "cardiomegaly is an enlargement of the heart, typically appearing as an enlarged white shadow in the cardiac area.",
    "hilar_enlargement is an enlargement of the hilar region of the lungs, often presenting as increased white density in that area.",
    "fracture_old is a discontinuity or irregularity in the bone structure, possibly showing as white lines or gaps in a bone.",
    "fibrosis is a diffuse, irregular pattern of increased opacity, often appearing as white streaks or patches throughout the lung.",
    "aortic_calcification is the calcification or hardening of the aorta, typically seen as white deposits along the aortic wall.",
    "tortuous_aorta is a twisted or curved appearance of the aorta, often seen as an irregular white line.",
    "thickened_pleura is a thickening of the pleural lining, usually appearing as a white or opaque layer along the lung's outer surface.",
    "TB is a pattern of nodules or irregularities, often appearing as white or gray areas within the lung tissue.",
    "pneumothorax is an air-filled space within the pleural cavity, often seen as a dark area or absence of lung markings.",
    "emphysema is a pattern of increased lung volume with reduced density, often appearing as darker areas in the lung.",
    "atelectasis is a collapse or partial collapse of the lung, often appearing as a white or opaque area within the lung tissue.",
    "calcification is the presence of calcium deposits, typically seen as white spots or areas within a tissue.",
    "pulmonary_edema is an accumulation of fluid in the lungs, often appearing as increased white density throughout the lung fields.",
    "increased_lung_markings is an increased prominence of lung markings, often appearing as darker lines or streaks on the X-ray.",
    "elevated_diaphragm is an elevation or abnormal position of the diaphragm, often seen as an irregular white line near the lower lung border.",
    "consolidation is a solidification of lung tissue, often appearing as a white, opaque area within the lung."
]

chest_descriptions = [
    "pleural_effusion is a fluid accumulation, often seen as a darker area on the X-ray, located between the lung and chest wall in the X-ray image domain.",
    "nodule is a small lump, often seen as a small rounded shadow on the X-ray, located in the lung tissue in the X-ray image domain.",
    "pneumonia is an infection, often seen as a cloudy area on the X-ray, located in the lung tissue in the X-ray image domain.",
    "cardiomegaly is an enlarged heart, often seen as an increased heart size on the X-ray, located in the cardiac region in the X-ray image domain.",
    "hilar_enlargement is an enlargement of the lung hilum, often seen as a prominent shadow on the X-ray, located in the central part of the lungs in the X-ray image domain.",
    "fracture_old is a healed bone break, often seen as a line or irregularity on the X-ray, located in the bones in the X-ray image domain.",
    "fibrosis is a scarring, often seen as streaky shadows on the X-ray, located in the lung tissue in the X-ray image domain.",
    "aortic_calcification is a calcium buildup, often seen as a bright line on the X-ray, located in the aorta in the X-ray image domain.",
    "tortuous_aorta is a twisted aorta, often seen as a curvy shadow on the X-ray, located in the aortic region in the X-ray image domain.",
    "thickened_pleura is a thickened pleural lining, often seen as a dense line on the X-ray, located around the lungs in the X-ray image domain.",
    "TB is a tuberculosis infection, often seen as patchy shadows on the X-ray, located in the lung tissue in the X-ray image domain.",
    "pneumothorax is a collapsed lung, often seen as a clear space without lung markings on the X-ray, located next to the lung edge in the X-ray image domain.",
    "emphysema is a lung condition, often seen as areas with decreased density on the X-ray, located in the lung tissue in the X-ray image domain.",
    "atelectasis is a collapsed or closed off lung, often seen as an area of increased density on the X-ray, located in the affected lung region in the X-ray image domain.",
    "calcification is a calcium deposit, often seen as bright spots on the X-ray, located in various tissues in the X-ray image domain.",
    "pulmonary_edema is fluid accumulation, often seen as fluffy shadows on the X-ray, located in the lung tissue in the X-ray image domain.",
    "increased_lung_markings is an increased visibility of lung vessels, often seen as prominent lines on the X-ray, located in the lung tissue in the X-ray image domain.",
    "elevated_diaphragm is a raised diaphragm, often seen as a higher position of the diaphragm shadow on the X-ray, located at the base of the lungs in the X-ray image domain.",
    "consolidation is a lung tissue thickening, often seen as a solid white area on the X-ray, located in the affected lung region in the X-ray image domain."
]

chest_descriptions_wo_domain = [
    "pleural_effusion is a fluid accumulation, often seen as a darker area, located between the lung and chest wall.",
    "nodule is a small lump, often seen as a small rounded shadow, located in the lung tissue.",
    "pneumonia is an infection, often seen as a cloudy area, located in the lung tissue.",
    "cardiomegaly is an enlarged heart, often seen as an increased heart size, located in the cardiac region.",
    "hilar_enlargement is an enlargement of the lung hilum, often seen as a prominent shadow, located in the central part of the lungs.",
    "fracture_old is a healed bone break, often seen as a line or irregularity, located in the bones.",
    "fibrosis is a scarring, often seen as streaky shadows, located in the lung tissue.",
    "aortic_calcification is a calcium buildup, often seen as a bright line, located in the aorta.",
    "tortuous_aorta is a twisted aorta, often seen as a curvy shadow, located in the aortic region.",
    "thickened_pleura is a thickened pleural lining, often seen as a dense line, located around the lungs.",
    "TB is a tuberculosis infection, often seen as patchy shadows, located in the lung tissue.",
    "pneumothorax is a collapsed lung, often seen as a clear space without lung markings, located next to the lung edge.",
    "emphysema is a lung condition, often seen as areas with decreased density, located in the lung tissue.",
    "atelectasis is a collapsed or closed off lung, often seen as an area of increased density, located in the affected lung region.",
    "calcification is a calcium deposit, often seen as bright spots, located in various tissues.",
    "pulmonary_edema is fluid accumulation, often seen as fluffy shadows, located in the lung tissue.",
    "increased_lung_markings is an increased visibility of lung vessels, often seen as prominent lines, located in the lung tissue.",
    "elevated_diaphragm is a raised diaphragm, often seen as a higher position of the diaphragm shadow, located at the base of the lungs.",
    "consolidation is a lung tissue thickening, often seen as a solid white area, located in the affected lung region."
]


endo_gpt35_combined_label =[
    "ulcer is a sore or lesion on the inner lining of the digestive tract, often appearing as an open, crater-like area.",
    "erosion is the gradual wearing away of the lining of the digestive tract, often presenting as a shallow, superficial loss of tissue.",
    "polyp is a small growth or mass protruding from the inner lining of the digestive tract, often appearing as a round or elongated structure.",
    "tumor is an abnormal mass or lump within the digestive tract, often showing as an irregular, solid growth."
]


endoscopy_descriptions = [
    "ulcer is a sore, often appearing as a deep, crater-like opening, located in the mucosal layer of the endoscopy image.",
    "erosion is a superficial damage, often appearing as a shallow, reddened area, located on the surface of the mucosa in the endoscopy image.",
    "polyp is a growth, often appearing as a raised and rounded protrusion, located on the inner lining of the colon or rectum in the endoscopy image.",
    "tumor is a mass or lump, often appearing as a larger, irregular growth, located in the affected tissue or organ in the endoscopy image."
]

endoscopy_descriptions_wo_domain = [
    "ulcer is a sore, often appearing as a deep, crater-like opening, located in the mucosal layer.",
    "erosion is a superficial damage, often appearing as a shallow, reddened area, located on the surface of the mucosa.",
    "polyp is a growth, often appearing as a raised and rounded protrusion, located on the inner lining of the colon or rectum.",
    "tumor is a mass or lump, often appearing as a larger, irregular growth, located in the affected tissue or organ."
]

chest_domain = [
    "pleural_effusion in the X-ray image domain.",
    "nodule in the X-ray image domain.",
    "pneumonia in the X-ray image domain.",
    "cardiomegaly in the X-ray image domain.",
    "hilar_enlargement in the X-ray image domain.",
    "fracture_old in the X-ray image domain.",
    "fibrosis in the X-ray image domain.",
    "aortic_calcification in the X-ray image domain.",
    "tortuous_aorta in the X-ray image domain.",
    "thickened_pleura in the X-ray image domain.",
    "TB in the X-ray image domain.",
    "pneumothorax in the X-ray image domain.",
    "emphysema in the X-ray image domain.",
    "atelectasis in the X-ray image domain.",
    "calcification in the X-ray image domain.",
    "pulmonary_edema in the X-ray image domain.",
    "increased_lung_markings in the X-ray image domain.",
    "elevated_diaphragm in the X-ray image domain.",
    "consolidation in the X-ray image domain."
]

endoscopy_domain = [
    "ulcer in the endoscopy image.",
    "erosion in the endoscopy image.",
    "polyp in the endoscopy image.",
    "tumor in the endoscopy image."
]



gpt35_label_dict = {"chest": chest_gpt35_combined_label, "endo": endo_gpt35_combined_label}

gpt4_domain_dict = {"chest": chest_domain, "endo": endoscopy_domain}

gpt4_pattern_shape_location_domain_dict = {"chest": chest_descriptions, "endo": endoscopy_descriptions}

gpt4_pattern_shape_location_dict = {"chest": chest_descriptions_wo_domain, "endo": endoscopy_descriptions_wo_domain}

label_dict = {"chest": ['pleural_effusion','nodule','pneumonia','cardiomegaly','hilar_enlargement',
              'fracture_old','fibrosis','aortic_calcification','tortuous_aorta','thickened_pleura' ,'TB','pneumothorax',
              'emphysema','atelectasis','calcification','pulmonary_edema','increased_lung_markings',
              'elevated_diaphragm','consolidation'], "endo":['ulcer','erosion','polyp','tumor']}

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

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def train_transform(n_px):
    return Compose([
        RandomResizedCrop(n_px, scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=3),  # RandomResizedCrop
        RandomHorizontalFlip(),  # RandomFlip
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

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

def zeroshot_process(args,data_loader, model, label_list):

    mean_corr = []
    #predictions = []
    #labels = []

    predictions = np.empty((0,len(label_list)),dtype=np.float32)
    labels = np.empty((0, len(label_list)), dtype=np.float32)
    Softmax = torch.nn.Softmax(dim=1)

    for batch in tqdm(data_loader, total=len(data_loader)):
        images, texts, label = batch

        images = images.to(device)
        texts = texts[0].to(device)

        with torch.no_grad():
            if args.coop:
                logits_per_image = model(images)
            elif args.dualcoop:
                logits_per_image = model(images,None)
                logits_per_image = Softmax(logits_per_image.detach())[:, 1, :]
            elif args.vpt:
                logits_per_image, _, _ = model(images)
            else:
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


def text_emb_learning(args,model,optimizer, cls_texts, prior_disc,margin,epochs=20,save=False):
    model.eval()
    # Freeze all model parameters
    for name, param in model.named_parameters():
        # print (name)
        if name.find("token_embedding") != -1 or name.find("text_projection") != -1:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_priors = [item for sublist in prior_disc for item in sublist]

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

    if args.coop :
        text_encoder = model.text_encoder
    else:
        text_encoder = model.encode_text


    for epoch_idx in range(epochs):
        for cls_idx, cls_text in enumerate(cls_texts):
            tokenized_cls = clip.tokenize(cls_text).to(device)

            if args.coop:
                cls_emb = text_encoder(tokenized_cls,tokenized_cls)
            else:
                cls_emb = text_encoder(tokenized_cls)

            # pos_emb과 neg_emb에 대한 그래디언트 계산을 비활성화
            with torch.no_grad():
                pos_set = set(prior_disc[cls_idx])
                neg_set = set(total_priors) - set(pos_set)
                #print(pos_set, neg_set)

                pos_list = list(pos_set)
                neg_list = list(neg_set)

                pos_txt = clip.tokenize(pos_list).to(device)
                neg_txt = clip.tokenize(neg_list).to(device)

                if args.coop:
                    pos_emb = text_encoder(pos_txt, pos_txt)
                else:
                    pos_emb = text_encoder(pos_txt)

                if args.coop:
                    neg_emb = text_encoder(neg_txt, neg_txt)
                else:
                    neg_emb = text_encoder(neg_txt)
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
                if args.coop:
                    cls_emb = text_encoder(tokenized_cls, tokenized_cls)
                else:
                    cls_emb = text_encoder(tokenized_cls)

                pos_set = set(prior_disc[cls_idx])
                neg_set = set(total_priors) - set(pos_set)

                pos_list = list(pos_set)
                neg_list = list(neg_set)

                pos_txt = clip.tokenize(pos_list).to(device)
                neg_txt = clip.tokenize(neg_list).to(device)

                if args.coop:
                    pos_emb = text_encoder(pos_txt, pos_txt)
                else:
                    pos_emb = text_encoder(pos_txt)

                if args.coop:
                    neg_emb = text_encoder(neg_txt, neg_txt)
                else:
                    neg_emb = text_encoder(neg_txt)

                #pos_emb = text_encoder(pos_txt)
                #neg_emb = text_encoder(neg_txt)

                test_loss = contrastive_loss(cls_emb, pos_emb, neg_emb, margin)
                test_loss_list.append(test_loss)

        print (epoch_idx, torch.mean(torch.stack(test_loss_list)))
    if save:
        torch.save(model.state_dict(), "model_weights.pth")



def finetune_model(args,train_dataloader,optimizer, model, classifier , update_layer, epoch ):

    model.eval()
    # Freeze all model parameters

    if update_layer[0] =='x': # dualcoop
        for name, param in model.named_parameters():
            if name.find("text_encoder")!=-1:
                param.requires_grad = False

        model.prompt_learner.train()

    elif update_layer[0] =='o':
        model.prompt_learner.train()
        model.image_encoder.attnpool.train()
        model.image_encoder.train()

    else:
        for name, param in model.named_parameters():

            param.requires_grad = False

            for layer_name in update_layer:
                if name.find(layer_name) != -1:
                    param.requires_grad = True


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

    loss_img = nn.BCEWithLogitsLoss()
    coop_mlc_asl_gamma_neg = 2
    coop_mlc_asl_gamma_pos = 1

    criterion = AsymmetricLoss(coop_mlc_asl_gamma_neg, coop_mlc_asl_gamma_pos)
    criterion2 = AsymmetricLoss2(coop_mlc_asl_gamma_neg, coop_mlc_asl_gamma_pos)
    criterion3 = AsymmetricLoss3(coop_mlc_asl_gamma_neg, coop_mlc_asl_gamma_pos)


    for epoch_iter in range(epoch):
        for batch in train_dataloader:
            images, texts, label = batch
            #print(images.shape, texts.shape)

            images = images.to(device)
            texts = texts[0].to(device)

            if classifier:
                if classifier.in_features == 768:
                    image_features = model.encode_image(images)
            #text_features = model.encode_text(texts)
            if args.coop:
                logits_per_image = model(images)
            elif args.dualcoop:
                logits_per_image = model(images, None)
            elif args.vpt:
                logits_per_image, _,_ = model(images)
            else:
                logits_per_image, logits_per_text = model(images, texts)
            #logits_per_text = logits_per_text.T
            with torch.no_grad():
                ground_truth = torch.tensor(label, dtype=torch.float32).to(device).detach()

            if classifier:
                if classifier.in_features ==19:
                    logits_per_image = torch.sigmoid(classifier(logits_per_image.float()))
                elif classifier.in_features ==768:
                    logits_per_image = torch.sigmoid(classifier(image_features.float()))

            #logits_per_text = torch.sigmoid(logits_per_text)

            #print (logits_per_image,logits_per_text )
            optimizer.zero_grad()

            args_loss_w = 0.01

            if args.dualcoop:
                if logits_per_image.dim() == 3:
                    total_loss = args_loss_w * criterion(logits_per_image, ground_truth)
                elif args.single_prompt == 'pos':
                    total_loss = args_loss_w * criterion2(logits_per_image, ground_truth)
                elif args.single_prompt == 'neg':
                    total_loss = args_loss_w * criterion3(logits_per_image, ground_truth)
                else:
                    raise ValueError

            else:
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
    parser.add_argument("-usage_classifier", type=int, help="usage_classifier", default=0)
    parser.add_argument("-usage_prior_label", type=int, help="usage_prior_label", default=0)
    parser.add_argument("-usage_aug", type=int, help="usage_aug", default=0)
    parser.add_argument("-test_freq", type=int, help="test_freq", default=5)
    parser.add_argument("-test_mode", type=str, help="test_mode", default='val')
    parser.add_argument("-coop", type=int, help="coop", default=0)
    parser.add_argument("-csc", type=int, help="csc", default=0)
    parser.add_argument("-dualcoop", type=int, help="dualcoop", default=0)
    parser.add_argument("-n_ctx_pos", type=int, help="n_ctx_pos", default=16)
    parser.add_argument("-n_ctx_neg", type=int, help="n_ctx_neg", default=16)
    parser.add_argument("-clip_model_name", type=str, help="clip_model_name", default="ViT-L/14")
    parser.add_argument("-vpt", type=int, help="vpt", default=0)


    args = parser.parse_args()

    wandb.init(project=WANDB_TITLE, config=args)

    task = args.task
    shot = args.shot
    exp = args.exp

    LOAD = False

    train_data_list = f'/data/MedFMC/{task}/{task}_{shot}-shot_train_exp{exp}.txt'
    val_data_list = f'/data/MedFMC/{task}/{task}_{shot}-shot_val_exp{exp}.txt'
    test_data_list = f'/data/MedFMC/{task}/test_WithoutLabel2.txt'
    file_prefix = f'/data/MedFMC/{task}/images'

    if args.usage_aug == 0:
        model, preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    else:
        model, preprocess = clip.load(args.clip_model_name, device=device, jit=False)
        train_preprocess =  train_transform(model.visual.input_resolution)

    if LOAD:
        model.load_state_dict(torch.load("model_weights.pth"))

    if args.usage_prior_label==3:
        label_list = gpt4_domain_dict[task]
    elif args.usage_prior_label==2:
        label_list = gpt4_pattern_shape_location_domain_dict[task]
    elif args.usage_prior_label ==1:
        label_list = gpt4_pattern_shape_location_dict[task]
    elif args.usage_prior_label ==-1:
        label_list = gpt35_label_dict[task]
    else:
        label_list = label_dict[task]

    if args.coop:
        model = CustomCLIP(label_list, model.to('cpu'),args)
        model.to(device)
    elif args.dualcoop:
        model = build_model(args.clip_model_name,label_list,args)
    elif args.vpt:
        model = coop_vpt.CustomCLIP(label_list,model.to('cpu'))
        model.to(device)

    if args.usage_classifier == 1:
        classifier = nn.Linear(len(label_list), len(label_list)).to(device)
        params_optimize = list(model.parameters()) + list(classifier.parameters())
    elif args.usage_classifier == 2:
        classifier = nn.Linear(768, len(label_list)).to(device)
        params_optimize = list(model.parameters()) + list(classifier.parameters())
    else:
        classifier = None
        params_optimize = model.parameters() #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    if args.usage_aug == 0:
        train_dataset = CustomDataset(file_prefix, train_data_list,task ,label_list, preprocess)
    else:
        train_dataset = CustomDataset(file_prefix, train_data_list, task, label_list, train_preprocess)
    train_dataloader = DataLoader(train_dataset,batch_size = args.bs)
    val_dataset = CustomDataset(file_prefix, val_data_list,task ,label_list , preprocess)
    val_dataloader = DataLoader(val_dataset,batch_size = args.bs)
    test_dataset = CustomDataset(file_prefix, test_data_list,task ,label_list , preprocess)
    test_dataloader = DataLoader(test_dataset,batch_size = args.bs)

    if args.test_mode == 'val':
        selected_dataloader =  val_dataloader
    else:
        selected_dataloader = test_dataloader

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
    elif args.update_mode == 6:
        scaler_list = ['logit_scale']
        projector_list = ['text_projection','visual.proj']
        visual_list = ['visual.transformer','visual.ln_post','visual.ln_pre']
        text_list = ['transformer.resblocks','token_embedding','ln_final','text_projection']

        update_list = scaler_list+projector_list+visual_list+text_list

    if args.coop:
        update_list = ['prompt_learner.ctx']
        params_optimize = model.prompt_learner.parameters()
    elif args.dualcoop:
        prompt_params = model.prompt_params()
        prompt_group = {'params': prompt_params}
        print('num of params in prompt learner: ', len(prompt_params))

        if args.update_mode == 6:
            update_list = ['o']
            params_optimize = model.parameters()
        else:
            update_list = ['x']
            params_optimize = [prompt_group]
    elif args.vpt:
        update_list = ['prompt_learner.vis_ctx','prompt_learner.txt_ctx']

    optimizer_all = torch.optim.Adam(params_optimize, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer_text = torch.optim.Adam(model.parameters() , lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for stage in range(args.epochs):
        if args.text_emb:
            if args.coop:
                model.text_encoder.use_forward_text = True
            text_emb_learning(args,model,optimizer_text,label_list,chest_prior_disc,margin=1.0,epochs=1)
            if args.coop:
                model.text_encoder.use_forward_text = False

        finetune_model(args,train_dataloader,optimizer_all, model, classifier,update_list, epoch=1)
        if stage%args.test_freq ==0:
            score = zeroshot_process(args,selected_dataloader,model,label_list)
        wandb.log({"val_mAP": score})


if __name__ == "__main__":
    main()


