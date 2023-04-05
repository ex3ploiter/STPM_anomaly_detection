import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
import shutil
import time
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
import pandas as pd
import dill 


#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

def data_transforms(load_size=256, mean_train=mean_train, std_train=std_train):
    data_transforms = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])
    return data_transforms

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def cal_loss(fs_list, ft_list, criterion):
    tot_loss = 0
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        _, _, h, w = fs.shape
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        # a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        # f_loss = (1/(w*h))*torch.sum(a_map)
        f_loss = (0.5/(w*h))*criterion(fs_norm, ft_norm)
        tot_loss += f_loss

    return tot_loss

def cal_anomaly_map(fs_list, ft_list, out_size=224):
    anomaly_map = torch.ones(out_size, out_size).cuda()
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        fs_norm = F.normalize(fs, p=2)
        ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs_norm, ft_norm)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear')
        # a_map = a_map[0,0,:,:].to('cpu').detach().numpy()
        a_map = a_map[0,0,:,:]
        a_map_list.append(a_map)
        anomaly_map *= a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    

class STPM():
    def __init__(self,attack_eps,steps,dataset_name,normal_class,load_robust):
        self.load_robust=load_robust
        self.load_model()
        self.data_transform = data_transforms(load_size=load_size, mean_train=mean_train, std_train=std_train)
        self.eps=attack_eps
        self.steps=steps
        self.dataset_name=dataset_name
        self.normal_class=normal_class
       
       


    def load_robust_model(self,model):
        mode=1
        resume_path='/content/STPM_anomaly_detection/legacy/resnet18_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI='
        # resume_path='/content/ExposureExperiment/resnet18_linf_eps8.0.ckpt?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI='
        
        checkpoint = torch.load(resume_path, pickle_module=dill)
        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'
        
        sd = checkpoint[state_dict_path]
        sd = {k[len('module.'):]:v for k,v in sd.items()}

        if mode ==0: # Model
            sd_t = {k[len('model.'):]:v for k,v in sd.items() if k.split('.')[0]=='model'} 
        
        elif mode ==1: # Attacker
            sd_t = {k[len('attacker.model.'):]:v for k,v in sd.items() if k.split('.')[0]=='attacker' and k.split('.')[1]!='normalize'}
        

            model.load_state_dict(sd_t)        
            
        return model 
    
    def load_model(self):
        self.features_t = []
        self.features_s = []
        def hook_t(module, input, output):
            self.features_t.append(output)
        def hook_s(module, input, output):
            self.features_s.append(output)
        
        self.model_t = resnet18(pretrained=True).to(device)
        
        

        if self.load_robust=='True' :
            
            self.model_t= self.load_robust_model(self.model_t)


        self.model_t.layer1[-1].register_forward_hook(hook_t)
        self.model_t.layer2[-1].register_forward_hook(hook_t)
        self.model_t.layer3[-1].register_forward_hook(hook_t)

        self.model_s = resnet18(pretrained=False).to(device)
        
        if self.load_robust=='True' :
            self.model_s= self.load_robust_model(self.model_s)


        self.model_s.layer1[-1].register_forward_hook(hook_s)
        self.model_s.layer2[-1].register_forward_hook(hook_s)
        self.model_s.layer3[-1].register_forward_hook(hook_s)
        
    def train(self,trainloader):

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model_s.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

        
        
        start_time = time.time()
        global_step = 0

        for epoch in range(num_epochs):
            print('-'*20)
            print('Time consumed : {}s'.format(time.time()-start_time))
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-'*20)
            self.model_t.eval()
            self.model_s.train()
            # for idx, (batch, _) in enumerate(trainloader): # batch loop
            
            with tqdm(trainloader, unit="batch") as tepoch:
                for idx, (batch, _) in enumerate(tepoch):
                    global_step += 1
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        self.features_t = []
                        self.features_s = []
                        _ = self.model_t(batch)
                        _ = self.model_s(batch)
                        # get loss using features.
                        loss = cal_loss(self.features_s, self.features_t, self.criterion)
                        loss.backward()
                        self.optimizer.step()

                    tepoch.set_postfix(loss=float(loss.data))

                    # if idx%2 == 0:
                    #     print('Epoch : {} | Loss : {:.4f}'.format(epoch, float(loss.data)))

        print('Total time consumed : {}'.format(time.time() - start_time))
        print('Train end.')
        # if save_weight:
        #     print('Save weights.')
        #     torch.save(self.model_s.state_dict(), os.path.join(weight_save_path, 'model_s.pth'))

    
    def AttackImage(self,images,labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()



        loss = torch.nn.BCEWithLogitsLoss()

        adv_images = images.clone().detach()

        alpha = (2.5 * self.eps) / self.steps



        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.get_logits(adv_images)
            outputs=self.getScore(adv_images)


            cost = loss(outputs, torch.tensor(labels.float().item()).cuda())

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
    
    def getScore(self,test_img):
        self.features_t = []
        self.features_s = []
        _ = self.model_t(test_img)
        _ = self.model_s(test_img)
    
        anomaly_map, _ = cal_anomaly_map(self.features_s, self.features_t, out_size=input_size)
        
        return torch.max(anomaly_map)
        
        # pred_list_img_lvl.append(np.max(anomaly_map))

    def test(self,test_loader=None):
        print('Test phase start')
        # try:
        #     self.model_s.load_state_dict(torch.load(glob.glob(weight_save_path+'/*.pth')[0]))
        # except:
        #     raise Exception('Check saved model path.')

        self.model_t.eval()
        self.model_s.eval()
        
        
        labels=[]
        clean_scores=[]
        adv_scores=[]

        start_time = time.time()
         
        for idx,(test_img,lbl) in enumerate(tqdm(test_loader)):

            test_img = test_img.to(device)
            lbl = lbl.to(device)

            
        
            clean_score=self.getScore(test_img)
            
            adv_image=self.AttackImage(test_img,lbl)
            adv_score=self.getScore(adv_image)

            # print(f'Label : {lbl.detach().data}')
            # print(f'clean_score : {clean_score.detach().data}')
            # print(f'adv_score : {adv_score.detach().data}')
            # print("\n")

            



            clean_scores.append(clean_score.detach().item())
            adv_scores.append(adv_score.detach().item())
            labels.append(lbl.detach().item()) 

            # if lbl==0:
            #     break

            # if idx==30:
                # break

  
        print('Total test time consumed : {}'.format(time.time() - start_time))
        # print("Total image-level auc-roc score :")
        # print(roc_auc_score(gt_list_img_lvl, pred_list_img_lvl))
        print("roc_auc_score(labels, clean_scores) : ",roc_auc_score(labels, clean_scores))
        print("roc_auc_score(labels, adv_scores) : ", roc_auc_score(labels, adv_scores))


        df = pd.DataFrame({'Clean_AUC': [roc_auc_score(labels, clean_scores)], 'Adv_AUC': [roc_auc_score(labels, adv_scores)]})
        if not os.path.isdir('./result_dir/'):
            os.mkdir('./result_dir/')
        
        df.to_csv(f'./result_dir/results_{self.dataset_name}_NormalClass_{self.normal_class}_eps_{self.eps:.3f}.csv', index=False)


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--dataset_path', default=r'/content/mvtec_anomaly_detection/tile') #D:\Dataset\mvtec_anomaly_detection\transistor')
    parser.add_argument('--num_epoch', default=100)
    parser.add_argument('--lr', default=0.4)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--load_size', default=256)
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--project_path', default=r'/content/project_results/STPM_results') #D:\Project_Train_Results\mvtec_anomaly_detection\transistor_new_temp')
    parser.add_argument('--save_src_code', default=False)
    parser.add_argument('--save_anomaly_map', default=True)
    
    parser.add_argument('--dataset', default='cifar10',type=str)
    parser.add_argument('--normal_class', default=0,type=int)
    parser.add_argument('--attack_eps', default=8/255,type=float)
    parser.add_argument('--steps', default=10,type=int)
    parser.add_argument('--load_robust', default='True',type=str)
    args = parser.parse_args()
    return args




def get_CIFAR10(normal_class_indx:int, transform):
    trainset = CIFAR10(root='./data', train=True, download=True,transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset, testset

def get_CIFAR100(normal_class_indx:int, transform):
    trainset = CIFAR100(root=CIFAR10_PATH, train=True, download=False,transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR100(root=CIFAR10_PATH, train=False, download=False, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset, testset


def get_MNIST(normal_class_indx:int, transform):
    trainset = MNIST(root=MNIST_PATH, train=True, download=False,transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = MNIST(root=MNIST_PATH, train=False, download=False, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset, testset

def get_FASHION_MNIST(normal_class_indx:int, transform):
    trainset = FashionMNIST(root=FMNIST_PATH, train=True, download=False,transform=transform)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = FashionMNIST(root=FMNIST_PATH, train=False, download=False, transform=transform)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset, testset



def getDatasetLoader(dataset,normal_class_indx):

    transform_1d = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.ANTIALIAS),
             transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])


    transform_3d = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(input_size),
            transforms.Normalize(mean=mean_train,
                                std=std_train)])

    if dataset == 'cifar10':
        trainset, testset=get_CIFAR10(normal_class_indx, transform_3d)
    elif dataset == 'cifar100':
        trainset, testset=get_CIFAR100(normal_class_indx, transform_3d)
    elif dataset == 'mnist':
        trainset, testset=get_MNIST(normal_class_indx, transform_1d)
    elif dataset == 'fashion':
        trainset, testset=get_FASHION_MNIST(normal_class_indx, transform_1d)
    # elif dataset == 'svhn':
    #     trainset, testset=get_SVHN(normal_class_indx, transform)
    # elif dataset == 'mvtec':
        # trainset, testset=get_MVTEC(normal_class_indx, transform)
    
    trainset_loader=DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0) #, pin_memory=True)
    testset_loader=DataLoader(testset, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True)

    return trainset_loader,testset_loader


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    # print(torch.cuda.get_device_name(device))
    
    args = get_args()
    phase = args.phase
    dataset_path = args.dataset_path
    category = dataset_path.split('\\')[-1]
    num_epochs = int(args.num_epoch)
    lr = args.lr
    batch_size = int(args.batch_size)
    save_weight = True
    load_size = args.load_size
    input_size = args.input_size
    save_src_code = args.save_src_code
    project_path = args.project_path
    sample_path = os.path.join(project_path, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    weight_save_path = os.path.join(project_path, 'saved')
    if save_weight:
        os.makedirs(weight_save_path, exist_ok=True)
    if save_src_code:
        source_code_save_path = os.path.join(project_path, 'src')
        os.makedirs(source_code_save_path, exist_ok=True)
        copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README']) # copy source code
    

    stpm = STPM(args.attack_eps,args.steps,args.dataset,args.normal_class,args.load_robust)

    trainset_loader,testset_loader=getDatasetLoader(args.dataset,args.normal_class)

    if phase == 'train':
        stpm.train(trainset_loader)
        stpm.test(testset_loader)
    elif phase == 'test':
        stpm.test(testset_loader)
    else:
        print('Phase argument must be train or test.')
