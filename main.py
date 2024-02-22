import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,ConcatDataset
import clip
import numpy as np
import copy
import json
from tqdm import tqdm
import datetime

from model import SpikeModel,StraightThrough,get_maximum_activation,replace_gelu_with_relu,get_zeroshot_classifier
from utils import seed_all,convert_model_precision
from datasets.cifar10 import CIFAR101,CIFAR102,CIFAR10C

data_root = 'data'
num_workers = 8
dtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class VisualClassifyModel(nn.Module):
    """
    Initialize to load a finetuned ViT with classifier
    """
    def __init__(self, featurizer, fea_len, class_num):
        super().__init__()
        self.Featurizer = featurizer
        self.Classifier = nn.Linear(fea_len,class_num)
    def forward(self, images):
        fea = self.Featurizer(images).type(torch.float32)
        pred = self.Classifier(fea)
        return pred


def get_dataloader(dataset_name,preprocess):
    """
    Get train_loader & test_loader, supporting CIFAR-10, CIFAR-100, ImageNet-1k, ImageNet-200
    """
    dataset_name = dataset_name.lower()
    names = dataset_name.split(":")
    kwargs = {'corruption':names[1]} if len(names) == 2 else {}
    train_loader = None
    test_loader = None
    if 'cifar10' == dataset_name:
        train_dataset = torchvision.datasets.CIFAR10(root=f"{data_root}/cifar10", train=True, transform=preprocess, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=f"{data_root}/cifar10", train=False, transform=preprocess, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=num_workers,pin_memory=True)
    if 'cifar100' == dataset_name:
        train_dataset = torchvision.datasets.CIFAR100(root=f"{data_root}/cifar100", train=True, transform=preprocess, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=f"{data_root}/cifar100", train=False, transform=preprocess, download=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=num_workers,pin_memory=True)
    if 'imagenet200' == dataset_name:
        train_dataset = torchvision.datasets.ImageFolder(root=f'{data_root}/tiny-imagenet-200/train',transform=preprocess)
        test_dataset = torchvision.datasets.ImageFolder(root=f'{data_root}/tiny-imagenet-200/val',transform=preprocess)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=num_workers,pin_memory=True)
    if 'imagenet' == dataset_name:
        train_dataset = torchvision.datasets.ImageFolder(root=f'{data_root}/imagenet/train',transform=preprocess)
        test_dataset = torchvision.datasets.ImageFolder(root=f'{data_root}/imagenet/val',transform=preprocess)
        idx_to_name = json.load(open(f"{data_root}/imagenet_class_index.json"))
        idx_to_name = [idx_to_name[str(i)][1] for i in range(1000)]
        test_dataset.classes = idx_to_name
        train_dataset.classes = idx_to_name
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True,num_workers=num_workers,pin_memory=True)
    # only for testing, including cifar10c, cifar10.1, cifar10.2
    if 'cifar' in dataset_name and dataset_name not in {'cifar10','cifar100'}:
        test_dataset = eval(names[0].upper())(preprocess=preprocess,location=f'{data_root}',shuffle=True,batch_size=args.batch_size,num_workers=num_workers,**kwargs)
        test_loader = test_dataset.test_loader
        test_loader.dataset.classes = test_dataset.classnames
    return train_loader,test_loader


def get_concat_dataloader(dataset_names,preprocess):
    """
    Get train_loader & test_loader of multiple datasets
    """
    datasets = []
    for dataset_name in dataset_names:
        _, test_loader = get_dataloader(dataset_name, preprocess)
        datasets.append(test_loader.dataset)
    concat_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader



def test_accuracy(visual_model,clip_model,logit_scale,data_loader,classifier=None,T=1):
    
    ## remove post modules ##
    if not isinstance(visual_model,SpikeModel):
        ln_post = visual_model.ln_post
        proj = visual_model.proj
        setattr(visual_model,'ln_post',StraightThrough())
        visual_model.proj = None
    else:
        ln_post = visual_model.model.ln_post
        proj = visual_model.model.proj
        setattr(visual_model.model,'ln_post',StraightThrough())
        visual_model.model.proj = None
    
    ## evaluate ##
    acc_list = []
    convert_model_precision(clip_model,device = device)
    if classifier is None:
        zeroshot_classifier = get_zeroshot_classifier(clip_model=clip_model,logit_scale=logit_scale,template_name='openai_imagenet_template',classnames=data_loader.dataset.classes,device=device)
    with torch.no_grad():
        visual_model.eval()
        for idx,(images,labels) in enumerate(tqdm(data_loader)):
            # encode images
            images = images.to(device)
            image_features_pre = visual_model(images)/T
            image_features = ln_post(image_features_pre)
            if proj is not None:
                image_features = image_features @ proj
            # classify
            if classifier is None: # zeroshot classify
                logits = zeroshot_classifier(image_features)
                probs = logits.softmax(dim=-1).cpu().numpy()  
            else: # finetune classify
                probs = classifier(image_features).cpu().numpy()
            # count
            acc = np.count_nonzero(np.argmax(probs,axis=1) == labels.cpu().numpy())/len(labels)
            acc_list.append(acc)
            # if idx==2: break
    
    ## recover post modules ##
    if not isinstance(visual_model,SpikeModel):
        setattr(visual_model,'ln_post',ln_post)
        visual_model.proj = proj
    else:
        setattr(visual_model.model,'ln_post',ln_post)
        visual_model.model.proj = proj
        
    avg_acc = sum(acc_list)/len(acc_list)
    return avg_acc


def model_convertion(ann,train_loader):

    convert_layers = [f'transformer.resblocks[{s}]' for s in args.convert_layers]

    ## replace gelu ##
    gnn = copy.deepcopy(ann)
    replace_gelu_with_relu(gnn,convert_layers=convert_layers,device=device,path=args.gelu_path,num_neurons=64)

    ## convert to snn ##
    snn = copy.deepcopy(gnn)
    snn = SpikeModel(snn,args.T,convert_layers=convert_layers, bipolar_with_memory=args.bipolar_with_memory, burst_T=args.burst_T)

    ## find threshold ##
    mse = False if args.method =='normal' else True
    get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=args.iters, mse=mse, percentile=args.percentile, T=args.T, neuron_wise=args.neuron_wise)

    torch.set_num_threads(10)
    
    snn.set_spike_state(use_spike=True)
    return gnn,snn



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--clip_model', default='ViT-B/32', type=str, help='network architecture', choices=['ViT-B/32'])
    parser.add_argument('--sample_data', default='cifar10', type=str, help='dataset for sampling', choices=['cifar10','cifar100','imagenet','imagenet200'])
    parser.add_argument('--test_datas',default='',nargs='+',help='dataset(s) for testing')
    parser.add_argument('--batch_size', default=50, type=int, help='minibatch size')
    parser.add_argument('--method', default='mmse', type=str, help='calibration methods', choices=['normal','mmse'])
    parser.add_argument('--T', default=32, type=int, help='snn simulation length')
    parser.add_argument('--seed', default=6, type=int, help='random seed')
    parser.add_argument('--iters', default=5, type=int, help='number of batches when getting maxinum activation')
    parser.add_argument('--percentile', default=1.0, type=float, help='percentile of maximum activation, invalid when mse is True')
    parser.add_argument('--convert_layers',default=['0','1','2','3','4','5','6','7','8','9','10','11'],nargs='+',help='layers to be converted')
    parser.add_argument('--neuron_wise', action='store_true', default=True, help='use channel-wise threshold')
    parser.add_argument('--bipolar_with_memory',action='store_true', default=True, help='use bipolar neuron with memory potential')
    parser.add_argument('--burst_T', default=2, type=int, help='simulation length for burst spikes')
    parser.add_argument('--load_vcm',action='store_true', help='load visual-classify model')
    parser.add_argument('--vcm_path', default=r'premodels/cifar-100.pth',type=str,help='finetuned visual-classifier model path')
    parser.add_argument('--gelu_path', default=r'premodels/distilled_gelu_64.pth',type=str,help='distilled gelu path')
    args = parser.parse_args()

    seed_all(args.seed)
    from model import DEVICE as device
    clip_model,preprocess = clip.load(args.clip_model,device = device)

    ## data prepare ##
    train_on_fullset = False
    train_on_testset = False
    test_datas = [args.sample_data] if args.test_datas == '' else args.test_datas
    train_loader, test_loader = get_dataloader(dataset_name=args.sample_data, preprocess=preprocess)
    
    if train_on_fullset:
        train_loader = get_concat_dataloader(dataset_names=test_datas, preprocess=preprocess)
    if train_on_testset:
        train_loader = DataLoader(test_loader.dataset, batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=num_workers,pin_memory=True)

    ## model prepare ##
    if args.load_vcm:
        fea_len = clip_model.visual.output_dim
        class_num = len(train_loader.dataset.classes)
        vcm = VisualClassifyModel(clip_model.visual,fea_len,class_num).cuda()
        vcm.load_state_dict(torch.load(args.vcm_path))
        ann = vcm.Featurizer
        classifier = vcm.Classifier
    else:
        ann = clip_model.visual
        classifier = None
    convert_model_precision(ann,device=device)
    logit_scale = clip_model.logit_scale

    ## convert ##
    gnn,snn = model_convertion(ann,train_loader)
    
    save_model = False
    if save_model:
        model_file_name = f'{args.sample_data}_{args.clip_model}_{args.T}_{args.method}_{ "+".join(args.convert_layers)}'
        torch.save(snn,f'converted_snn/{model_file_name}.pt')
        print("Model saved.")

    ## evaluate ##
    record = True
    torch.set_num_threads(10)
    res_dict = copy.deepcopy(dict(args.__dict__))
    test_acc = {}
    for data_name in test_datas:
        data_name = data_name.lower()
        print(data_name)
        _, test_loader = get_dataloader(dataset_name=data_name,preprocess=preprocess)
        aacc = test_accuracy(visual_model=ann,clip_model=clip_model,logit_scale=logit_scale,data_loader=test_loader,classifier=classifier,T=1)
        gacc = test_accuracy(visual_model=gnn,clip_model=clip_model,logit_scale=logit_scale,data_loader=test_loader,classifier=classifier,T=1)
        snn.set_spike_state(False)
        facc = test_accuracy(visual_model=snn,clip_model=clip_model,logit_scale=logit_scale,data_loader=test_loader,classifier=classifier,T=1)
        snn.set_spike_state(True)
        sacc = test_accuracy(visual_model=snn,clip_model=clip_model,logit_scale=logit_scale,data_loader=test_loader,classifier=classifier,T=args.T)
        print('ANN accuracy:',aacc)
        print('GNN accuracy:',gacc)
        print('FNN accuracy:',facc)
        print('SNN accuracy:',sacc)
        test_acc[f'{data_name.lower()}']={"ANN_acc":aacc,"GNN_acc":gacc,"FNN_acc":facc,"SNN_acc":sacc}
    res_dict['test_acc'] = test_acc
    # supplement other args
    res_dict['datetime'] = dtime
    res_dict['train_on_fullset'] = train_on_fullset
    res_dict['train_on_testset'] = train_on_testset
    
    ## save records ##
    # file_name = r'result_standard.json' if args.load_vcm else r'result_zeroshot.json'
    # try:
    #     with open(file_name,'r') as f:
    #         res_list = json.load(f) 
    # except json.JSONDecodeError:
    #     res_list = []
    # res_dict['convert_layers'] = "+".join(res_dict["convert_layers"])
    # res_list.append(res_dict)
    # with open(file_name,'w') as f:
    #     if record: f.write(json.dumps(res_list,indent=2))
    