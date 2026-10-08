#!/usr/bin/env python
"""Evaluate patch aggregation choices without changing weights or training."""
import argparse,json,copy
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from torchvision import models
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score
import simplenet
from datasets.lens_dirt import LensDirtDataset,DatasetSplit
from sensor_thresholds import sensor_from_path,youden_threshold

def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',default='results/LensDirt_Results/sensor_heads/run/sensor_model.pth');p.add_argument('--split',default='results/LensDirt_Results/stereo_validated/run_v2/split.json');p.add_argument('--subset',choices=('validation','test'),default='validation');p.add_argument('--out',default='report/stereo_aggregation');a=p.parse_args()
 dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');b=torch.load(a.bundle,map_location=dev);bb=models.wide_resnet50_2(weights=None);bb.name='wideresnet50';bb.seed=None;net=simplenet.SimpleNet(dev);net.load(backbone=bb,layers_to_extract_from=['layer2','layer3'],device=dev,input_shape=(3,180,240),pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,embedding_size=256,meta_epochs=10,gan_epochs=4,noise_std=.015,dsc_hidden=1024,dsc_layers=2,dsc_margin=.5,pre_proj=1);net.backbone.load_state_dict(b['backbone']);h=b['heads']['stereo'];net.pre_projection.load_state_dict(h['pre_projection']);net.discriminator.load_state_dict(h['discriminator']);net.forward_modules.eval(); net.pre_projection.eval(); net.discriminator.eval()
 split=json.loads(Path(a.split).read_text())[a.subset];paths=[r[0] for r in split];labels=np.array([r[1] for r in split]);ds=LensDirtDataset('./data',split=DatasetSplit.TEST,frame_stride=2,balance_scenes=False,strict_test_split=True);idx={r[2]:i for i,r in enumerate(ds.data_to_iterate)};loader=DataLoader(Subset(ds,[idx[x] for x in paths]),batch_size=4,shuffle=False);raw=[]
 with torch.no_grad():
  for batch in loader:
   f,_=net._embed(batch['image'].to(dev),provide_patch_shapes=True,evaluation=True);f=net.pre_projection(f);raw.extend((-net.discriminator(f)).detach().cpu().numpy().reshape(len(batch['image']),-1))
 raw=np.asarray(raw);methods={'max':1,'top3':3,'top5':5,'top10':10};rows=[]
 for name,k in methods.items():
  scores=np.sort(raw,axis=1)[:,-k:].mean(1);auc=float(roc_auc_score(labels,scores));thr=youden_threshold(labels,scores);pred=scores>=thr;cm=confusion_matrix(labels,pred,labels=[0,1]).tolist();rows.append({'aggregation':name,'k':k,'auroc':auc,'pr_auc':float(average_precision_score(labels,scores)),'threshold':thr,'f1':float(f1_score(labels,pred)),'confusion_matrix':cm})
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/f'{a.subset}.json').write_text(json.dumps({'subset':a.subset,'results':rows,'note':'Weights unchanged; thresholds fitted only for this diagnostic subset.'},indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
