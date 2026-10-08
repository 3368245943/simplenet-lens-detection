#!/usr/bin/env python
"""Train independent left/right stereo heads with frozen shared backbone."""
import argparse, copy, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import roc_auc_score
import simplenet, utils
from datasets.lens_dirt import DatasetSplit, LensDirtDataset
from sensor_thresholds import sensor_from_path

def side(p): return 'left' if '/left/' in p else 'right'

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='results/LensDirt_Results/sensor_heads/run/sensor_model.pth');p.add_argument('--out',default='results/LensDirt_Results/stereo_sides/run');p.add_argument('--epochs',type=int,default=2);p.add_argument('--lr',type=float,default=5e-6);a=p.parse_args();out=Path(a.out)
 if out.exists():p.error(f'exists: {out}')
 utils.fix_seeds(0);dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); b=torch.load(a.base,map_location='cpu')
 tr=LensDirtDataset('./data',split=DatasetSplit.TRAIN,frame_stride=2,per_scene_train=100000,seed=0); pool=LensDirtDataset('./data',split=DatasetSplit.TEST,frame_stride=2,balance_scenes=False,strict_test_split=True,seed=0)
 split=json.loads(Path('results/LensDirt_Results/stereo_validated/run_v2/split.json').read_text()); row={r[2]:i for i,r in enumerate(pool.data_to_iterate)}
 class Eval(torch.utils.data.Dataset):
  def __init__(self,records,s):self.records=[r for r in records if side(r[0])==s]
  def __len__(self):return len(self.records)
  def __getitem__(self,i):
   p,y=self.records[i];x=dict(pool[row[p]]);x['is_anomaly']=y;return x
 metrics={}; heads={}
 for s in ('left','right'):
  ti=[i for i,r in enumerate(tr.data_to_iterate) if sensor_from_path(r[2])=='stereo' and side(r[2])==s]
  va=Eval(split['validation'],s);vl=DataLoader(va,batch_size=4,shuffle=False);tl=DataLoader(Subset(tr,ti),batch_size=4,shuffle=True)
  bb=models.wide_resnet50_2(weights=None);bb.name='wideresnet50';bb.seed=None;net=simplenet.SimpleNet(dev);net.load(backbone=bb,layers_to_extract_from=['layer2','layer3'],device=dev,input_shape=(3,180,240),pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,embedding_size=256,meta_epochs=a.epochs,gan_epochs=2,noise_std=.015,dsc_hidden=1024,dsc_layers=2,dsc_margin=.5,pre_proj=1);net.backbone.load_state_dict(b['backbone']);
  for q in net.backbone.parameters():q.requires_grad_(False)
  net.pre_projection.load_state_dict(b['heads']['stereo']['pre_projection']);net.discriminator.load_state_dict(b['heads']['stereo']['discriminator']);net.proj_opt=torch.optim.AdamW(net.pre_projection.parameters(),lr=a.lr);net.dsc_opt=torch.optim.Adam(net.discriminator.parameters(),lr=a.lr,weight_decay=1e-5);net.set_model_dir(str(out/'logs'/s),'lens')
  y=np.array([x[1] for x in va.records]); best=(-1,copy.deepcopy(net.pre_projection.state_dict()),copy.deepcopy(net.discriminator.state_dict())); history=[]
  for e in range(a.epochs+1):
   scores=np.asarray(net.predict(vl)[0]).reshape(-1);auc=float(roc_auc_score(y,scores));history.append(auc);print(s,'epoch',e,'val_auc',auc,flush=True)
   if auc>best[0]:best=(auc,copy.deepcopy(net.pre_projection.state_dict()),copy.deepcopy(net.discriminator.state_dict()))
   if e<a.epochs:net._train_discriminator(tl)
  heads[s]={'pre_projection':{k:v.cpu() for k,v in best[1].items()},'discriminator':{k:v.cpu() for k,v in best[2].items()}};metrics[s]={'train':len(ti),'validation':len(va),'best_val_auroc':best[0],'history':history}
 out.mkdir(parents=True,exist_ok=True);result=copy.deepcopy(b);result['stereo_side_heads']=heads;torch.save(result,out/'sensor_model_sides.pth');(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print('saved',out/'sensor_model_sides.pth')
if __name__=='__main__':main()
