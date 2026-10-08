#!/usr/bin/env python
"""Compare normal-only per-location patch calibration; anomaly labels are evaluation only."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from torchvision import models
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score
import simplenet
from datasets.lens_dirt import DatasetSplit,LensDirtDataset
from sensor_thresholds import sensor_from_path,youden_threshold

def patch_scores(net,loader,device):
 out=[];net.forward_modules.eval();net.pre_projection.eval();net.discriminator.eval()
 with torch.no_grad():
  for i,batch in enumerate(loader):
   image=batch['image'].to(device);n=len(image);features,shapes=net._embed(image,provide_patch_shapes=True,evaluation=True);p=-net.discriminator(net.pre_projection(features)).cpu().numpy();p=net.patch_maker.unpatch_scores(p,batchsize=n);p=p.reshape(n,-1);out.extend(p)
   if i%20==0:print('batches',i,'/',len(loader),flush=True)
 return np.asarray(out,dtype=np.float32)
def metrics(y,s,threshold):
 pred=s>=threshold;return {'auroc':float(roc_auc_score(y,s)),'pr_auc':float(average_precision_score(y,s)),'threshold':float(threshold),'f1':float(f1_score(y,pred)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',default='results/LensDirt_Results/sensor_heads/run/sensor_model.pth');ap.add_argument('--split',default='results/LensDirt_Results/stereo_validated/run_v2/split.json');ap.add_argument('--out',default='report/stereo_normal_calibration');a=ap.parse_args();dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');bundle=torch.load(a.bundle,map_location=dev);bb=models.wide_resnet50_2(weights=None);bb.name='wideresnet50';bb.seed=None;net=simplenet.SimpleNet(dev);net.load(backbone=bb,layers_to_extract_from=['layer2','layer3'],device=dev,input_shape=(3,180,240),pretrain_embed_dimension=1536,target_embed_dimension=1536,patchsize=3,embedding_size=256,meta_epochs=10,gan_epochs=4,noise_std=.015,dsc_hidden=1024,dsc_layers=2,dsc_margin=.5,pre_proj=1);net.backbone.load_state_dict(bundle['backbone']);h=bundle['heads']['stereo'];net.pre_projection.load_state_dict(h['pre_projection']);net.discriminator.load_state_dict(h['discriminator'])
 tr=LensDirtDataset('./data',split=DatasetSplit.TRAIN,frame_stride=2,per_scene_train=100000,seed=0);train_ids=[i for i,r in enumerate(tr.data_to_iterate) if sensor_from_path(r[2])=='stereo'];print('normal stereo training samples',len(train_ids),flush=True);train=patch_scores(net,DataLoader(Subset(tr,train_ids),batch_size=4,shuffle=False,num_workers=0),dev)
 pool=LensDirtDataset('./data',split=DatasetSplit.TEST,frame_stride=2,balance_scenes=False,strict_test_split=True,seed=0);lookup={r[2]:i for i,r in enumerate(pool.data_to_iterate)};sp=json.loads(Path(a.split).read_text());vals={};ys={}
 for name in ('validation','test'):
  rec=sp[name];ids=[lookup[p] for p,_ in rec];ys[name]=np.asarray([y for _,y in rec]);vals[name]=patch_scores(net,DataLoader(Subset(pool,ids),batch_size=4,shuffle=False,num_workers=0),dev)
 # Per-patch location normal score references; only training normals contribute.
 med=np.median(train,axis=0);q95=np.quantile(train,.95,axis=0);q99=np.quantile(train,.99,axis=0);scale=np.maximum(q95-med,1e-3)
 def transform(x):return {'raw_max':x.max(1),'raw_top10':np.sort(x,axis=1)[:,-10:].mean(1),'normal_z_top10':np.sort((x-med)/scale,axis=1)[:,-10:].mean(1),'normal_q95_count':(x>q95).sum(1).astype(float),'normal_q99_count':(x>q99).sum(1).astype(float),'normal_q99_excess':np.maximum(x-q99,0).sum(1)}
 outputs={k:{name:transform(v)[k] for name,v in vals.items()} for k in transform(vals['validation'])};rows=[]
 for method in outputs:
  threshold=youden_threshold(ys['validation'],outputs[method]['validation']);rows.append({'method':method,'validation':metrics(ys['validation'],outputs[method]['validation'],threshold),'test':metrics(ys['test'],outputs[method]['test'],threshold)})
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);payload={'training_normal_images':len(train),'patch_positions':int(train.shape[1]),'results':rows,'selection':'Highest validation AUROC only; test is not used for selection.','caveat':'Current fixed test was inspected in prior diagnostics and is not a pristine independent batch.'};(out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');np.savez_compressed(out/'raw_scores.npz',train=train,validation=vals['validation'],test=vals['test']);print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
