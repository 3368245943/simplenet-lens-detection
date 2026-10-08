#!/usr/bin/env python
"""Compare stereo preprocessing and emit an intuitive histogram + confusion report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN=np.array([.485,.456,.406],np.float32).reshape(3,1,1)
STD=np.array([.229,.224,.225],np.float32).reshape(3,1,1)
MODES=('current','percentile','roi','roi_percentile')
MODE_NAMES={'current':'当前预处理','percentile':'百分位归一化','roi':'中心ROI','roi_percentile':'ROI+百分位'}

def image_tensor(path, mode):
    with Image.open(path) as im:
        gray=np.asarray(im.convert('L'),dtype=np.float32)/255.
    if mode in ('roi','roi_percentile'):
        h,w=gray.shape
        gray=gray[int(h*.05):int(h*.95),int(w*.05):int(w*.95)]
    if mode in ('percentile','roi_percentile'):
        lo,hi=np.percentile(gray,[1,99])
        gray=np.clip((gray-lo)/max(hi-lo,1e-6),0,1)
    im=Image.fromarray(np.uint8(gray*255)).resize((240,180),Image.BILINEAR)
    x=np.broadcast_to(np.asarray(im,dtype=np.float32)/255.,(3,180,240)).copy()
    return ((x-MEAN)/STD)[None]

def side(path): return 'left' if '/left/' in path else 'right'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx')
    parser.add_argument('--split',default='results/LensDirt_Results/stereo_validated/run_v2/split.json')
    parser.add_argument('--subset',choices=('validation','test'),default='test')
    parser.add_argument('--out',default='report/stereo_analysis')
    args=parser.parse_args()
    records=json.loads(Path(args.split).read_text())[args.subset]
    paths=[r[0] for r in records]
    labels=np.asarray([r[1] for r in records],dtype=np.int64)
    sides=np.asarray([side(p) for p in paths])
    session=ort.InferenceSession(args.model,providers=['CPUExecutionProvider'])
    threshold=float(session.get_modelmeta().custom_metadata_map['thresholds'].split(',')[1])
    data={mode:[] for mode in MODES}
    for index,path in enumerate(paths):
        if index%50==0: print(f'{index}/{len(paths)}',flush=True)
        for mode in MODES:
            score,_=session.run(None,{'image':image_tensor(path,mode),'sensor_id':np.array([1],np.int64)})
            data[mode].append(float(score[0]))
    for mode in MODES: data[mode]=np.asarray(data[mode])
    summary=[]
    for mode in MODES:
        for group in ('all','left','right'):
            mask=np.ones(len(paths),dtype=bool) if group=='all' else sides==group
            yy=labels[mask]; ss=data[mode][mask]
            summary.append({'preprocess':mode,'side':group,'n':int(mask.sum()),
                'normal':int((yy==0).sum()),'anomaly':int((yy==1).sum()),
                'auroc':float(roc_auc_score(yy,ss)),
                'pr_auc':float(average_precision_score(yy,ss)),
                'normal_mean':float(ss[yy==0].mean()),'anomaly_mean':float(ss[yy==1].mean()),
                'threshold_used_for_confusion':threshold if mode=='current' else None})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    payload={'subset':args.subset,'results':summary,
        'threshold':threshold,
        'note':'Confusion matrices use the embedded threshold with current preprocessing only. Other preprocessing AUROC comparisons are threshold-independent; do not reuse the threshold for changed preprocessing.'}
    (out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    for family in ('Noto Sans CJK SC','WenQuanYi Micro Hei','DejaVu Sans'):
        try: font_manager.findfont(family,fallback_to_default=False);plt.rcParams['font.family']=family;break
        except ValueError: continue
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(16,10),gridspec_kw={'height_ratios':[1.05,1.0]})
    groups=('left','right','all'); colors={0:'#247b69',1:'#c25746'}
    score=data['current']; lo,hi=float(score.min()),float(score.max())
    if lo==hi:lo,hi=lo-.01,hi+.01
    bins=np.linspace(lo,hi,25)
    for col,group in enumerate(groups):
        mask=np.ones(len(paths),bool) if group=='all' else sides==group
        yy=labels[mask];ss=score[mask]
        ax=axes[0,col]
        ax.hist(ss[yy==0],bins=bins,color=colors[0],alpha=.72,label=f'正常 {int((yy==0).sum())}')
        ax.hist(ss[yy==1],bins=bins,color=colors[1],alpha=.65,label=f'脏污 {int((yy==1).sum())}')
        ax.axvline(threshold,color='#263746',linestyle='--',linewidth=1.8,label=f'阈值 {threshold:.3f}')
        row=next(r for r in summary if r['preprocess']=='current' and r['side']==group)
        ax.set_title(f"{ {'left':'左目','right':'右目','all':'左右目合计'}[group] }  ·  AUROC {row['auroc']:.3f}",fontsize=12)
        ax.set_xlabel('异常分数（越高越异常）');ax.set_ylabel('图片数');ax.legend(frameon=False,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        # Use decisions made by the real model for the unchanged preprocessing.
        pred=(ss>=threshold).astype(np.int64);cm=confusion_matrix(yy,pred,labels=[0,1])
        ax=axes[1,col];ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(1,int(cm.max())))
        ax.set_title(f"混淆矩阵  ·  F1 {f1_score(yy,pred):.3f}",fontsize=12)
        ax.set_xticks([0,1],labels=['正常','脏污']);ax.set_yticks([0,1],labels=['正常','脏污'])
        ax.set_xlabel('模型预测');ax.set_ylabel('真实标签')
        for r in range(2):
            for c in range(2):
                ax.text(c,r,str(int(cm[r,c])),ha='center',va='center',fontsize=16,
                    color='white' if cm[r,c]>cm.max()/2 else '#263746')
    auc_lines=[]
    for mode in MODES:
        values=[next(r['auroc'] for r in summary if r['preprocess']==mode and r['side']==g) for g in groups]
        auc_lines.append(f"{MODE_NAMES[mode]}：左 {values[0]:.3f}  /  右 {values[1]:.3f}  /  合计 {values[2]:.3f}")
    fig.suptitle(f"双目脏污区分报告 · {'验证集' if args.subset=='validation' else '测试集'}",fontsize=17,y=.99)
    fig.text(.5,.025,'预处理 AUROC 对照（不依赖阈值）\n'+'\n'.join(auc_lines),ha='center',va='bottom',fontsize=10,
        bbox={'facecolor':'#f3f5f6','edgecolor':'none','boxstyle':'round,pad=.55'})
    fig.text(.5,.155,'混淆矩阵使用模型内原阈值；仅当前预处理可直接解释，其他预处理尚未重新标定阈值。',ha='center',fontsize=9,color='#596775')
    fig.tight_layout(rect=(0,.20,1,.96))
    fig.savefig(out/'report.png',dpi=150);plt.close(fig)
    print(json.dumps(payload,ensure_ascii=False,indent=2));print('PNG report:',out/'report.png')
if __name__=='__main__':main()
