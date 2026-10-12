"""Extract Figure 5a curves from the user-supplied raster; preserve gaps.

Colours, axes and legend masks are explicit. This is approximate image-derived
reference data, not author raw data. Uncertainty: roughly 0.02 T and 3 Hz away
from steep features; line overlaps/labels are excluded rather than invented.
"""
from pathlib import Path
import json,csv,os
import numpy as np
from PIL import Image
from .paper_benchmark import ROOT

SOURCE=Path('/var/folders/lr/jhxj6sc57135vjzd356wp8w80000gp/T/codex-clipboard-bf42a992-57e9-4e80-a747-80984c03b7a9.png')

def main():
    im=Image.open(SOURCE if SOURCE.exists() else ROOT/'sources'/'user_figure5.png').convert('RGB');original=im.size
    im=im.resize((1471,round(im.height*1471/im.width)),Image.Resampling.LANCZOS)
    a=np.array(im).astype(float);r,g,b=a[:,:,0],a[:,:,1],a[:,:,2]
    masks=[(np.maximum.reduce([r,g,b])-np.minimum.reduce([r,g,b])<13)&(r>60)&(r<190),
           (r>170)&(g<145)&(b<150)&(r>g+45),
           (b>130)&(r<100)&(g>65)&(b>g+25),
           (g>95)&(g>r+30)&(g>b+12),
           (b>145)&(r>115)&(g<165)&(r>g+25)&(b>g+35),
           (r>135)&(g>90)&(b<100)&(r>g+15)]
    x0,x1=75,722;ytop,ybottom=61,862;dy=(ybottom-ytop)/6
    labels=[.4,.47,.53,.6,.67,.73];points=[]
    for j,(label,mask) in enumerate(zip(labels,masks)):
        top=ytop+j*dy;bottom=top+dy
        ok=np.zeros(mask.shape,bool);ok[int(top)+3:int(bottom)-(3 if j==0 else 0),x0+2:x1-1]=True
        ok[int(bottom-.1*dy)-2:int(bottom-.1*dy)+3]=False
        # Legends sit at top-right in first three rows, top-left in last three.
        la,lb=(510,722) if j<3 else (80,310)
        ok[int(top)+1:int(top)+30,la:lb]=False
        if j==0:ok[:,395:402]=False
        mask &=ok
        for x in range(x0+2,x1-1):
            ys=np.flatnonzero(mask[:,x])
            if not len(ys) or np.ptp(ys)>7:continue
            y=float(np.median(ys));f=1700+(x-x0)*600/(x1-x0);T=(bottom-y)/dy
            points.append(dict(openness=label,frequency_hz=f,T=T,pixel_x=x,pixel_y=y))
    path=ROOT/'reference_fig5a.csv'
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=points[0].keys());w.writeheader();w.writerows(points)
    (ROOT/'reference_provenance.json').write_text(json.dumps(dict(source=str(SOURCE),original_size=original,
        resized_width=1471,x_pixels=[x0,x1],frequency_hz=[1700,2300],top_bottom_pixels=[ytop,ybottom],
        note='Digitised user-provided Figure 5a, stacked traces each span T=0..1. No digitisation of field colours.',
        approximate_T_uncertainty=.02,approximate_frequency_uncertainty_hz=3),indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,10));ax.imshow(im);ax.scatter([p['pixel_x'] for p in points],[p['pixel_y'] for p in points],s=1,color='cyan')
    ax.set(xlim=(40,750),ylim=(890,20),title='Digitisation QA: cyan points over supplied reference');fig.savefig(ROOT/'reference_digitisation_check.png',dpi=140);plt.close(fig)
    print(len(points),'reference points')

if __name__=='__main__':main()
