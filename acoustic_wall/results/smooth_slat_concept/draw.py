import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from pathlib import Path
out=Path(__file__).parent
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,8),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
solid='#374553';air='#eff7f8';accent='#118b88';orange='#d57b27'
# Top-down plan: slats repeat horizontally, sound travels in depth direction.
for a in [ax,bx]:a.set_aspect('equal');a.set_facecolor('white')
for x in [0,200]:ax.add_patch(Rectangle((x,0),40,180,facecolor=solid))
ax.add_patch(Rectangle((40,0),160,180,facecolor=air))
ax.annotate('',(200,195),(40,195),arrowprops=dict(arrowstyle='<->',color=accent,lw=2))
ax.text(120,201,'160 mm unobstructed gap',ha='center',color=accent,fontsize=11)
ax.annotate('',(40,-15),(0,-15),arrowprops=dict(arrowstyle='<->',color=solid))
ax.text(20,-23,'40 mm',ha='center',va='top')
ax.annotate('',(200,-43),(0,-43),arrowprops=dict(arrowstyle='<->',color=solid))
ax.text(100,-50,'200 mm pitch',ha='center',va='top')
for x in [80,120,160]:ax.annotate('',(x,65),(x,-2),arrowprops=dict(arrowstyle='->',color=accent,lw=2))
ax.text(120,-75,'ROAD / INCIDENT SOUND',ha='center',fontsize=10)
ax.text(120,232,'YARD',ha='center',fontsize=10)
ax.set(xlim=(-15,255),ylim=(-85,245),title='A · Horizontal section through the fence\n80% clear before posts and rails')
ax.axis('off')
# One hollow slat: two straight closed-ended cavities each opened sideways at end.
# 40mm width, 180mm illustrative depth, 3mm external wall, 4mm central septum.
bx.add_patch(Rectangle((0,0),40,180,facecolor=solid))
# left cavity z20..127, right z60..146: gross axial lengths107,86. last20mm side mouth.
for x,z,L,side,col,label in [(3,20,107,'left',accent,'L₁ ≈ 107 mm'),(22,60,86,'right',orange,'L₂ ≈ 86 mm')]:
 bx.add_patch(Rectangle((x,z),15,L,facecolor=air))
 mouth_z=z; mouth_h=20
 bx.add_patch(Rectangle((0 if side=='left' else 37,mouth_z),3,mouth_h,facecolor=air))
 screenx=0 if side=='left' else 40
 bx.plot([screenx,screenx],[mouth_z,mouth_z+mouth_h],color=col,lw=2,linestyle=':')
 mid=x+7.5
 bx.annotate('',(mid,z+L-3),(mid,z+3),arrowprops=dict(arrowstyle='<->',color=col,lw=1.5))
 bx.text(mid,z+L/2,label,rotation=90,ha='center',va='center',fontsize=9,color=col,bbox=dict(facecolor=air,edgecolor='none',pad=1))
# outside geometry notes
bx.annotate('Large side-facing mouth\n20 mm illustrative opening\nRemovable insect screen',xy=(0,30),xytext=(-95,5),fontsize=10,ha='left',color=accent)
bx.annotate('Second side-facing mouth\nPosition along depth is variable',xy=(40,70),xytext=(65,45),fontsize=10,arrowprops=dict(arrowstyle='-',color=orange),color=orange)
bx.annotate('Closed ends\nStraight, smooth passages',xy=(30,149),xytext=(65,155),fontsize=10,arrowprops=dict(arrowstyle='-',color=solid))
bx.annotate('',(-12,180),(-12,0),arrowprops=dict(arrowstyle='<->',color=solid))
bx.text(-18,105,'Depth D ≈ 180 mm',rotation=90,ha='right',va='center',fontsize=10)
bx.annotate('',(40,195),(0,195),arrowprops=dict(arrowstyle='<->',color=solid))
bx.text(20,202,'Slat width b = 40 mm',ha='center',fontsize=11)
bx.text(20,-25,'Sound direction ↑',ha='center',color=accent,fontsize=11)
bx.text(20,-48,'Removable cover on the slat’s broad face\nopens both cavities for cleaning (not drawn).',ha='center',fontsize=10)
bx.set(xlim=(-100,170),ylim=(-85,245),title='B · Enlarged slat interior — proposed topology\nTwo smooth cavities; no fine internal ribs')
bx.axis('off')
fig.suptitle('Maintainable resonator slat: a geometry to optimise',fontsize=19,fontweight='bold')
fig.supxlabel('Vary: passage lengths L₁/L₂, mouth size and position, slat depth D, and cavity count.\nHard constraint: ≥75% true front-view opening across the complete wall, including posts and rails.\nIllustrative dimensions only. Effective acoustic lengths require end corrections and coupling; no attenuation has been simulated for this design.',fontsize=11)
fig.savefig(out/'cross_section.png',dpi=180);fig.savefig(out/'cross_section.svg');plt.close(fig)
