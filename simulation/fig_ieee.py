import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S=json.load(open('results/summary.json'))
plt.rcParams.update({"font.size":7,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
SC=[("med",1),("med",2),("high",1),("high",2)]
C=[("fixed","Fixed-time","#9b9a97",""),("actuated","Actuated","#eda100","...."),("mp","Reactive MP","#2a78d6",""),("pmp_hgb_guard","Predictive MP + guard","#eb6834","////")]
fig,ax=plt.subplots(figsize=(3.45,2.15)); w=0.2
for i,(k,n,col,h) in enumerate(C):
    m=[S[f"{d}{b}_{k}_delay"][0] for d,b in SC]; e=[S[f"{d}{b}_{k}_delay"][1] for d,b in SC]
    ax.bar(np.arange(4)+(i-1.5)*w,m,w*0.9,yerr=e,color=col,hatch=h,edgecolor="white",linewidth=0.4,error_kw=dict(lw=0.5,capsize=1.2,ecolor="#52514e"),label=n)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{'Medium' if d=='med' else 'High'}\n{'1 of 2 lanes' if b==1 else 'full closure'}" for d,b in SC],fontsize=6.5)
ax.set_ylabel("Average delay (s/veh)"); ax.yaxis.grid(True,color="#e6e5e1",lw=0.5); ax.set_axisbelow(True)
ax.legend(ncol=2,fontsize=6,frameon=False,loc="upper center",bbox_to_anchor=(0.5,1.25))
fig.tight_layout(pad=0.3); fig.savefig('../paper/figures/fig_delay_ieee.pdf'); fig.savefig('results/fig/fig_delay_ieee.png',dpi=220)
fig,axs=plt.subplots(1,2,figsize=(3.45,1.75))
for ax,d in zip(axs,("med","high")):
    Ls=[0,60,120,300]; m=[S[f"rob_{d}_{L}"][0] for L in Ls]; e=[S[f"rob_{d}_{L}"][1] for L in Ls]
    ax.errorbar(Ls,m,yerr=e,color="#eb6834",marker="o",ms=3,lw=1.2,capsize=1.5,label="Predictive MP + guard")
    ax.axhline(S[f"{d}2_mp_delay"][0],color="#2a78d6",lw=1,ls="--",label="Reactive MP")
    ax.axhline(S[f"{d}2_actuated_delay"][0],color="#eda100",lw=1,ls=":",label="Actuated")
    ax.set_title(f"{'Medium' if d=='med' else 'High'} demand",fontsize=7); ax.set_xlabel("Detection latency (s)"); ax.set_xticks(Ls)
    ax.yaxis.grid(True,color="#e6e5e1",lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel("Avg. delay (s/veh)")
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,ncol=3,fontsize=5.8,frameon=False,loc="upper center")
fig.tight_layout(pad=0.3,rect=(0,0,1,0.88)); fig.savefig('../paper/figures/fig_latency_ieee.pdf')
