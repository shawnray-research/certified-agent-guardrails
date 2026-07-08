"""Central-claim figure: judge discrimination (AUC), not scale, sets the frontier.
Left: AUC vs parameter count across three model families on both datasets, showing
AUC is not monotone in size across families (8B Llama < 3B Qwen on AgentDojo).
Right: utility loss at fixed safety delta=0.10 collapses onto a single decreasing
curve in AUC regardless of family/size, i.e. the frontier is a function of AUC."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style as _style; _style.apply_style()
import enforce as E
from agentdojo.task_suite.load_suites import get_suites

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
sc = json.load(open(f"{HERE}/scores_cache.json")); bc = json.load(open(f"{HERE}/bench_scores_cache.json"))

# (model, params_B, family)
JUDGES = [("qwen2.5:0.5b",0.5,"Qwen"),("qwen2.5:3b",3,"Qwen"),("qwen2.5:7b",7,"Qwen"),
          ("qwen2.5:32b",32,"Qwen"),("llama3.1:8b",8,"Llama"),("llama3.3:70b",70,"Llama"),
          ("gemma2:9b",9,"Gemma")]
# Okabe-Ito colorblind-safe palette; families ALSO differ by marker shape
FAM_C = {"Qwen":"#0072B2","Llama":"#D55E00","Gemma":"#009E73"}
FAM_M = {"Qwen":"o","Llama":"s","Gemma":"^"}

acts = json.load(open(f"{HERE}/actions.json"))["actions"]
cur_t=[a["text"] for a in acts]; cur_l=np.array([a["label"] for a in acts])
suites=get_suites("v1.2.1"); ben_t=[]; ben_l=[]
for n,s in suites.items():
    for u in s.user_tasks.values(): ben_t.append(u.PROMPT); ben_l.append(0)
    for i in s.injection_tasks.values(): ben_t.append(i.GOAL); ben_l.append(1)
ben_l=np.array(ben_l)

def auc(u,s): return float(np.mean([(x>y)+0.5*(x==y) for x in u for y in s]))
def fast_auc(s,y):
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True); csum=np.cumsum(cnt)
    ranks=((csum-cnt+csum+1)/2.0)[inv]; npos=y.sum(); nneg=len(y)-npos
    return (ranks[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
def boot_ci(s,y,n=1000,seed=20260706):
    rng=np.random.default_rng(seed); b=[]
    for _ in range(n):
        idx=rng.integers(0,len(s),len(s))
        if len(set(y[idx]))<2: continue
        b.append(fast_auc(s[idx],y[idx]))
    return np.percentile(b,[2.5,97.5])
def uloss(scores,labels,delta=0.10,n=400):
    rng=np.random.default_rng(20260704); i1=np.where(labels==1)[0]; i0=np.where(labels==0)[0]; U=[]
    for _ in range(n):
        u=rng.permutation(i1); ci=u[:len(u)//2]; th=E.conformal_threshold(scores[ci],delta); U.append(np.mean(scores[i0]>=th))
    return float(np.mean(U))

rows=[]; bench_ci={}
for m,p,fam in JUDGES:
    cs=np.array([sc.get(f"{m}::{t}",np.nan) for t in cur_t])
    bs=np.array([bc.get(f"{m}::::{t}",np.nan) for t in ben_t])
    ca=auc(cs[cur_l==1],cs[cur_l==0]); ba=auc(bs[ben_l==1],bs[ben_l==0]) if not np.isnan(bs).any() else np.nan
    cu=uloss(cs,cur_l); bu=uloss(bs,ben_l) if not np.isnan(bs).any() else np.nan
    if not np.isnan(bs).any(): bench_ci[m]=boot_ci(bs,ben_l)
    rows.append((m,p,fam,ca,ba,cu,bu))

fig,(axL,axR)=plt.subplots(1,2,figsize=(10.6,3.0))
# left: AUC vs params
for ds,key,mk_alpha,lbl in [("cur",3,1.0,"curated"),("adojo",4,0.45,"AgentDojo")]:
    for m,p,fam,ca,ba,cu,bu in rows:
        y=ca if ds=="cur" else ba
        if np.isnan(y): continue
        if ds=="adojo" and m in bench_ci:   # bootstrap 95% CI error bars on the ceiling test
            lo,hi=bench_ci[m]
            axL.errorbar(p,y,yerr=[[y-lo],[hi-y]],fmt="none",ecolor=FAM_C[fam],alpha=0.5,capsize=3,lw=1.1,zorder=2)
        axL.scatter(p,y,c=FAM_C[fam],marker=FAM_M[fam],s=80,alpha=mk_alpha,edgecolors="k",linewidths=0.5,zorder=3)
# within-Qwen trend line per dataset
for ds,idx in [("cur",3),("adojo",4)]:
    q=sorted([(p,(ca if ds=="cur" else ba)) for m,p,fam,ca,ba,cu,bu in rows if fam=="Qwen" and not np.isnan(ca if ds=="cur" else ba)])
    axL.plot([p for p,_ in q],[y for _,y in q],c="#0072B2",lw=1.2,alpha=0.4 if ds=="adojo" else 0.9,zorder=1)
axL.set_xscale("log"); axL.set_xlabel("judge size (B params, log)"); axL.set_ylabel("judge AUC")
axL.set_xticks([0.5,3,7,9,32,70]); axL.set_xticklabels(["0.5","3","7","9","32","70"])
axL.annotate("8B, 70B Llama < 9B Gemma\n(AgentDojo)",xy=(8,0.66),xytext=(1.2,0.55),fontsize=10.5,
             arrowprops=dict(arrowstyle="->",color="#D55E00",lw=1.3),color="#D55E00")
axL.set_ylim(0.45,1.02); axL.set_title("AUC is not monotone in size (AgentDojo bars: 95% CI)")
# right: utility loss vs AUC (both datasets)
for m,p,fam,ca,ba,cu,bu in rows:
    axR.scatter(ca,cu,c=FAM_C[fam],marker=FAM_M[fam],s=80,alpha=1.0,edgecolors="k",linewidths=0.5,zorder=3)
    if not np.isnan(ba): axR.scatter(ba,bu,c=FAM_C[fam],marker=FAM_M[fam],s=80,alpha=0.45,edgecolors="k",linewidths=0.5,zorder=3)
axR.set_xlabel("judge AUC"); axR.set_ylabel(r"utility loss $U$ at $\delta=0.10$")
axR.set_title(r"Utility loss falls with AUC, across families")
axR.set_xlim(0.5,1.02); axR.set_ylim(-0.03,1.05)
# family legend
from matplotlib.lines import Line2D
h=[Line2D([0],[0],marker=FAM_M[f],color="w",markerfacecolor=FAM_C[f],markeredgecolor="k",markersize=9,label=f) for f in FAM_C]
h+=[Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markeredgecolor="k",markersize=9,alpha=1.0,label="curated"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markeredgecolor="k",markersize=9,alpha=0.45,label="AgentDojo")]
# single shared legend BELOW both panels (centered) so the figure is horizontally
# symmetric; a right-side legend made bbox_inches="tight" pad the right and look off-center
fig.legend(handles=h,fontsize=11,loc="lower center",bbox_to_anchor=(0.5,-0.01),
           ncol=len(h),framealpha=0.9,frameon=False)
fig.subplots_adjust(left=0.06,right=0.98,bottom=0.30,top=0.90,wspace=0.28)
fig.savefig(os.path.join(FIG,"scaling.pdf"),bbox_inches="tight"); plt.close()
print("wrote figures/scaling.pdf")

# compact single-column single-panel version for the main text (the decisive panel)
figM,axM=plt.subplots(figsize=(5,3.5))
for ds in ["cur","adojo"]:
    a_alpha=1.0 if ds=="cur" else 0.45
    for m,p,fam,ca,ba,cu,bu in rows:
        y=ca if ds=="cur" else ba
        if np.isnan(y): continue
        axM.scatter(p,y,c=FAM_C[fam],marker=FAM_M[fam],s=95,alpha=a_alpha,edgecolors="k",linewidths=0.5,zorder=3)
for ds in ["cur","adojo"]:
    q=sorted([(p,(ca if ds=="cur" else ba)) for m,p,fam,ca,ba,cu,bu in rows if fam=="Qwen" and not np.isnan(ca if ds=="cur" else ba)])
    axM.plot([p for p,_ in q],[y for _,y in q],c="#0072B2",lw=1.2,alpha=0.4 if ds=="adojo" else 0.9,zorder=1)
axM.set_xscale("log"); axM.set_xlabel("judge size (B params, log)"); axM.set_ylabel("judge AUC")
axM.set_xticks([0.5,3,7,9,32,70]); axM.set_xticklabels(["0.5","3","7","9","32","70"])
axM.annotate("8B, 70B Llama beaten\nby 9B Gemma (AgentDojo)",xy=(8,0.66),xytext=(1.0,0.55),fontsize=10.5,
             arrowprops=dict(arrowstyle="->",color="#D55E00",lw=1.2),color="#D55E00")
axM.set_ylim(0.45,1.03)
hh=[Line2D([0],[0],marker=FAM_M[f],color="w",markerfacecolor=FAM_C[f],markeredgecolor="k",markersize=9,label=f) for f in FAM_C]
hh+=[Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markeredgecolor="k",markersize=9,label="curated"),
     Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markeredgecolor="k",markersize=9,alpha=0.45,label="AgentDojo")]
axM.legend(handles=hh,fontsize=9,loc="lower right",ncol=1,framealpha=0.9)
figM.tight_layout(); figM.savefig(os.path.join(FIG,"scaling_main.pdf")); plt.close()
print("wrote figures/scaling_main.pdf")
for m,p,fam,ca,ba,cu,bu in rows:
    print(f"  {m:14s} {fam:10s} p={p:4} curAUC={ca:.3f} adojoAUC={ba if not np.isnan(ba) else 0:.3f}")
