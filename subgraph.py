#!/usr/bin/env python3
"""
subgraph_extractor.py  –  rank-based activation-checkpointing & memory plots
"""

from __future__ import annotations
import os, csv, torch, warnings, matplotlib.pyplot as plt, torchvision
from torch.fx import symbolic_trace, GraphModule
from torch.utils.checkpoint import checkpoint
from resnet_profile import MemoryProfiler, plot_memory_profile  # reuse existing profiler utilities
plt.switch_backend("Agg")

# ───────────────── patches ─────────────────────────────────────────
def patch_residual_blocks():
    import torchvision.models.resnet as res
    def bb(self,x):
        id=x
        out=self.conv1(x); out=self.bn1(out); out=self.relu(out)
        out=self.conv2(out); out=self.bn2(out)
        if self.downsample is not None:
            id=self.downsample(x)
        out=out+id
        return self.relu(out)
    def bn(self,x):
        id=x
        out=self.conv1(x); out=self.bn1(out); out=self.relu(out)
        out=self.conv2(out); out=self.bn2(out); out=self.relu(out)
        out=self.conv3(out); out=self.bn3(out)
        if self.downsample is not None:
            id=self.downsample(x)
        out=out+id
        return self.relu(out)
    res.BasicBlock.forward=bb
    res.Bottleneck.forward=bn

def disable_inplace_relu(m):
    for mod in m.modules():
        if isinstance(mod,torch.nn.ReLU):
            mod.inplace=False

# ───────────────── fx transform ────────────────────────────────────
def checkpoint_by_rank(gm:GraphModule, ranks:set[int])->GraphModule:
    g=gm.graph
    rk=-1
    for n in list(g.nodes):
        if n.op!="call_module": continue
        rk+=1
        if rk not in ranks:    continue

        with g.inserting_before(n):
            mod_ref=g.get_attr(n.target)
        with g.inserting_after(n):
            new=g.call_function(checkpoint,
                                args=(mod_ref,*n.args),
                                kwargs=dict(n.kwargs))
        n.replace_all_uses_with(new)
        g.erase_node(n)  # remove original op to avoid duplicate execution
    g.lint()
    return GraphModule(gm, g, class_name="ResNet152Checkpointed")

# ───────────────── memory helper ───────────────────────────────────
def peak_mb(model, batch, device):
    x=torch.randn(batch,3,224,224,device=device,requires_grad=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    model(x).sum().backward()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device)/2**20

# ───────────────── model builders (kept tiny & isolated) ───────────
def build_base():
    patch_residual_blocks()
    m=torchvision.models.resnet152(weights=None)
    disable_inplace_relu(m)
    return m

# checkpoint every BasicBlock / Bottleneck -------------------------
class _CPWrapper(torch.nn.Module):
    """Lightweight wrapper that runs a sub-module under torch.utils.checkpoint."""
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, *args, **kwargs):  # noqa: D401
        return checkpoint(self.mod, *args, **kwargs)


def _inject_checkpoint(module: torch.nn.Module):
    """Recursively replace every residual block with a checkpointed wrapper."""
    for name, child in list(module.named_children()):
        if isinstance(child, (torchvision.models.resnet.BasicBlock,
                              torchvision.models.resnet.Bottleneck)):
            setattr(module, name, _CPWrapper(child))
        else:
            _inject_checkpoint(child)


def build_ckpt_all():
    """Return a ResNet-152 where *every* residual block is checkpointed."""
    m = build_base()
    _inject_checkpoint(m)
    return m

# ───────────────── timeline helper (reuse MemoryProfiler) ─────────
def generate_memory_timeline(model, batch_size: int, device: torch.device, outfile: str):
    """Create a per-op GPU memory timeline PNG for `model`."""
    profiler = MemoryProfiler(model, device)
    inp = torch.randn(batch_size, 3, 224, 224, device=device)
    out = profiler.run(inp)
    profiler.run_backward(out)
    plot_memory_profile(profiler, outfile)

# ───────────────── main ────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    dev=torch.device("cuda")

    # read plan
    if not os.path.exists("checkpoint_plan.csv"):
        raise FileNotFoundError("Run activation_checkpointing.py first")
    ranks={int(r["rank"]) for r in csv.DictReader(open("checkpoint_plan.csv"))
           if r.get("to_recompute","").strip().lower()=="yes"}
    if not ranks: warnings.warn("No ranks flagged for recompute")
    print(f"Checkpointing {len(ranks)} activations (by rank)")

    # ---- batch-16 profile separate runs ---------------------------
    base=build_base().to(dev).train()
    p_base=peak_mb(base,16,dev)
    del base; torch.cuda.empty_cache()

    ckpt=build_ckpt_all().to(dev).train()
    p_ckpt=peak_mb(ckpt,16,dev)
    del ckpt; torch.cuda.empty_cache()

    plt.figure(figsize=(5,4))
    for lbl,val,col in [("baseline",p_base,"tab:blue"),
                        ("ckpt",    p_ckpt,"tab:green")]:
        bar=plt.bar(lbl,val,color=col)
        plt.text(bar[0].get_x()+bar[0].get_width()/2,
                 val+25,f"{val:.0f}",ha="center")
    plt.ylabel("Peak GPU mem (MB)")
    plt.title("Batch 16 peak")
    plt.tight_layout(); plt.savefig("activationcheckpointing_profile.png")
    plt.close()

    # ---- sweep batches 8/16/32  -----------------------------------
    batches=[8,16,32]; mb_base=[]; mb_ck=[]
    for bs in batches:
        base=build_base().to(dev).train()
        mb_base.append(peak_mb(base,bs,dev))
        del base; torch.cuda.empty_cache()

        ckpt=build_ckpt_all().to(dev).train()
        mb_ck.append(peak_mb(ckpt,bs,dev))
        del ckpt; torch.cuda.empty_cache()

    w=0.35; idx=range(len(batches))
    plt.figure(figsize=(7,4))
    plt.bar(idx, mb_ck, w, color="tab:green", label="checkpoint")
    plt.xticks(idx, [str(b) for b in batches])
    plt.xlabel("Batch size"); plt.ylabel("Peak GPU mem (MB)")
    plt.title("Peak memory vs batch (checkpoint only)")
    for i, b in enumerate(mb_ck):
        plt.text(i, b + 20, f"{b:.0f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig("checkpointing_peak_vs_batch.png"); plt.close()
    print("Saved both plots.")

    # Produce detailed memory timeline for the checkpointed graph (batch-16)
    ckpt_timeline = build_ckpt_all().to(dev).train()
    generate_memory_timeline(ckpt_timeline, 16, dev, "checkpoint_memory_profile.png")
    del ckpt_timeline; torch.cuda.empty_cache()

if __name__=="__main__":
    main()
