import os, sys
os.environ.update(INFERENCE_BACKEND="llamacpp", LLAMACPP_CTX="4096", MAX_CONTEXT_TOKENS="8000", LLAMACPP_HOST="http://127.0.0.1:9")
sys.path.insert(0, ".")
import types
for _mod,_cls in (("langchain_ollama","ChatOllama"),("langchain_openai","ChatOpenAI")):
    try: __import__(_mod)
    except ImportError:
        _m=types.ModuleType(_mod); setattr(_m,_cls,object); sys.modules[_mod]=_m
import agent_graph as ag
from agent_state import Chunk
import inspect
fails=[]
def check(n,c,x=""):
    print(("PASS " if c else "FAIL ")+n+(f" [{x}]" if (x and not c) else "")); 
    if not c: fails.append(n)

st={"active_backend":"llamacpp","active_model_tier":"heavy","retrieved_chunks":[],"execution_trace":[]}
# /props unreachable (port 9): must use the configured window, NOT the tier table's 8192
ag._llamacpp_props = lambda: None
check("props unreachable -> LLAMACPP_CTX (4096), source=config", ag._effective_ctx_info(st)==(4096,"config"), str(ag._effective_ctx_info(st)))
ag._llamacpp_props = lambda: {"default_generation_settings":{"n_ctx":4096}}
check("props ok + config -> min, source=server+config", ag._effective_ctx_info(st)==(4096,"server+config"))
ag._llamacpp_props = lambda: {"default_generation_settings":{"n_ctx":2048}}
check("server smaller than config -> server wins", ag._effective_ctx_info(st)[0]==2048)
# neither known -> tier table but flagged
ag.LLAMACPP_CTX=0; ag._llamacpp_props=lambda: None
w,src=ag._effective_ctx_info(st); check("nothing known -> tier table, flagged UNVERIFIED", (w,src)==(8192,"tier-table-UNVERIFIED"), str((w,src)))
# non-llamacpp backend unaffected
st2=dict(st, active_backend="ollama"); check("ollama backend unchanged", ag._effective_ctx_info(st2)==(8192,"tier-table"))

# context assembly: oversize retrieval is trimmed to fit 4096 with room to spare
ag.LLAMACPP_CTX=4096; ag._llamacpp_props=lambda: None
import inspect
sig=inspect.signature(Chunk); 
def mk(i, n=2500):
    kw={}
    for name in sig.parameters:
        kw[name]={"content":"def f(x):\n    return x+%d\n"%i*(n//30),"source_file":f"f{i}.py","start_line":1,"end_line":50,"confidence":0.7}.get(name, None if name not in ("metadata",) else {})
    return Chunk(**kw)
chunks=[mk(i) for i in range(10)]
st["retrieved_chunks"]=chunks
out=ag.node_context_assembly(st)
tr=out["execution_trace"][-1]
ctx_chars=len(out["final_context"])
est_real_tokens=ctx_chars/3.0
check("context fits: est tokens + 700 overhead < 4096", est_real_tokens+700<4096, f"{est_real_tokens:.0f} tok est, {ctx_chars} chars")
check("trace reports window 4096 and its source", "model window 4096 [config]" in tr["detail"], tr["detail"])
ag.LLAMACPP_CTX=0
out=ag.node_context_assembly(st); tr=out["execution_trace"][-1]
check("unverified window -> warn status in the trace", tr["status"]=="warn" and "WARNING: llama-server's context window could not be read" in tr["detail"], str(tr))
print("\nALL OK" if not fails else f"\nFAILED: {fails}"); sys.exit(1 if fails else 0)
