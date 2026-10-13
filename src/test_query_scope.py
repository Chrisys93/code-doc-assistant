import os, sys, types
os.environ.update(INFERENCE_BACKEND="llamacpp", LLAMACPP_CTX="4096", LLAMACPP_HOST="http://127.0.0.1:9")
sys.path.insert(0, ".")
for _mod, _cls in (("langchain_ollama", "ChatOllama"), ("langchain_openai", "ChatOpenAI")):
    try: __import__(_mod)
    except ImportError:
        _m = types.ModuleType(_mod); setattr(_m, _cls, object); sys.modules[_mod] = _m
try:
    import mlflow  # noqa: F401
except ImportError:      # the tests need no tracking server; a laptop without mlflow can still run them
    from unittest import mock
    sys.modules["mlflow"] = mock.MagicMock()
import agent_graph as ag
fails = []
def check(n, c, x=""):
    print(("PASS " if c else "FAIL ") + n + (f" [{x}]" if (x and not c) else ""))
    if not c: fails.append(n)

CDA, ICA, CH3 = "repo_code_doc_assistant_dev_e308113f", "repo_icarusreposend_f5929c11", "repo_chapter3_02bb4f1d"
ALL = [CDA, ICA, CH3]
def scope(q): return ag._scope_for_query(q, ALL)[0]

check("question names chapter3 -> only chapter3", scope("Which scripts in this repository (chapter3) duplicate each other?") == [CH3])
check("question names IcarusRepoSEND -> only icarus", scope("Associate IcarusRepoSEND with network engineering principles.") == [ICA])
check("names code-doc-assistant (branch suffix in slug) -> only cda", scope("What should the next steps be for the code-doc-assistant repository?") == [CDA])
check("names two repos -> both", set(scope("How are the hashes in IcarusRepoSEND related to the embedding model in code-doc-assistant?")) == {ICA, CDA})
check("names none -> all", scope("What does config.py configure, and which settings matter most?") == ALL)
check("case and punctuation are ignored", scope("about ICARUS-repo_SEND please") == [ICA])
check("branch token alone does not match", scope("what does the dev branch do?") == ALL)
ag.QUERY_REPO_SCOPE = False
check("QUERY_REPO_SCOPE=off -> all", scope("chapter3 only please") == ALL)
ag.QUERY_REPO_SCOPE = True
check("single active collection untouched", ag._scope_for_query("chapter3", [CH3]) == ([CH3], []))

# the planner's own collections are overridden by the scope, and the scoped set reaches the tool call
args = {"query": "q", "collections": ALL}
dropped = ag._scope_vector_search_args(args, [CH3])
check("planner plan is narrowed to the scoped collection", args["collections"] == [CH3], str(args))
check("narrowing is reported", any("in scope" in d for d in dropped), str(dropped))

# chunks carry their repo into the context, so two files with the same name (README.md, config.py) can be told apart
from agent_state import Chunk
check("repo label from collection name", ag._repo_label(ICA) == "icarusreposend" and ag._repo_label(CDA) == "code_doc_assistant_dev")
c = Chunk(content="x", source_file="config.py", repo=ICA)
check("Chunk keeps its repo", c.repo == ICA)
check("Chunk repo defaults to empty", Chunk(content="x", source_file="a.py").repo == "")
print("\nALL OK" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
