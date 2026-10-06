{{/*
============================================================
Code Documentation Assistant — Template Helpers
============================================================
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "code-doc-assistant.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Fully qualified app name.
*/}}
{{- define "code-doc-assistant.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "code-doc-assistant.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "code-doc-assistant.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
============================================================
MODEL TIER + QUANTISATION RESOLUTION
============================================================
modelTier  → capability selector (which base model)
quantisation → resource selector (precision / memory footprint)
_helpers.tpl composes them into the final Ollama model tag.

Tier → base model string:
  full        → mistral-nemo:12b-instruct
  balanced    → deepseek-coder-v2:16b-lite-instruct
  heavy       → deepseek-coder-v2:16b-lite-instruct (same model as balanced)
  lightweight → phi3.5               (already Q4; no suffix appended)
  minimal     → qwen2.5-coder:3b-instruct (already Q4; no suffix appended)

Quantisation suffix (appended for full + balanced only):
  q4_K_M  → -q4_K_M
  q8_0    → -q8_0
  fp16    → (no suffix)

Example compositions:
  balanced + q4_K_M → deepseek-coder-v2:16b-lite-instruct-q4_K_M
  full     + q8_0   → mistral-nemo:12b-instruct-q8_0
  full     + fp16   → mistral-nemo:12b-instruct
  minimal  + q4_K_M → qwen2.5-coder:3b-instruct   (suffix suppressed)
============================================================
*/}}

{{/*
Resolve base model name from modelTier (without quant suffix).
*/}}
{{- define "code-doc-assistant.baseModel" -}}
{{- if eq .Values.modelTier "full" -}}
mistral-nemo:12b-instruct
{{- else if or (eq .Values.modelTier "balanced") (eq .Values.modelTier "heavy") -}}
deepseek-coder-v2:16b-lite-instruct
{{- else if eq .Values.modelTier "lightweight" -}}
phi3.5
{{- else if eq .Values.modelTier "minimal" -}}
qwen2.5-coder:3b-instruct
{{- else -}}
mistral-nemo:12b-instruct
{{- end -}}
{{- end -}}

{{/*
Compose final Ollama model tag from modelTier + quantisation.
Suffix is suppressed for lightweight and minimal (already quantised by default).
*/}}
{{- define "code-doc-assistant.ollamaModel" -}}
{{- $base := include "code-doc-assistant.baseModel" . -}}
{{- $quant := .Values.quantisation | default "q4_K_M" -}}
{{- if or (eq .Values.modelTier "lightweight") (eq .Values.modelTier "minimal") -}}
{{- $base -}}
{{- else if eq $quant "fp16" -}}
{{- $base -}}
{{- else -}}
{{- printf "%s-%s" $base $quant -}}
{{- end -}}
{{- end -}}

{{/*
Resolve Ollama resource requests/limits.
local:   returns empty (no requests/limits — avoid scheduling friction on dev machines)
cluster: enforced limits scaled by modelTier + quantisation.
         GPU node affinity only for full tier.
*/}}
{{- define "code-doc-assistant.ollamaResources" -}}
{{- if eq (.Values.deploymentTarget | default "local") "local" -}}
{}
{{- else if eq .Values.modelTier "full" -}}
{{- $quant := .Values.quantisation | default "q4_K_M" -}}
{{- if eq $quant "fp16" }}
requests:
  memory: "14Gi"
  cpu: "2"
  nvidia.com/gpu: "1"
limits:
  memory: "18Gi"
  cpu: "4"
  nvidia.com/gpu: "1"
{{- else if eq $quant "q8_0" }}
requests:
  memory: "10Gi"
  cpu: "2"
  nvidia.com/gpu: "1"
limits:
  memory: "14Gi"
  cpu: "4"
  nvidia.com/gpu: "1"
{{- else }}
requests:
  memory: "8Gi"
  cpu: "2"
  nvidia.com/gpu: "1"
limits:
  memory: "12Gi"
  cpu: "4"
  nvidia.com/gpu: "1"
{{- end }}
{{- else if or (eq .Values.modelTier "balanced") (eq .Values.modelTier "heavy") -}}
{{- $quant := .Values.quantisation | default "q4_K_M" -}}
{{- if eq $quant "fp16" }}
requests:
  memory: "12Gi"
  cpu: "2"
limits:
  memory: "16Gi"
  cpu: "4"
{{- else if eq $quant "q8_0" }}
requests:
  memory: "8Gi"
  cpu: "2"
limits:
  memory: "10Gi"
  cpu: "4"
{{- else }}
requests:
  memory: "6Gi"
  cpu: "2"
limits:
  memory: "8Gi"
  cpu: "4"
{{- end }}
{{- else if eq .Values.modelTier "lightweight" }}
requests:
  memory: "4Gi"
  cpu: "1"
limits:
  memory: "6Gi"
  cpu: "2"
{{- else if eq .Values.modelTier "minimal" }}
requests:
  memory: "2Gi"
  cpu: "1"
limits:
  memory: "3Gi"
  cpu: "2"
{{- else }}
requests:
  memory: "8Gi"
  cpu: "2"
limits:
  memory: "12Gi"
  cpu: "4"
{{- end }}
{{- end -}}

{{/*
App container resource profile.
local:   no requests/limits
cluster: modest fixed limits (app is lightweight relative to Ollama/ChromaDB)
*/}}
{{- define "code-doc-assistant.appResources" -}}
{{- if eq (.Values.deploymentTarget | default "local") "local" -}}
{}
{{- else }}
requests:
  memory: "512Mi"
  cpu: "250m"
limits:
  memory: "1Gi"
  cpu: "1"
{{- end }}
{{- end -}}

{{/*
Resolve context-window and timeout config from modelTier.
Minimal tier gets a tighter context window to reduce memory pressure.
llamacpp backend overrides context to match llama-server's --ctx-size.
*/}}
{{- define "code-doc-assistant.inferenceConfig" -}}
{{- if eq .Values.inferenceBackend "llamacpp" }}
OLLAMA_NUM_CTX: "{{ include "code-doc-assistant.llamacppCtx" . }}"
OLLAMA_TIMEOUT: "{{ if eq .Values.modelTier "heavy" }}180{{ else }}60{{ end }}"
{{- else if eq .Values.modelTier "full" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- else if eq .Values.modelTier "balanced" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- else if eq .Values.modelTier "heavy" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "180"
{{- else if eq .Values.modelTier "lightweight" }}
OLLAMA_NUM_CTX: "4096"
OLLAMA_TIMEOUT: "60"
{{- else if eq .Values.modelTier "minimal" }}
OLLAMA_NUM_CTX: "2048"
OLLAMA_TIMEOUT: "45"
{{- else }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- end }}
{{- end -}}

{{/*
Ollama internal service hostname.
*/}}
{{- define "code-doc-assistant.ollamaHost" -}}
http://{{ include "code-doc-assistant.fullname" . }}-ollama:{{ .Values.ollama.service.port }}
{{- end -}}

{{/*
============================================================
EMBEDDING MODEL RESOLUTION
============================================================
*/}}

{{- define "code-doc-assistant.embeddingModel" -}}
{{- if eq .Values.embeddingModel "default" -}}
nomic-embed-text
{{- else if eq .Values.embeddingModel "lightweight" -}}
all-minilm
{{- else if eq .Values.embeddingModel "rich" -}}
mxbai-embed-large
{{- else -}}
nomic-embed-text
{{- end -}}
{{- end -}}

{{- define "code-doc-assistant.embeddingDimension" -}}
{{- if eq .Values.embeddingModel "default" -}}
768
{{- else if eq .Values.embeddingModel "lightweight" -}}
384
{{- else if eq .Values.embeddingModel "rich" -}}
1024
{{- else -}}
768
{{- end -}}
{{- end -}}

{{/*
============================================================
CHROMADB HNSW CONFIGURATION
============================================================
Emits the ChromaDB collection metadata string for vector_store.py.
Parameters sourced from values.yaml vectordb.hnsw.*

deploymentTarget=local  → searchEf overridden to 20
                          (faster queries, less accurate — fine for single-user dev)
deploymentTarget=cluster → full searchEf from values.yaml
                           (better recall under concurrent load)

The resulting string is passed as CHROMA_HNSW_CONFIG env var and consumed
by VectorStoreImpl.get_or_create_collection() in vector_store.py.
*/}}
{{- define "code-doc-assistant.chromaHnswConfig" -}}
{{- $space          := .Values.vectordb.hnsw.space          | default "cosine" -}}
{{- $M              := .Values.vectordb.hnsw.M              | default 16 -}}
{{- $constructionEf := .Values.vectordb.hnsw.constructionEf | default 100 -}}
{{- $searchEf       := .Values.vectordb.hnsw.searchEf       | default 50 -}}
{{- if eq (.Values.deploymentTarget | default "local") "local" -}}
  {{- $searchEf = 20 -}}
{{- end -}}
hnsw:space={{ $space }},hnsw:M={{ $M }},hnsw:construction_ef={{ $constructionEf }},hnsw:search_ef={{ $searchEf }}
{{- end -}}

{{/*
============================================================
DEV BRANCH ADDITIONS
============================================================
*/}}

{{/*
MLflow tracking server internal hostname.
*/}}
{{- define "code-doc-assistant.mlflowHost" -}}
http://{{ include "code-doc-assistant.fullname" . }}-mlflow:{{ .Values.mlflow.service.port }}
{{- end -}}

{{/*
vLLM inference server internal hostname.
*/}}
{{- define "code-doc-assistant.vllmHost" -}}
http://{{ include "code-doc-assistant.fullname" . }}-vllm:{{ .Values.vllm.service.port }}
{{- end -}}

{{/*
llama-server (llama.cpp) internal hostname.
Only active when inferenceBackend=llamacpp.
*/}}
{{- define "code-doc-assistant.llamacppHost" -}}
http://{{ include "code-doc-assistant.fullname" . }}-llamacpp:{{ .Values.llamacpp.service.port }}
{{- end -}}

{{/*
============================================================
CHUNKING STRATEGY
============================================================
Default chunking strategy from the modelTier (CHUNKING_STRATEGY env).
  lightweight   → text (SentenceSplitter, lower resource usage)
  everything else → ast (tree-sitter CodeSplitter)
The app falls back to text chunking automatically if AST parsing fails.
*/}}
{{- define "code-doc-assistant.chunkingStrategy" -}}
{{- if eq .Values.modelTier "lightweight" -}}
text
{{- else -}}
ast
{{- end -}}
{{- end -}}

{{/*
============================================================
BACKEND SWITCHES
============================================================
Return "true" or "" (so they work directly in `if include ...`).
A backend is deployed when it is the selected inferenceBackend OR its own
`enabled` flag is set (side-by-side deployments, inferenceBackend=auto).
*/}}
{{- define "code-doc-assistant.llamacppEnabled" -}}
{{- if or (eq .Values.inferenceBackend "llamacpp") .Values.llamacpp.enabled -}}true{{- end -}}
{{- end -}}

{{- define "code-doc-assistant.vllmEnabled" -}}
{{- if or (eq .Values.inferenceBackend "vllm") .Values.vllm.enabled -}}true{{- end -}}
{{- end -}}

{{/*
Should the Ollama pod pull the chat (LLM) model? The embedding model is always pulled.
Default: only when Ollama is, or may be, the answering backend (ollama / auto), so that
a llamacpp or vllm deployment does not download ~10GB it will never use.
ollama.pullLlm = "true" / "false" forces it.
*/}}
{{- define "code-doc-assistant.ollamaPullLlm" -}}
{{- $force := toString .Values.ollama.pullLlm -}}
{{- if eq $force "true" -}}
true
{{- else if eq $force "false" -}}
{{- else if or (eq .Values.inferenceBackend "ollama") (eq .Values.inferenceBackend "auto") -}}
true
{{- end -}}
{{- end -}}

{{/*
============================================================
LLAMA.CPP (llama-server) TIER DEFAULTS
============================================================
Mirror docker-compose.dev.yml's llamacpp service. Explicit llamacpp.* values win.

  tier      GGUF file                                     ctx    GPU layers
  heavy     deepseek-coder-v2-lite-instruct-q4_k_m.gguf   4096   24
  full      mistral-nemo-instruct-2407-q4_k_m.gguf        8192   999
  minimal   qwen2.5-coder-3b-instruct-q4_k_m.gguf         2048   999
  other     (no verified default GGUF: llamacpp.modelFile is required)
*/}}
{{- define "code-doc-assistant.llamacppModelFile" -}}
{{- if .Values.llamacpp.modelFile -}}
{{- .Values.llamacpp.modelFile -}}
{{- else if eq .Values.modelTier "heavy" -}}
deepseek-coder-v2-lite-instruct-q4_k_m.gguf
{{- else if eq .Values.modelTier "full" -}}
mistral-nemo-instruct-2407-q4_k_m.gguf
{{- else if eq .Values.modelTier "minimal" -}}
qwen2.5-coder-3b-instruct-q4_k_m.gguf
{{- else -}}
{{- fail (printf "llamacpp: modelTier %q has no default GGUF - set llamacpp.modelFile" .Values.modelTier) -}}
{{- end -}}
{{- end -}}

{{/*
Name llama-server reports (-a), which the app reads back from /v1/models.
*/}}
{{- define "code-doc-assistant.llamacppServedName" -}}
{{- if .Values.llamacpp.servedModelName -}}
{{- .Values.llamacpp.servedModelName -}}
{{- else -}}
{{- include "code-doc-assistant.llamacppModelFile" . | trimSuffix ".gguf" -}}
{{- end -}}
{{- end -}}

{{- define "code-doc-assistant.llamacppCtx" -}}
{{- $set := toString .Values.llamacpp.contextSize -}}
{{- if ne $set "" -}}
{{- $set -}}
{{- else if eq .Values.modelTier "heavy" -}}
4096
{{- else if eq .Values.modelTier "minimal" -}}
2048
{{- else if eq .Values.modelTier "lightweight" -}}
4096
{{- else -}}
8192
{{- end -}}
{{- end -}}

{{/*
--n-gpu-layers. Note 0 is a legitimate value (CPU only), so "unset" is the empty string,
not falsy — never use `default` on this.
*/}}
{{- define "code-doc-assistant.llamacppGpuLayers" -}}
{{- $set := toString .Values.llamacpp.gpuLayers -}}
{{- if ne $set "" -}}
{{- $set -}}
{{- else if eq .Values.modelTier "heavy" -}}
24
{{- else -}}
999
{{- end -}}
{{- end -}}

{{/*
Download URL for the optional model-fetch init container. An explicit llamacpp.download.url
wins; the tier URLs are only used when the tier's default GGUF is what is being served.
(bartowski's Q4_K_M quants — the same ones docker-compose.dev.yml documents.)
*/}}
{{- define "code-doc-assistant.llamacppDownloadUrl" -}}
{{- if .Values.llamacpp.download.url -}}
{{- .Values.llamacpp.download.url -}}
{{- else if .Values.llamacpp.modelFile -}}
{{- else if eq .Values.modelTier "heavy" -}}
https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf
{{- else if eq .Values.modelTier "full" -}}
https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf
{{- end -}}
{{- end -}}

{{/*
============================================================
MLFLOW --allowed-hosts
============================================================
MLflow answers 403 "Invalid Host header" to any Host it was not told about, which
silently turns tracking off. Allow: the in-cluster Service names (short, .<ns>,
.<ns>.svc, .<ns>.svc.cluster.local — each with and without the port), localhost
(kubectl port-forward), plus mlflow.allowedHosts. Comma-separated.
*/}}
{{- define "code-doc-assistant.mlflowAllowedHosts" -}}
{{- $svc := printf "%s-mlflow" (include "code-doc-assistant.fullname" .) -}}
{{- $port := toString .Values.mlflow.service.port -}}
{{- $ns := .Release.Namespace -}}
{{- $hosts := list -}}
{{- range $h := list $svc (printf "%s.%s" $svc $ns) (printf "%s.%s.svc" $svc $ns) (printf "%s.%s.svc.cluster.local" $svc $ns) "localhost" "127.0.0.1" -}}
{{- $hosts = append $hosts $h -}}
{{- $hosts = append $hosts (printf "%s:%s" $h $port) -}}
{{- end -}}
{{- range .Values.mlflow.allowedHosts -}}
{{- $hosts = append $hosts . -}}
{{- end -}}
{{- join "," $hosts -}}
{{- end -}}

{{/*
MLflow resource profile. The current MLflow image needs well over 512Mi just to start
(a 512Mi limit gets the pod OOMKilled in a crash loop), so:
local:   no requests/limits
cluster: 512Mi requested, 2Gi limit
Override completely with mlflow.resources.
*/}}
{{- define "code-doc-assistant.mlflowResources" -}}
{{- if eq (.Values.deploymentTarget | default "local") "local" -}}
{}
{{- else }}
requests:
  memory: "512Mi"
  cpu: "100m"
limits:
  memory: "2Gi"
  cpu: "1"
{{- end }}
{{- end -}}
