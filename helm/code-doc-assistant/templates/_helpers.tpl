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
{{- else if eq .Values.modelTier "balanced" -}}
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
{{- else if eq .Values.modelTier "balanced" -}}
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
*/}}
{{- define "code-doc-assistant.inferenceConfig" -}}
{{- if eq .Values.modelTier "full" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- else if eq .Values.modelTier "balanced" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
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
