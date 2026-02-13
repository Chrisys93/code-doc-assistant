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
MODEL TIER RESOLUTION
============================================================
Maps the high-level modelTier value to specific Ollama model
tags. This single value cascades through:
  - Model selection (which model Ollama pulls and serves)
  - Resource allocation (memory, GPU requests/limits)
  - Context window / timeout tuning

Usage in templates:
  {{ include "code-doc-assistant.ollamaModel" . }}
  {{ include "code-doc-assistant.ollamaResources" . | nindent 12 }}
============================================================
*/}}

{{/*
Resolve the Ollama model tag from the modelTier value.
  full        → mistral-nemo (12B, best comprehension + explanation)
  balanced    → qwen2.5-coder:7b (7B, best code understanding at mid-range)
  lightweight → phi3.5 (3.8B, edge/low-resource deployments)
*/}}
{{- define "code-doc-assistant.ollamaModel" -}}
{{- if eq .Values.modelTier "full" -}}
mistral-nemo
{{- else if eq .Values.modelTier "balanced" -}}
qwen2.5-coder:7b
{{- else if eq .Values.modelTier "lightweight" -}}
phi3.5
{{- else -}}
mistral-nemo
{{- end -}}
{{- end -}}

{{/*
Resolve resource requests/limits based on modelTier.
GPU is only requested for the full tier; balanced and lightweight
are designed to run on CPU-only nodes if needed.
*/}}
{{- define "code-doc-assistant.ollamaResources" -}}
{{- if eq .Values.modelTier "full" }}
requests:
  memory: "12Gi"
  cpu: "2"
  nvidia.com/gpu: "1"
limits:
  memory: "16Gi"
  cpu: "4"
  nvidia.com/gpu: "1"
{{- else if eq .Values.modelTier "balanced" }}
requests:
  memory: "8Gi"
  cpu: "2"
limits:
  memory: "10Gi"
  cpu: "4"
{{- else if eq .Values.modelTier "lightweight" }}
requests:
  memory: "4Gi"
  cpu: "1"
limits:
  memory: "6Gi"
  cpu: "2"
{{- else }}
requests:
  memory: "12Gi"
  cpu: "2"
limits:
  memory: "16Gi"
  cpu: "4"
{{- end }}
{{- end -}}

{{/*
Resolve context-window and inference tuning based on modelTier.
Larger models can handle more context but need longer timeouts.
*/}}
{{- define "code-doc-assistant.inferenceConfig" -}}
{{- if eq .Values.modelTier "full" }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- else if eq .Values.modelTier "balanced" }}
OLLAMA_NUM_CTX: "4096"
OLLAMA_TIMEOUT: "90"
{{- else if eq .Values.modelTier "lightweight" }}
OLLAMA_NUM_CTX: "2048"
OLLAMA_TIMEOUT: "60"
{{- else }}
OLLAMA_NUM_CTX: "8192"
OLLAMA_TIMEOUT: "120"
{{- end }}
{{- end -}}

{{/*
Ollama internal service hostname — used by the app to connect.
*/}}
{{- define "code-doc-assistant.ollamaHost" -}}
http://{{ include "code-doc-assistant.fullname" . }}-ollama:{{ .Values.ollama.service.port }}
{{- end -}}

{{/*
============================================================
EMBEDDING MODEL RESOLUTION
============================================================
The embedding model is a tightly-coupled, locality-sensitive
configuration — changing it requires full re-ingestion.
The vector dimension is derived automatically to ensure the
vector DB is always correctly configured.

Usage in templates:
  {{ include "code-doc-assistant.embeddingModel" . }}
  {{ include "code-doc-assistant.embeddingDimension" . }}
============================================================
*/}}

{{/*
Resolve the embedding model from the embeddingModel value.
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

{{/*
Derive the vector dimension from the embedding model choice.
This ensures the vector DB index is always configured with the
correct dimensionality — a mismatch would cause silent failures.
*/}}
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
CHUNKING STRATEGY RESOLUTION
============================================================
The chunking strategy is tied to the model tier:
  - Full/balanced tiers use AST-aware chunking (tree-sitter)
    which produces higher-quality semantic chunks but uses
    more CPU/memory during ingestion.
  - Lightweight tier defaults to text-based chunking
    (SentenceSplitter) for lower resource usage.

Both strategies are always available at runtime — this
configuration controls the *default* behaviour. The app
falls back to text chunking automatically if AST parsing
fails regardless of this setting.

Usage in templates:
  {{ include "code-doc-assistant.chunkingStrategy" . }}
============================================================
*/}}

{{/*
Resolve the default chunking strategy from the modelTier.
  full/balanced → ast (tree-sitter CodeSplitter)
  lightweight   → text (SentenceSplitter, lower resource usage)
*/}}
{{- define "code-doc-assistant.chunkingStrategy" -}}
{{- if eq .Values.modelTier "lightweight" -}}
text
{{- else -}}
ast
{{- end -}}
{{- end -}}
