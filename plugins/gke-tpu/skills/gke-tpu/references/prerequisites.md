# Prerequisites

## Install

```bash
# Google Cloud SDK (see https://cloud.google.com/sdk/docs/install for your platform)
# macOS (Homebrew):
brew install --cask google-cloud-sdk

# kubectl + auth plugin
gcloud components install kubectl gke-gcloud-auth-plugin beta --quiet

# Auth — use project from gke.toml
gcloud auth login
gcloud config set project <gke.project>
gcloud auth application-default login
```

## Connect to cluster

```bash
gcloud container clusters get-credentials <gke.cluster> --zone=<gke.zone> --project=<gke.project>
```

## Proxy (if needed)

If gcloud/kubectl timeout, set proxy:

```bash
export https_proxy=<your-proxy-url>
```
