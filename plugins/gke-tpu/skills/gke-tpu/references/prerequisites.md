# Prerequisites

## PATH setup (needed in every shell/command)

```bash
export PATH="/opt/homebrew/bin:/opt/homebrew/share/google-cloud-sdk/bin:/usr/bin:$PATH"
```

## Install

```bash
# Google Cloud SDK
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
export https_proxy=http://127.0.0.1:7890
```
