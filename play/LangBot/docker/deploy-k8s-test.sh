#!/bin/bash
# Quick test script for LangBot Kubernetes deployment
# This script helps you test the Kubernetes deployment locally

set -e

echo "🚀 LangBot Kubernetes Deployment Test Script"
echo "=============================================="
echo ""

# Check for kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    echo "Visit: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

echo "✓ kubectl is installed"

# Check if kubectl can connect to a cluster
if ! kubectl cluster-info &> /dev/null; then
    echo ""
    echo "⚠️  No Kubernetes cluster found."
    echo ""
    echo "To test locally, you can use:"
    echo "  - kind: https://kind.sigs.k8s.io/"
    echo "  - minikube: https://minikube.sigs.k8s.io/"
    echo "  - k3s: https://k3s.io/"
    echo ""
    echo "Example with kind:"
    echo "  kind create cluster --name langbot-test"
    echo ""
    exit 1
fi

echo "✓ Connected to Kubernetes cluster"
kubectl cluster-info
echo ""

# Ask user to confirm
read -p "Do you want to deploy LangBot to this cluster? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "📦 Deploying LangBot..."
kubectl apply -f kubernetes.yaml

echo ""
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=langbot -n langbot --timeout=300s
kubectl wait --for=condition=ready pod -l app=langbot-plugin-runtime -n langbot --timeout=300s

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Deployment status:"
kubectl get all -n langbot

echo ""
echo "🌐 To access LangBot Web UI, run:"
echo "   kubectl port-forward -n langbot svc/langbot 5300:5300"
echo ""
echo "Then visit: http://localhost:5300"
echo ""
echo "📝 To view logs:"
echo "   kubectl logs -n langbot -l app=langbot -f"
echo ""
echo "🗑️  To uninstall:"
echo "   kubectl delete namespace langbot"
echo ""
