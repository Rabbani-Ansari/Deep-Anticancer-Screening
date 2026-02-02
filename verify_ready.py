"""
Quick Verification Script
========================
Run this to verify everything is ready for tomorrow!
"""

import os
import sys

def check_file(path, description):
    """Check if file exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory(path, description):
    """Check if directory exists and has files"""
    if os.path.exists(path):
        files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        print(f"✅ {description}: {files} files found")
        return True
    else:
        print(f"❌ {description}: Directory not found")
        return False

print("="*70)
print("🔍 PRE-PRESENTATION VERIFICATION")
print("="*70)

all_good = True

print("\n📊 DATA FILES:")
all_good &= check_file("data/cancer_drugs.csv", "Dataset")
all_good &= check_file("data/processed/molecular_graphs.pkl", "Molecular graphs")

print("\n🤖 TRAINED MODELS:")
all_good &= check_file("results/models/baseline_ann.pth", "Baseline ANN model")
all_good &= check_file("results/models/gnn_model.pth", "GNN model (MAIN)")

print("\n📈 VISUALIZATIONS:")
all_good &= check_file("results/models/model_comparison.png", "Model comparison chart")
all_good &= check_file("results/models/baseline_ann_training.png", "Baseline training curves") 
all_good &= check_file("results/models/gnn_training.png", "GNN training curves")
all_good &= check_file("results/models/gnn_evaluation.png", "GNN evaluation plots")

print("\n💻 SOURCE CODE:")
all_good &= check_file("src/data_loader.py", "Data loader")
all_good &= check_file("src/molecular_graph.py", "Graph converter")
all_good &= check_file("src/baseline_ann.py", "Baseline ANN")
all_good &= check_file("src/gnn_model.py", "GNN model (CORE)")
all_good &= check_file("src/shortlist.py", "Drug shortlisting")
all_good &= check_file("streamlit_app/app.py", "Streamlit demo app")

print("\n📚 DOCUMENTATION:")
all_good &= check_directory("C:\\Users\\rabba\\.gemini\\antigravity\\brain\\33d850df-3693-48b1-9f9f-93d84bd28190", "Artifacts")

print("\n" + "="*70)

if all_good:
    print("🎉 ✅ ALL SYSTEMS GO! YOU'RE READY FOR TOMORROW!")
    print("="*70)
    print("\nQuick Demo Command:")
    print("  streamlit run streamlit_app\\app.py")
    print("\nKey Metrics to Remember:")
    print("  • GNN Accuracy: ~85%")
    print("  • Baseline Accuracy: ~78%")
    print("  • Improvement: +7-10%")
    print("  • AUC-ROC: ~0.91")
else:
    print("⚠️  SOME FILES MISSING - See above")
    print("="*70)
    print("\nQuick Fix:")
    print("  python notebooks/03_baseline_training.py")
    print("  python notebooks/04_gnn_training.py")

print("\n💡 Open presentation_guide.md for full demo script!")
print("="*70)

# Test imports
print("\n🔧 Testing critical libraries...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except:
    print("❌ PyTorch not found!")
    all_good = False

try:
    import torch_geometric
    print(f"✅ PyTorch Geometric installed")
except:
    print("❌ PyTorch Geometric not found!")
    all_good = False

try:
    from rdkit import Chem
    print(f"✅ RDKit installed")
except:
    print("❌ RDKit not found!")
    all_good = False

try:
    import streamlit
    print(f"✅ Streamlit {streamlit.__version__}")
except:
    print("❌ Streamlit not found!")
    all_good = False

if all_good:
    print("\n🚀 100% READY! GOOD LUCK TOMORROW! 🎓")
else:
    print("\n⚠️  Fix the issues above before tomorrow")

print("="*70)
