from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import base64
from io import BytesIO

# Get absolute path to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add parent directory to path to import src
sys.path.append(PROJECT_ROOT)

from src.shortlist import DrugShortlister
from src.molecular_graph import MolecularGraphConverter
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

app = FastAPI(title="OncoScreen AI API", description="GNN-based Drug Discovery API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
shortlister = None
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "models", "gnn_model.pth")

class MoleculeInput(BaseModel):
    smiles: str

class BatchInput(BaseModel):
    smiles_list: list[str]

@app.on_event("startup")
async def startup_event():
    global shortlister
    if os.path.exists(MODEL_PATH):
        print(f"Loading GNN model from {MODEL_PATH}...")
        shortlister = DrugShortlister(MODEL_PATH)
    else:
        print("⚠️ Warning: Model file not found. API will fail on prediction.")

@app.get("/")
def read_root():
    return {"status": "online", "model": "MolecularGNN", "version": "1.0.0"}

@app.post("/predict")
def predict_molecule(data: MoleculeInput):
    if not shortlister:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Pre-validate molecule
        mol = Chem.MolFromSmiles(data.smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")

        prob, pred_class = shortlister.predict_single(data.smiles)
        
        # Calculate Scientific Descriptors (The "Research" part)
        descriptors = {
            "MolecularWeight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "H_Donors": Descriptors.NumHDonors(mol),
            "H_Acceptors": Descriptors.NumHAcceptors(mol),
            "RotatableBonds": Descriptors.NumRotatableBonds(mol),
            "RingCount": Descriptors.RingCount(mol)
        }
        
        # Lipinski Rule of 5 Check
        lipinski_violations = 0
        if descriptors["MolecularWeight"] > 500: lipinski_violations += 1
        if descriptors["LogP"] > 5: lipinski_violations += 1
        if descriptors["H_Donors"] > 5: lipinski_violations += 1
        if descriptors["H_Acceptors"] > 10: lipinski_violations += 1
        
        druglikeness = "High" if lipinski_violations == 0 else ("Moderate" if lipinski_violations == 1 else "Low")

        # Determine confidence and simple explanation
        confidence = prob if prob > 0.5 else 1 - prob
        prediction_text = "Active Anticancer Agent" if prob > 0.5 else "Inactive / Low Potency"
        
        # Generate image
        img = Draw.MolToImage(mol, size=(600, 600))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "smiles": data.smiles,
            "probability": float(prob),
            "prediction_class": int(pred_class),
            "prediction_text": prediction_text,
            "confidence": float(confidence),
            "image": img_base64,
            "properties": descriptors,
            "analysis": {
                "lipinski_violations": lipinski_violations,
                "druglikeness": druglikeness
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/shortlist")
def shortlist_molecules(data: BatchInput):
    if not shortlister:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results_df = shortlister.shortlist_drugs(data.smiles_list, top_k=len(data.smiles_list))
        # Convert DataFrame to list of dictionaries for JSON serialization
        results = results_df.to_dict(orient='records')
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
