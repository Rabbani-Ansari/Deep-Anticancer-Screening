from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import re
import base64
from io import BytesIO
import urllib.request
import urllib.parse
import json
import time

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
PUBCHEM_CACHE: dict[tuple[str, str], dict | None] = {}

class MoleculeInput(BaseModel):
    smiles: str

class BatchInput(BaseModel):
    smiles_list: list[str]

class ResolveInput(BaseModel):
    query: str

# ──────────────────────────────────────────────
# Smart Input Resolution (SMILES / Name / Formula)
# ──────────────────────────────────────────────

def _is_smiles(text: str) -> bool:
    """Heuristic check: does the string look like a SMILES?"""
    mol = Chem.MolFromSmiles(text)
    return mol is not None

def _is_molecular_formula(text: str) -> bool:
    """Check if text looks like a molecular formula (e.g. C9H8O4, H2O, C6H12O6)."""
    return bool(re.fullmatch(r'([A-Z][a-z]?\d*)+', text.strip()))

def _pubchem_lookup(identifier: str, namespace: str = "name") -> dict | None:
    """
    Query PubChem PUG-REST to resolve a compound.
    namespace: 'name', 'formula', or 'smiles'
    Returns dict with cid, iupac_name, molecular_formula, canonical_smiles or None.
    """
    def _extract_smiles(prop: dict) -> str | None:
        """Extract SMILES from PubChem property dict (field name varies)."""
        for key in ("CanonicalSMILES", "ConnectivitySMILES", "SMILES", "IsomericSMILES"):
            if prop.get(key):
                return prop[key]
        return None

    cache_key = (namespace, identifier.lower().strip())
    if cache_key in PUBCHEM_CACHE:
        return PUBCHEM_CACHE[cache_key]

    def _request_json(url: str, timeout: int = 12) -> dict:
        last_error = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "OncoScreenAI/1.0 (+https://localhost)"
                    }
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except Exception as exc:
                last_error = exc
                # Exponential backoff for transient PubChem failures.
                time.sleep(0.35 * (2 ** attempt))
        raise last_error

    try:
        base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
        encoded = urllib.parse.quote(identifier, safe="")

        if namespace == "formula":
            # Formula search returns a list – pick the first CID
            url = f"{base}/fastformula/{encoded}/cids/JSON"
            data = _request_json(url)
            cid = data.get("IdentifierList", {}).get("CID", [None])[0]
            if not cid:
                return None
        else:
            # Name search – go straight to property lookup
            url = f"{base}/{namespace}/{encoded}/property/CanonicalSMILES,MolecularFormula,IUPACName/JSON"
            data = _request_json(url)
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if not properties:
                return None
            prop = properties[0]
            result = {
                "cid": prop.get("CID"),
                "canonical_smiles": _extract_smiles(prop),
                "molecular_formula": prop.get("MolecularFormula"),
                "iupac_name": prop.get("IUPACName", identifier),
            }
            PUBCHEM_CACHE[cache_key] = result
            return result

        # For formula path, fetch properties using the CID
        url2 = f"{base}/cid/{cid}/property/CanonicalSMILES,MolecularFormula,IUPACName/JSON"
        data2 = _request_json(url2)
        properties = data2.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return None
        prop = properties[0]
        result = {
            "cid": prop.get("CID"),
            "canonical_smiles": _extract_smiles(prop),
            "molecular_formula": prop.get("MolecularFormula"),
            "iupac_name": prop.get("IUPACName", identifier),
        }
        PUBCHEM_CACHE[cache_key] = result
        return result
    except Exception as e:
        print(f"PubChem lookup failed for '{identifier}' ({namespace}): {e}")
        return None

def resolve_input_to_smiles(query: str) -> dict:
    """
    Accepts a SMILES, chemical name, or molecular formula and returns
    a dict with resolved_smiles, input_type, and metadata.
    """
    query = query.strip()
    if not query:
        raise ValueError("Empty input")

    # 1) Try as SMILES first
    if _is_smiles(query):
        return {
            "resolved_smiles": query,
            "input_type": "smiles",
            "original_query": query,
            "compound_name": None,
            "molecular_formula": None,
        }

    # 2) Try as molecular formula
    if _is_molecular_formula(query):
        result = _pubchem_lookup(query, namespace="formula")
        if result and result.get("canonical_smiles"):
            return {
                "resolved_smiles": result["canonical_smiles"],
                "input_type": "formula",
                "original_query": query,
                "compound_name": result.get("iupac_name"),
                "molecular_formula": result.get("molecular_formula"),
            }

    # 3) Try as chemical name (e.g., "aspirin", "paclitaxel")
    result = _pubchem_lookup(query, namespace="name")
    if result and result.get("canonical_smiles"):
        return {
            "resolved_smiles": result["canonical_smiles"],
            "input_type": "name",
            "original_query": query,
            "compound_name": result.get("iupac_name"),
            "molecular_formula": result.get("molecular_formula"),
        }

    raise ValueError(
        f"Could not resolve '{query}'. Please enter a valid SMILES string, "
        f"chemical name (e.g. aspirin), or molecular formula (e.g. C9H8O4)."
    )

@app.on_event("startup")
async def startup_event():
    global shortlister
    if os.path.exists(MODEL_PATH):
        print(f"Loading GNN model from {MODEL_PATH}...")
        shortlister = DrugShortlister(MODEL_PATH)
    else:
        print("[WARNING] Model file not found. API will fail on prediction.")

@app.get("/")
def read_root():
    return {"status": "online", "model": "MolecularGNN", "version": "1.0.0"}

@app.post("/resolve")
def resolve_compound(data: ResolveInput):
    """Resolve a chemical name, formula, or SMILES to a canonical SMILES."""
    try:
        result = resolve_input_to_smiles(data.query)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_molecule(data: MoleculeInput):
    if not shortlister:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Smart resolve: try SMILES first, then name/formula via PubChem
        resolved = resolve_input_to_smiles(data.smiles)
        smiles = resolved["resolved_smiles"]
        input_type = resolved["input_type"]
        compound_name = resolved.get("compound_name")
        molecular_formula = resolved.get("molecular_formula")

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Could not parse the resolved SMILES structure")

        prob, pred_class = shortlister.predict_single(smiles)
        
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
            "smiles": smiles,
            "original_input": data.smiles,
            "input_type": input_type,
            "compound_name": compound_name,
            "molecular_formula": molecular_formula,
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
        # Resolve each input (could be SMILES, name, or formula)
        resolved_smiles = []
        resolution_report = []
        for entry in data.smiles_list:
            cleaned = entry.strip()
            if not cleaned:
                continue
            try:
                resolved = resolve_input_to_smiles(cleaned)
                resolved_smiles.append(resolved["resolved_smiles"])
                resolution_report.append({
                    "original_input": cleaned,
                    "resolved_smiles": resolved["resolved_smiles"],
                    "input_type": resolved["input_type"],
                    "resolved": True
                })
            except ValueError:
                # Keep original if resolution fails – let graph parser decide validity.
                resolved_smiles.append(cleaned)
                resolution_report.append({
                    "original_input": cleaned,
                    "resolved_smiles": cleaned,
                    "input_type": "unknown",
                    "resolved": False
                })

        results_df = shortlister.shortlist_drugs(
            resolved_smiles,
            top_k=len(resolved_smiles),
            threshold=0.0
        )
        # Convert DataFrame to list of dictionaries for JSON serialization
        results = results_df.to_dict(orient='records')
        return {
            "results": results,
            "resolution_report": resolution_report,
            "total_inputs": len(resolution_report),
            "total_ranked": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
