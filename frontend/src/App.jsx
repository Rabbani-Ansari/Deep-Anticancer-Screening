import { useState, useRef } from 'react';
import axios from 'axios';
import {
  Atom,
  Search,
  Activity,
  ShieldCheck,
  ShieldAlert,
  Microscope,
  Zap,
  Info,
  Server,
  FileText,
  Download,
  Beaker,
  Droplets,
  Layers,
  Scale,
  List,
  Filter,
  CheckCircle2,
  LayoutDashboard,
  Settings,
  HelpCircle,
  ChevronRight,
  Database,
  Eye,
  BarChart2,
  TrendingUp,
  Award,
  X
} from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import './App.css';

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [smiles, setSmiles] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Batch Screening State
  const [batchSmiles, setBatchSmiles] = useState('');
  const [batchResults, setBatchResults] = useState(null);
  const [scanCount, setScanCount] = useState(0);
  const [scoreFilter, setScoreFilter] = useState(0.5);
  const [selectedMolecule, setSelectedMolecule] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!smiles) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, { smiles });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to connect to analysis server. Make sure 'run_modern_app.bat' is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleBatchScreen = async (e) => {
    e.preventDefault();
    if (!batchSmiles) return;

    // Split by newlines and filter empty
    const smilesList = batchSmiles.split('\n').filter(s => s.trim().length > 0);
    if (smilesList.length === 0) return;

    setLoading(true);
    setError(null);
    setBatchResults(null);

    try {
      const response = await axios.post(`${API_URL}/shortlist`, { smiles_list: smilesList });
      console.log("Batch Results:", response.data.results);
      setBatchResults(response.data.results);
      setScanCount(smilesList.length);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "Batch screening failed. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (exampleSmiles) => {
    setSmiles(exampleSmiles);
    setActivePage('dashboard');
  };

  const loadBatchExample = () => {
    const examples = [
      "CC1=C2C(=C(C=C1)O)C(=O)C3=C(C2=O)C(=CC=C3)O", // Doxorubicin
      "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O", // Methotrexate
      "CC(=O)Oc1ccccc1C(=O)O", // Aspirin
      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", // Caffeine
      "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C", // Paclitaxel
      "CCO" // Ethanol
    ].join('\n');
    setBatchSmiles(examples);
  };

  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const lines = text.split(/\r?\n/);

      if (lines.length === 0) return;

      // Basic CSV parsing
      // If first line contains "smiles", we try to find that column
      const headers = lines[0].toLowerCase().split(',');
      const smilesIndex = headers.indexOf('smiles');

      let extractedSmiles = [];

      if (smilesIndex !== -1) {
        // Use the 'smiles' column
        for (let i = 1; i < lines.length; i++) {
          const columns = lines[i].split(',');
          if (columns[smilesIndex] && columns[smilesIndex].trim()) {
            extractedSmiles.push(columns[smilesIndex].trim());
          }
        }
      } else {
        // Just take the first column if no 'smiles' header
        for (let i = 0; i < lines.length; i++) {
          const firstCol = lines[i].split(',')[0];
          if (firstCol && firstCol.trim()) {
            // Basic SMILES check (doesn't contain spaces and not empty)
            if (!firstCol.includes(' ') && firstCol.length > 0) {
              extractedSmiles.push(firstCol.trim());
            }
          }
        }
      }

      if (extractedSmiles.length > 0) {
        setBatchSmiles(extractedSmiles.join('\n'));
      }
    };
    reader.readAsText(file);
    // Reset input
    e.target.value = null;
  };

  // Prepare data for charts
  const getRadarData = (props) => [
    { subject: 'MW', A: Math.min(props.MolecularWeight / 500 * 100, 100), fullMark: 100 },
    { subject: 'LogP', A: Math.min(Math.max(props.LogP, 0) / 5 * 100, 100), fullMark: 100 },
    { subject: 'TPSA', A: Math.min(props.TPSA / 140 * 100, 100), fullMark: 100 },
    { subject: 'H-Don', A: Math.min(props.H_Donors / 5 * 100, 100), fullMark: 100 },
    { subject: 'H-Acc', A: Math.min(props.H_Acceptors / 10 * 100, 100), fullMark: 100 },
  ];

  // Analytics Helpers for Batch
  const getBatchAnalytics = () => {
    if (!batchResults || batchResults.length === 0) return null;
    const scores = batchResults.map(r => r.Predicted_Probability);
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const max = Math.max(...scores);
    const highValue = scores.filter(s => s >= 0.9).length;
    return { avg, max, highValue };
  };

  const getDistributionData = () => {
    if (!batchResults) return [];
    const bins = [0, 0, 0, 0, 0]; // 0-20, 21-40, 41-60, 61-80, 81-100
    batchResults.forEach(r => {
      const score = r.Predicted_Probability;
      const binIdx = Math.min(Math.floor(score * 5), 4);
      bins[binIdx]++;
    });
    return [
      { range: '0-20%', count: bins[0] },
      { range: '21-40%', count: bins[1] },
      { range: '41-60%', count: bins[2] },
      { range: '61-80%', count: bins[3] },
      { range: '81-100%', count: bins[4] },
    ];
  };

  const handleInspect = async (smilesStr) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, { smiles: smilesStr });
      setSelectedMolecule(response.data);
      setIsModalOpen(true);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch molecule details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0B1121] text-slate-300 font-sans overflow-hidden selection:bg-cyan-500/20">

      {/* SIDEBAR */}
      <aside className="w-64 bg-[#0f1629] border-r border-white/5 flex flex-col backdrop-blur-xl z-20">
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-600 to-cyan-500 p-2 rounded-lg shadow-lg shadow-blue-500/20">
              <Atom className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-white tracking-tight">OncoScreen</h1>
              <p className="text-[10px] text-slate-500 font-medium tracking-wider">DRUG DISCOVERY AI</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-3 mb-2 mt-2">Modules</div>

          <button
            onClick={() => setActivePage('dashboard')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group ${activePage === 'dashboard' ? 'bg-blue-600/10 text-white border border-blue-500/20' : 'hover:bg-white/5 hover:text-white'}`}
          >
            <LayoutDashboard className={`w-4 h-4 ${activePage === 'dashboard' ? 'text-blue-400' : 'text-slate-500 group-hover:text-slate-300'}`} />
            <span>Molecule Analysis</span>
            {activePage === 'dashboard' && <ChevronRight className="w-3 h-3 ml-auto text-blue-500" />}
          </button>

          <button
            onClick={() => setActivePage('screening')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group ${activePage === 'screening' ? 'bg-purple-600/10 text-white border border-purple-500/20' : 'hover:bg-white/5 hover:text-white'}`}
          >
            <Filter className={`w-4 h-4 ${activePage === 'screening' ? 'text-purple-400' : 'text-slate-500 group-hover:text-slate-300'}`} />
            <span>Library Screening</span>
            {activePage === 'screening' && <ChevronRight className="w-3 h-3 ml-auto text-purple-500" />}
          </button>

          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-3 mb-2 mt-6">System</div>

          <div className="px-3 py-2">
            <div className="flex items-center gap-2 text-xs text-slate-400 bg-black/20 p-2 rounded border border-white/5">
              <Server className="w-3 h-3 text-emerald-500" />
              <span className="font-mono">API: ONLINE</span>
            </div>
          </div>
        </nav>

        <div className="p-4 border-t border-white/5">
          <div className="text-xs text-slate-500 text-center">v1.2.0 • Build 2026.02</div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 overflow-y-auto bg-gradient-to-br from-[#0B1121] to-[#111827] relative">
        {/* Background Ambient Glow */}
        <div className="absolute top-0 left-0 w-full h-[500px] bg-blue-500/5 blur-[100px] pointer-events-none"></div>

        <div className="container mx-auto max-w-7xl p-8 relative z-10">

          {/* Header */}
          <header className="mb-8 flex justify-between items-end border-b border-white/5 pb-6">
            <div>
              <h2 className="text-2xl font-bold text-white mb-1">
                {activePage === 'dashboard' ? 'Single-Molecule Analyzer' : 'Library Batch Screening'}
              </h2>
              <div className="flex items-center gap-2 mt-1">
                <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px] font-bold uppercase tracking-wider border border-blue-500/20">Specialized Oncology Domain</span>
                <p className="text-slate-400 text-sm">
                  {activePage === 'dashboard' ? 'Analyze drug-likeness for Tumor Growth Inhibition.' : 'Rapidly filter compound libraries for anticancer hits.'}
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              {activePage === 'dashboard' && (
                <button className="btn btn-secondary text-xs">
                  <Download className="w-3 h-3" /> Export Report
                </button>
              )}
            </div>
          </header>

          {/* === VIEW 1: LEAD OPTIMIZATION (DASHBOARD) === */}
          {activePage === 'dashboard' && (
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 animate-fade-in">

              {/* Input Panel */}
              <div className="xl:col-span-4 space-y-6">
                <div className="card p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wide flex items-center gap-2">
                      <Search className="w-4 h-4 text-blue-400" /> Molecular Input
                    </h3>
                  </div>

                  <form onSubmit={handlePredict} className="space-y-4">
                    <textarea
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      placeholder="Enter SMILES string (e.g. CCO)..."
                      className="input-field w-full h-32 font-mono text-sm resize-none"
                    />
                    <button
                      type="submit"
                      disabled={loading}
                      className="w-full btn btn-primary py-3 group"
                    >
                      {loading ? <span className="animate-spin">⟳</span> : <Microscope className="w-5 h-5 group-hover:scale-110 transition-transform" />}
                      {loading ? 'Analyzing Structure...' : 'Analyze Molecule'}
                    </button>
                  </form>

                  <div className="mt-8 border-t border-white/5 pt-6">
                    <h4 className="text-xs font-semibold text-slate-500 uppercase mb-3 px-1">Quick Load Examples</h4>
                    <div className="space-y-2">
                      {[
                        { name: "Doxorubicin", type: "Active (Anthracycline)", color: "bg-emerald-500", smiles: "CC1=C2C(=C(C=C1)O)C(=O)C3=C(C2=O)C(=CC=C3)O" },
                        { name: "Paclitaxel", type: "Active (Taxane)", color: "bg-emerald-500", smiles: "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C" },
                        { name: "Aspirin", type: "Inactive Control", color: "bg-slate-500", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
                      ].map((item, i) => (
                        <button
                          key={i}
                          onClick={() => loadExample(item.smiles)}
                          className="w-full text-left p-3 rounded-xl bg-white/5 hover:bg-white/10 hover:border-white/20 border border-transparent transition-all group"
                        >
                          <div className="flex justify-between items-center">
                            <span className="text-sm font-medium text-slate-300 group-hover:text-white">{item.name}</span>
                            <span className={`w-2 h-2 rounded-full ${item.color} shadow-[0_0_8px_currentColor]`}></span>
                          </div>
                          <div className="text-[10px] text-slate-500 mt-0.5">{item.type}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10 flex gap-3">
                  <Info className="w-5 h-5 text-blue-400 shrink-0" />
                  <p className="text-xs text-blue-300/80 leading-relaxed">
                    Model trained on NCI-60/ChEMBL cancer datasets. Predictions for research use only. Structure converted to graph representation for GNN analysis.
                  </p>
                </div>
              </div>

              {/* Results Panel */}
              <div className="xl:col-span-8 space-y-6">

                {/* Structure & Status Header */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Structure Viewer */}
                  <div className="card p-4 relative min-h-[300px] flex items-center justify-center bg-white group hover:shadow-2xl transition-all">
                    <div className="absolute top-4 left-4 z-10 bg-slate-900/10 backdrop-blur text-slate-800 px-3 py-1 rounded-full text-xs font-bold border border-slate-900/10">
                      2D Structure
                    </div>
                    {result?.image ? (
                      <img
                        src={`data:image/png;base64,${result.image}`}
                        alt="Molecule"
                        className="max-w-full max-h-[280px] object-contain mix-blend-multiply group-hover:scale-105 transition-transform duration-500"
                      />
                    ) : (
                      <div className="text-slate-400 text-sm flex flex-col items-center gap-2">
                        <Atom className="w-10 h-10 opacity-20" />
                        <span>No molecule loaded</span>
                      </div>
                    )}
                  </div>

                  {/* Prediction Status */}
                  <div className="card p-8 flex flex-col justify-center relative overflow-hidden">
                    {result ? (
                      <>
                        <div className="absolute top-0 right-0 p-8 opacity-10">
                          {result.prediction_class === 1 ? <Activity className="w-32 h-32 text-emerald-500" /> : <ShieldCheck className="w-32 h-32 text-slate-500" />}
                        </div>
                        <div className="relative z-10">
                          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                            <Activity className="w-3 h-3 text-emerald-500" />
                            Predicted Anticancer Potency
                          </div>
                          <h2 className={`text-3xl font-bold mb-4 ${result.prediction_class === 1 ? 'text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400' : 'text-slate-400'}`}>
                            {result.prediction_class === 1 ? 'Active Tumor Inhibitor' : 'Low Potency Candidate'}
                          </h2>

                          <div className="mb-6">
                            <div className="flex justify-between text-xs font-mono text-slate-400 mb-2">
                              <span className="flex items-center gap-1">Growth Inhibition (GI50) <Info className="w-3 h-3 opacity-50 cursor-help" title="Probability of significant cancer cell growth inhibition (pGI50 > 6.0)" /></span>
                              <span>{(result.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all duration-1000 ${result.prediction_class === 1 ? 'bg-gradient-to-r from-emerald-500 to-cyan-400' : 'bg-slate-600'}`}
                                style={{ width: `${result.confidence * 100}%` }}
                              ></div>
                            </div>
                          </div>

                          <div className="flex gap-2">
                            <div className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${result.analysis.druglikeness === 'High' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'}`}>
                              {result.analysis.druglikeness} Drug-Likeness
                            </div>
                            <div className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 border border-white/5 text-slate-400">
                              {result.analysis.lipinski_violations} Violations
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center text-slate-500">
                        <Activity className="w-12 h-12 mx-auto mb-4 opacity-20" />
                        <p>Awaiting Analysis Results</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Properties Grid */}
                {result && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="card p-6">
                      <h4 className="text-sm font-bold text-slate-300 mb-6 flex items-center gap-2">
                        <Scale className="w-4 h-4 text-cyan-400" /> Physicochemical Descriptors
                      </h4>
                      <div className="grid grid-cols-2 gap-y-6 gap-x-4">
                        {[
                          { label: "Mol. Weight", val: result.properties.MolecularWeight.toFixed(2), unit: "g/mol" },
                          { label: "LogP", val: result.properties.LogP.toFixed(2), unit: "", highlight: result.properties.LogP > 5 },
                          { label: "TPSA", val: result.properties.TPSA.toFixed(2), unit: "Å²" },
                          { label: "Rotatable Bonds", val: result.properties.RotatableBonds, unit: "" },
                          { label: "H-Bond Donors", val: result.properties.H_Donors, unit: "" },
                          { label: "H-Bond Acceptors", val: result.properties.H_Acceptors, unit: "" }
                        ].map((p, i) => (
                          <div key={i}>
                            <div className="text-[10px] text-slate-500 uppercase font-semibold mb-1">{p.label}</div>
                            <div className={`text-lg font-mono flex items-baseline gap-1 ${p.highlight ? 'text-yellow-400' : 'text-slate-200'}`}>
                              {p.val} <span className="text-[10px] text-slate-500 font-normal">{p.unit}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="card p-6 flex flex-col">
                      <h4 className="text-sm font-bold text-slate-300 mb-2 flex items-center gap-2">
                        <Radar className="w-4 h-4 text-purple-400" /> Bioavailability Radar
                      </h4>
                      <div className="flex-1 min-h-[200px] w-full -ml-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={getRadarData(result.properties)}>
                            <PolarGrid stroke="#334155" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                            <Radar name="Profile" dataKey="A" stroke="#8b5cf6" strokeWidth={2} fill="#8b5cf6" fillOpacity={0.4} />
                            <Tooltip contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f1f5f9' }} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                )}

              </div>
            </div>
          )}

          {/* === VIEW 2: VIRTUAL SCREENING (BATCH) === */}
          {activePage === 'screening' && (
            <div className="space-y-8 animate-fade-in">

              {/* Analytics Summary Bar */}
              {batchResults && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="card p-4 bg-blue-500/5 border-blue-500/10">
                    <div className="flex justify-between items-start mb-2">
                      <TrendingUp className="w-4 h-4 text-blue-400" />
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Avg. Probability</span>
                    </div>
                    <div className="text-2xl font-mono text-white">{(getBatchAnalytics().avg * 100).toFixed(1)}%</div>
                  </div>
                  <div className="card p-4 bg-emerald-500/5 border-emerald-500/10">
                    <div className="flex justify-between items-start mb-2">
                      <Award className="w-4 h-4 text-emerald-400" />
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Top Score</span>
                    </div>
                    <div className="text-2xl font-mono text-white">{(getBatchAnalytics().max * 100).toFixed(1)}%</div>
                  </div>
                  <div className="card p-4 bg-purple-500/5 border-purple-500/10">
                    <div className="flex justify-between items-start mb-2">
                      <Zap className="w-4 h-4 text-purple-400" />
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Hi-Value Hits</span>
                    </div>
                    <div className="text-2xl font-mono text-white">{getBatchAnalytics().highValue} <span className="text-xs text-slate-500 font-normal">(&gt;90%)</span></div>
                  </div>
                  <div className="card p-4 bg-slate-500/5 border-slate-500/10">
                    <div className="flex justify-between items-start mb-2">
                      <Database className="w-4 h-4 text-slate-400" />
                      <span className="text-[10px] font-bold text-slate-500 uppercase">Library Size</span>
                    </div>
                    <div className="text-2xl font-mono text-white">{scanCount}</div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
                <div className="xl:col-span-4 space-y-6">
                  <div className="card p-6 flex flex-col min-h-[500px]">
                    <div className="mb-4">
                      <h3 className="text-sm font-bold text-slate-200 uppercase flex items-center gap-2">
                        <Database className="w-4 h-4 text-purple-400" /> Screening Library
                      </h3>
                      <p className="text-xs text-slate-500 mt-1">Paste candidate SMILES or import CSV library.</p>
                    </div>

                    <form onSubmit={handleBatchScreen} className="flex-1 flex flex-col gap-4">
                      <textarea
                        value={batchSmiles}
                        onChange={(e) => setBatchSmiles(e.target.value)}
                        placeholder="Paste list of SMILES strings..."
                        className="input-field flex-1 font-mono text-xs resize-none min-h-[200px]"
                      />
                      <div className="grid grid-cols-2 gap-3">
                        <div className="flex flex-col gap-3">
                          <button type="button" onClick={loadBatchExample} className="btn btn-secondary text-xs w-full">Load Sample Set</button>
                          <button
                            type="button"
                            onClick={() => fileInputRef.current.click()}
                            className="btn btn-secondary text-xs w-full flex items-center justify-center gap-2 border-dashed border-slate-700"
                          >
                            <FileText className="w-3 h-3" /> Import CSV
                          </button>
                          <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileUpload}
                            accept=".csv,.txt"
                            className="hidden"
                          />
                        </div>
                        <button type="submit" disabled={loading} className="btn btn-primary bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500">
                          {loading ? 'Screening...' : 'Screen Library'}
                        </button>
                      </div>
                    </form>

                    {batchResults && (
                      <div className="mt-8 border-t border-white/5 pt-6">
                        <h4 className="text-xs font-bold text-slate-500 uppercase mb-4 flex items-center gap-2">
                          <BarChart2 className="w-3 h-3" /> Score Distribution
                        </h4>
                        <div className="h-40 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={getDistributionData()}>
                              <XAxis dataKey="range" hide />
                              <YAxis hide />
                              <Tooltip
                                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', fontSize: '10px' }}
                                cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                              />
                              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="xl:col-span-8 space-y-6">
                  {error && (
                    <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 flex items-center gap-3 animate-fade-in">
                      <ShieldAlert className="w-5 h-5" />
                      {error}
                    </div>
                  )}

                  <div className="card h-full min-h-[600px] flex flex-col">
                    <div className="p-4 border-b border-white/5 flex flex-wrap justify-between items-center bg-white/[0.02] gap-4">
                      <div className="flex items-center gap-3">
                        <List className="w-4 h-4 text-slate-400" />
                        <span className="text-sm font-semibold text-slate-300">Ranked Results</span>
                        {batchResults && (
                          <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 text-[10px] font-bold">
                            {batchResults.filter(r => r.Predicted_Probability >= scoreFilter).length} VISIBLE / {batchResults.length} HITS
                          </span>
                        )}
                      </div>

                      {batchResults && (
                        <div className="flex items-center gap-4 ml-auto">
                          <div className="flex items-center gap-3 bg-black/20 px-3 py-1.5 rounded-lg border border-white/5">
                            <span className="text-[10px] font-bold text-slate-500 uppercase">Min Score</span>
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.1"
                              value={scoreFilter}
                              onChange={(e) => setScoreFilter(parseFloat(e.target.value))}
                              className="w-24 accent-blue-500"
                            />
                            <span className="text-xs font-mono text-blue-400">{(scoreFilter * 100).toFixed(0)}%</span>
                          </div>
                          <button className="text-xs text-slate-500 hover:text-white transition-colors">Export CSV</button>
                        </div>
                      )}
                    </div>

                    <div className="flex-1 overflow-auto custom-scrollbar">
                      {!batchResults ? (
                        <div className="h-full flex flex-col items-center justify-center text-slate-600">
                          <Filter className="w-16 h-16 opacity-20 mb-4" />
                          <p>No screening data available</p>
                          <span className="text-xs opacity-50">Load a library to identify hits</span>
                        </div>
                      ) : (
                        <table className="w-full text-left border-collapse">
                          <thead className="sticky top-0 bg-[#151e32] z-10 shadow-lg">
                            <tr>
                              <th className="p-4 text-xs font-bold text-slate-500 uppercase tracking-wider w-16">Rank</th>
                              <th className="p-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Structure (SMILES)</th>
                              <th className="p-4 text-xs font-bold text-slate-500 uppercase tracking-wider w-32">Probability</th>
                              <th className="p-4 text-xs font-bold text-slate-500 uppercase tracking-wider w-40 text-right">Actions</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-white/5">
                            {batchResults
                              .filter(item => item.Predicted_Probability >= scoreFilter)
                              .map((item, index) => (
                                <tr key={index} className="hover:bg-white/5 transition-colors group">
                                  <td className="p-4 font-mono text-slate-500 text-xs">#{item.Rank}</td>
                                  <td className="p-4">
                                    <div className="font-mono text-xs text-slate-400 truncate max-w-[280px]" title={item.SMILES}>{item.SMILES}</div>
                                  </td>
                                  <td className="p-4">
                                    <div className="flex items-center gap-3">
                                      <span className={`text-xs font-bold ${item.Predicted_Probability > 0.8 ? 'text-emerald-400' : item.Predicted_Probability > 0.5 ? 'text-blue-400' : 'text-slate-500'}`}>
                                        {(item.Predicted_Probability * 100).toFixed(1)}%
                                      </span>
                                      <div className="flex-1 h-1 w-12 bg-slate-800 rounded-full overflow-hidden hidden md:block">
                                        <div className={`h-full ${item.Predicted_Probability > 0.5 ? 'bg-blue-500' : 'bg-slate-600'}`} style={{ width: `${item.Predicted_Probability * 100}%` }}></div>
                                      </div>
                                    </div>
                                  </td>
                                  <td className="p-4 text-right">
                                    <button
                                      onClick={() => handleInspect(item.SMILES)}
                                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-slate-300 text-[10px] font-bold uppercase tracking-wider transition-all border border-white/5"
                                    >
                                      <Eye className="w-3 h-3" /> Quick View
                                    </button>
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* QUICK INSPECT MODAL */}
          {isModalOpen && selectedMolecule && (
            <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
              <div className="absolute inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setIsModalOpen(false)}></div>
              <div className="card w-full max-w-4xl relative z-10 overflow-hidden animate-zoom-in max-h-[90vh] flex flex-col">
                <div className="p-6 border-b border-white/5 flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-bold text-white">Compound Analysis</h3>
                    <p className="text-xs text-slate-500 font-mono mt-1 break-all">{selectedMolecule.smiles}</p>
                  </div>
                  <button onClick={() => setIsModalOpen(false)} className="p-2 hover:bg-white/5 rounded-full transition-colors">
                    <X className="w-5 h-5 text-slate-400" />
                  </button>
                </div>

                <div className="flex-1 overflow-y-auto p-8">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    <div className="bg-white p-6 rounded-2xl flex items-center justify-center">
                      <img
                        src={`data:image/png;base64,${selectedMolecule.image}`}
                        alt="Molecule"
                        className="max-w-full max-h-[300px] object-contain mix-blend-multiply"
                      />
                    </div>

                    <div className="space-y-8">
                      <div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Prediction Verdict</div>
                        <div className={`text-2xl font-bold ${selectedMolecule.prediction_class === 1 ? 'text-emerald-400' : 'text-slate-400'}`}>
                          {selectedMolecule.prediction_text}
                        </div>
                        <div className="text-xs text-slate-500 mt-1">Confidence: {(selectedMolecule.confidence * 100).toFixed(1)}%</div>
                      </div>

                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">MW</div>
                          <div className="text-lg font-mono text-slate-200">{selectedMolecule.properties.MolecularWeight.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">LogP</div>
                          <div className="text-lg font-mono text-slate-200">{selectedMolecule.properties.LogP.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">TPSA</div>
                          <div className="text-lg font-mono text-slate-200">{selectedMolecule.properties.TPSA.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">Drug-likeness</div>
                          <div className={`text-lg font-bold ${selectedMolecule.analysis.druglikeness === 'High' ? 'text-emerald-400' : 'text-yellow-400'}`}>
                            {selectedMolecule.analysis.druglikeness}
                          </div>
                        </div>
                      </div>

                      <div className="p-4 rounded-xl bg-blue-500/5 border border-blue-500/10">
                        <h4 className="text-xs font-bold text-blue-400 uppercase mb-2">Lipinski Rule Analysis</h4>
                        <p className="text-xs text-blue-300/70 leading-relaxed">
                          This molecule has {selectedMolecule.analysis.lipinski_violations} violations.
                          {selectedMolecule.analysis.lipinski_violations === 0
                            ? " It perfectly adheres to the Rule of Five, indicating high potential for oral bioavailability."
                            : " It may have challenges with oral absorption according to traditional medicinal chemistry rules."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 border-t border-white/5 bg-white/[0.02] flex justify-end gap-3">
                  <button onClick={() => setIsModalOpen(false)} className="btn btn-secondary py-2">Close</button>
                  <button onClick={() => loadExample(selectedMolecule.smiles)} className="btn btn-primary py-2 px-6">Send to Lead Optimization</button>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
