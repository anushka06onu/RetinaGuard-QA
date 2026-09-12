import React, { useState, useRef, useEffect } from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  RefreshCw, 
  Eye, 
  Upload, 
  CheckCircle2, 
  HelpCircle, 
  Activity, 
  Layers, 
  ExternalLink,
  ChevronRight,
  Info,
  X,
  Clock
} from 'lucide-react';

interface QualityProbabilities {
  good: number;
  usable: number;
  reject: number;
}

interface QualityAttributes {
  artifact?: 'none' | 'mild' | 'severe' | null;
  clarity?: 'high' | 'moderate' | 'low' | null;
  field_definition?: 'adequate' | 'incomplete' | 'poor' | null;
}

interface PredictionResponse {
  model_version: string;
  quality: 'good' | 'usable' | 'reject';
  probabilities: QualityProbabilities;
  calibrated_confidence: number;
  uncertainty: number;
  ood_score: number;
  decision: 'accept' | 'recapture' | 'manual_review' | 'unsupported_input';
  quality_attributes: QualityAttributes;
  feedback: string[];
  disclaimer: string;
  latency_ms?: number;
}

const MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024; // 15 MB
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Revoke object URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const clearSelection = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (previewUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (previewUrl && typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(previewUrl);
    }

    if (!file.type.startsWith('image/') || (!file.type.includes('jpeg') && !file.type.includes('png') && !file.type.includes('jpg'))) {
      setError('Please select a valid image file (JPEG or PNG format).');
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File is too large (${(file.size / (1024 * 1024)).toFixed(2)} MB). Maximum allowed upload size is 15 MB.`);
      setSelectedFile(null);
      setPreviewUrl(null);
      return;
    }

    setError(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  };

  const runPrediction = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    const controller = new AbortController();
    abortControllerRef.current = controller;
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    try {
      const resp = await fetch(`${API_BASE}/api/v1/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!resp.ok) {
        if (resp.status === 503) {
          throw new Error('Backend service is not ready. Model or calibration artifacts are currently loading.');
        }
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with error status ${resp.status}`);
      }

      const data: PredictionResponse = await resp.json();
      setResult(data);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Inference request timed out after 30 seconds or was cancelled.');
      } else {
        setError(err.message || 'Inference error occurred. Please check backend connection.');
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  const loadFixture = (type: 'test_pattern_a' | 'test_pattern_b' | 'test_pattern_c' | 'ood_fixture') => {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 384;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 384, 384);

    if (type === 'ood_fixture') {
      ctx.fillStyle = '#334155';
      ctx.fillRect(40, 40, 304, 304);
      ctx.fillStyle = '#64748b';
      ctx.fillRect(100, 100, 80, 180);
      ctx.fillRect(204, 100, 80, 180);
    } else {
      ctx.save();
      ctx.beginPath();
      ctx.arc(192, 192, 165, 0, Math.PI * 2);
      ctx.clip();

      const grad = ctx.createRadialGradient(192, 192, 20, 192, 192, 170);
      if (type === 'test_pattern_a') {
        grad.addColorStop(0, '#f97316');
        grad.addColorStop(1, '#9a3412');
      } else if (type === 'test_pattern_b') {
        grad.addColorStop(0, '#ea580c');
        grad.addColorStop(1, '#7c2d12');
      } else {
        grad.addColorStop(0, '#7c2d12');
        grad.addColorStop(1, '#451a03');
      }
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 384, 384);

      ctx.beginPath();
      ctx.arc(120, 192, 28, 0, Math.PI * 2);
      ctx.fillStyle = '#fef08a';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(240, 192, 18, 0, Math.PI * 2);
      ctx.fillStyle = '#451a03';
      ctx.fill();
      ctx.restore();
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `interface_fixture_${type}.jpg`, { type: 'image/jpeg' });
        processFile(file);
      }
    }, 'image/jpeg');
  };

  const renderDecisionBadge = (decision: string) => {
    switch (decision) {
      case 'accept':
        return (
          <div data-testid="decision-badge" className="bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg p-4 flex items-start gap-3">
            <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Accept</div>
              <div className="text-sm text-emerald-700 mt-0.5">Model-assessed technical quality adequate under experimental quality gate. Qualified clinical review still required.</div>
            </div>
          </div>
        );
      case 'recapture':
        return (
          <div data-testid="decision-badge" className="bg-rose-50 border border-rose-200 text-rose-800 rounded-lg p-4 flex items-start gap-3">
            <RefreshCw className="w-6 h-6 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Recapture Recommended</div>
              <div className="text-sm text-rose-700 mt-0.5">Physical or optical quality degraded. Follow capture guidance.</div>
            </div>
          </div>
        );
      case 'manual_review':
        return (
          <div data-testid="decision-badge" className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-4 flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Manual Review Required</div>
              <div className="text-sm text-amber-700 mt-0.5">High uncertainty or borderline score near decision boundary. Clinician review requested.</div>
            </div>
          </div>
        );
      default:
        return (
          <div data-testid="decision-badge" className="bg-slate-100 border border-slate-300 text-slate-800 rounded-lg p-4 flex items-start gap-3">
            <HelpCircle className="w-6 h-6 text-slate-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Unsupported / Out-of-Distribution</div>
              <div className="text-sm text-slate-600 mt-0.5">Input is not recognized as a standard color fundus photograph.</div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800 font-sans">
      {/* 1. Accessible Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-teal-700 text-white flex items-center justify-center font-bold text-lg">
              <Eye className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-slate-900 tracking-tight text-lg">RetinaGuard<span className="text-teal-700">-QA</span></span>
              <span className="text-xs text-slate-500 ml-2 hidden sm:inline border-l border-slate-200 pl-2">Fundus Image QA (Research Prototype)</span>
            </div>
          </div>
          <nav aria-label="Main Navigation" className="flex items-center gap-4 sm:gap-6 text-xs sm:text-sm font-medium text-slate-600">
            <a href="#research" className="hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none rounded p-1">Research</a>
            <a href="#demo" className="hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none rounded p-1">Demo</a>
            <a href="#evidence" className="hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none rounded p-1">Evidence</a>
            <a href="#limitations" className="hover:text-slate-900 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none rounded p-1">Limitations</a>
            <a href="https://github.com/anushka06onu/RetinaGuard-QA" target="_blank" rel="noreferrer" className="flex items-center gap-1 text-teal-800 hover:text-teal-900 font-semibold p-1">
              GitHub <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </nav>
        </div>
      </header>

      {/* 2. Hero */}
      <section className="py-12 sm:py-16 px-4 bg-white border-b border-slate-200 text-center">
        <div className="max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200">
            <ShieldCheck className="w-3.5 h-3.5" /> Quality Assurance Pipeline (Research Prototype)
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Uncertainty-Aware Quality Control for Retinal Fundus Imaging
          </h1>
          <p className="text-sm sm:text-lg text-slate-600 leading-relaxed">
            A lightweight imaging pipeline designed for cross-dataset generalization evaluation, predicting acquisition quality attributes, and abstaining when predictive entropy is elevated.
          </p>
          <div className="pt-2 flex justify-center gap-4">
            <a href="#demo" className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-teal-700 text-white font-semibold text-sm hover:bg-teal-800 transition shadow-sm">
              Try Interactive Inspection <ChevronRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      {/* 3. Research Pillars */}
      <section id="research" className="py-12 px-4 max-w-6xl mx-auto w-full">
        <h2 className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-6 text-center">Three Pillars of Technical Reliability</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <Activity className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">1. Defocus & Blur Detection</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Optical defocus obscures microaneurysms and fine vascular arcades, causing quality degradation.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">2. Illumination & Artifacts</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Flash saturation and dark vignetting destroy macular dynamic range. The model predicts granular acquisition attributes.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">3. Calibrated Abstention</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Instead of producing overconfident errors on borderline images, the system computes predictive entropy and routes uncertain scans for review.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Interactive Demo Workspace */}
      <section id="demo" className="py-12 px-4 bg-slate-100 border-y border-slate-200">
        <div className="max-w-6xl mx-auto space-y-6">
          <div>
            <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Interactive Quality Inspection</h2>
            <p className="text-sm text-slate-600">Upload a color fundus image (JPEG/PNG up to 15MB) or test interface fixtures.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Upload Panel */}
            <div className="lg:col-span-5 bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col gap-5">
              <label 
                htmlFor="file-upload"
                data-testid="dropzone"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInputRef.current?.click();
                  }
                }}
                className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center cursor-pointer hover:border-teal-600 hover:bg-slate-50 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none transition flex flex-col items-center justify-center min-h-[200px]"
              >
                <input 
                  id="file-upload"
                  type="file" 
                  data-testid="file-input"
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                  accept="image/jpeg,image/png" 
                  className="hidden" 
                />
                {previewUrl ? (
                  <div className="space-y-3">
                    <img src={previewUrl} alt="Preview" className="max-h-48 rounded-lg object-contain mx-auto" />
                    {selectedFile && (
                      <div className="text-xs text-slate-600 font-medium flex items-center justify-center gap-2">
                        <span>{selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-slate-400 mb-2" />
                    <p className="text-sm font-semibold text-slate-700">Click or press Enter to upload fundus photo</p>
                    <p className="text-xs text-slate-500 mt-1">JPEG or PNG up to 15MB</p>
                  </>
                )}
              </label>

              {selectedFile && (
                <div className="flex justify-between items-center bg-slate-50 p-2.5 rounded-lg border border-slate-200">
                  <span className="text-xs text-slate-700 font-medium truncate max-w-[200px]">{selectedFile.name}</span>
                  <button 
                    type="button" 
                    onClick={clearSelection} 
                    className="text-xs text-slate-500 hover:text-rose-600 flex items-center gap-1 font-semibold"
                  >
                    <X className="w-3.5 h-3.5" /> Clear
                  </button>
                </div>
              )}

              {/* Interface Test Fixtures */}
              <div>
                <span className="text-xs font-bold text-slate-600 uppercase tracking-wider block mb-1">Interface Test Fixtures</span>
                <span className="text-[11px] text-slate-500 block mb-2">Synthetic canvas patterns for API/UI verification only (no expected clinical class).</span>
                <div className="grid grid-cols-2 gap-2">
                  <button data-testid="fixture-good" type="button" onClick={() => loadFixture('test_pattern_a')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Interface Fixture A
                  </button>
                  <button data-testid="fixture-blur" type="button" onClick={() => loadFixture('test_pattern_b')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Interface Fixture B
                  </button>
                  <button data-testid="fixture-underexposed" type="button" onClick={() => loadFixture('test_pattern_c')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Interface Fixture C
                  </button>
                  <button data-testid="fixture-ood" type="button" onClick={() => loadFixture('ood_fixture')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Non-Fundus Fixture
                  </button>
                </div>
              </div>

              <button
                data-testid="predict-button"
                type="button"
                disabled={!selectedFile || loading}
                onClick={runPrediction}
                className="w-full py-3 px-4 rounded-lg bg-teal-700 text-white font-bold text-sm hover:bg-teal-800 disabled:opacity-50 transition shadow-sm flex items-center justify-center gap-2 focus-visible:ring-2 focus-visible:ring-teal-600 outline-none"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
                {loading ? 'Evaluating Quality Gate...' : 'Execute Quality Inspection'}
              </button>

              <p className="text-xs text-slate-500 text-center flex items-center justify-center gap-1">
                <Info className="w-3.5 h-3.5" /> Transient in-memory processing. Zero server-side persistence.
              </p>
            </div>

            {/* Right Results Panel */}
            <div 
              aria-live="polite" 
              className="lg:col-span-7 bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between"
            >
              {error && (
                <div role="alert" className="bg-rose-50 border border-rose-200 text-rose-800 text-sm p-4 rounded-lg flex items-start justify-between">
                  <div>
                    <div className="font-bold mb-1">Inference Error</div>
                    <div>{error}</div>
                  </div>
                  <button onClick={runPrediction} className="text-xs font-bold text-rose-700 underline hover:text-rose-900 ml-4">Retry</button>
                </div>
              )}

              {!result && !loading && !error && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-500 space-y-3">
                  <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
                    <Activity className="w-6 h-6" />
                  </div>
                  <h3 className="font-bold text-slate-700">Awaiting Image Ingestion</h3>
                  <p className="text-xs max-w-sm">Upload a retinal scan to view automated quality triage, predictive uncertainty, and physical capture feedback.</p>
                </div>
              )}

              {loading && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3" role="status">
                  <RefreshCw className="w-8 h-8 text-teal-700 animate-spin" />
                  <p className="text-sm font-semibold text-slate-700">Executing Canonical FOV Alignment & Multi-Task Inference...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Decision Banner */}
                  {renderDecisionBadge(result.decision)}

                  {/* Quantitative Metrics Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 border-y border-slate-100 py-4">
                    <div>
                      <span className="text-xs font-semibold text-slate-500 uppercase block">Quality Grade</span>
                      <span className="text-xl font-bold text-slate-900 capitalize">{result.quality}</span>
                      <span className="text-xs text-slate-500 block">{(result.calibrated_confidence * 100).toFixed(1)}% Conf</span>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-slate-500 uppercase block">Uncertainty</span>
                      <span className="text-xl font-bold text-slate-900">{result.uncertainty.toFixed(3)} <small className="text-xs font-normal">bits</small></span>
                      <span className="text-xs text-slate-500 block">Shannon Entropy</span>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-slate-500 uppercase block">OOD Score</span>
                      <span className="text-xl font-bold text-slate-900">{result.ood_score.toFixed(2)}</span>
                      <span className="text-xs text-slate-500 block">Energy Metric</span>
                    </div>
                  </div>

                  {/* Granular Quality Attributes (If supported by model) */}
                  {(result.quality_attributes.artifact || result.quality_attributes.clarity || result.quality_attributes.field_definition) && (
                    <div>
                      <span className="text-xs font-bold text-slate-600 uppercase tracking-wider block mb-2">Quality Attributes</span>
                      <div className="grid grid-cols-3 gap-2 text-center text-xs">
                        {result.quality_attributes.clarity && (
                          <div className="bg-slate-50 border border-slate-200 p-2.5 rounded-lg">
                            <span className="text-slate-500 block text-[11px]">Clarity</span>
                            <span className="font-bold text-slate-900 capitalize">{result.quality_attributes.clarity}</span>
                          </div>
                        )}
                        {result.quality_attributes.artifact && (
                          <div className="bg-slate-50 border border-slate-200 p-2.5 rounded-lg">
                            <span className="text-slate-500 block text-[11px]">Artifact Severity</span>
                            <span className="font-bold text-slate-900 capitalize">{result.quality_attributes.artifact}</span>
                          </div>
                        )}
                        {result.quality_attributes.field_definition && (
                          <div className="bg-slate-50 border border-slate-200 p-2.5 rounded-lg">
                            <span className="text-slate-500 block text-[11px]">Field Definition</span>
                            <span className="font-bold text-slate-900 capitalize">{result.quality_attributes.field_definition}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Calibrated Probability Distribution */}
                  <div>
                    <span className="text-xs font-bold text-slate-600 uppercase tracking-wider block mb-2">Class Probabilities</span>
                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                          <span>Good</span>
                          <span>{(result.probabilities.good * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-600 rounded-full" style={{ width: `${result.probabilities.good * 100}%` }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                          <span>Usable</span>
                          <span>{(result.probabilities.usable * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div className="h-full bg-amber-500 rounded-full" style={{ width: `${result.probabilities.usable * 100}%` }}></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-xs font-medium text-slate-700 mb-1">
                          <span>Reject</span>
                          <span>{(result.probabilities.reject * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                          <div className="h-full bg-rose-600 rounded-full" style={{ width: `${result.probabilities.reject * 100}%` }}></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Actionable Feedback */}
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block mb-2">Actionable Capture Guidance</span>
                    <ul className="space-y-1.5 text-xs text-slate-700">
                      {result.feedback.map((f, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-teal-700 font-bold">•</span>
                          <span>{f}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Provenance & Latency Footer */}
                  <div className="flex justify-between items-center text-[11px] text-slate-500 border-t border-slate-100 pt-3">
                    <span>Model: v{result.model_version}</span>
                    {result.latency_ms && (
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {result.latency_ms.toFixed(1)} ms
                      </span>
                    )}
                  </div>

                  {/* Non-Diagnostic Disclaimer */}
                  <div className="text-[11px] text-slate-500 italic">
                    {result.disclaimer}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* 5. Evidence Section */}
      <section id="evidence" className="py-12 px-4 max-w-6xl mx-auto w-full space-y-6">
        <div>
          <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Empirical Evidence & Benchmark Evaluation</h2>
          <p className="text-sm text-slate-600">Separating internal supervised validation from zero-shot external transfer.</p>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-teal-800">
            <Info className="w-4 h-4" /> Multi-Dataset Evaluation Protocol
          </div>
          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
            Final empirical evaluation encompasses a 3-seed campaign with patient-isolated splits across EyeQ (3-class) and DeepDRiD (binary + attributes), baseline comparisons (Single-Task MobileNetV3, EfficientNet-B0), core architectural ablations, temperature scaling calibration, and selective prediction curves.
          </p>
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-600 space-y-2">
            <div className="font-semibold text-slate-800">Evaluation Categories:</div>
            <ul className="list-disc list-inside space-y-1">
              <li><strong>Internal Supervised Validation:</strong> Multi-task quality and attribute assessment on held-out test partitions.</li>
              <li><strong>Zero-Shot External Transfer:</strong> EyeQ-trained representations evaluated directly on DeepDRiD without target fine-tuning.</li>
              <li><strong>Uncertainty & OOD Gating:</strong> Predictive entropy selective prediction and energy-based OOD rejection.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* 6. Limitations & Ethics */}
      <section id="limitations" className="py-12 px-4 max-w-6xl mx-auto w-full border-t border-slate-200">
        <div className="bg-slate-100 rounded-xl p-6 border border-slate-200 space-y-3">
          <h3 className="font-bold text-slate-900 text-base">Clinical & Ethical Boundaries</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            RetinaGuard-QA is a research prototype intended to investigate automated technical image-quality assistance. It does not diagnose diabetic retinopathy, glaucoma, or age-related macular degeneration. Internal media opacities (e.g. dense cataracts) produce optical scatter that degrades technical capture; the system assesses physical image usability but does not establish clinical disease etiology.
          </p>
        </div>
      </section>

      {/* 7. Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-8 px-4 text-center text-xs text-slate-500">
        <p>© 2026 RetinaGuard-QA Contributors. Distributed under the MIT Open Source License.</p>
        <p className="mt-1">Built with PyTorch, ONNX Runtime, FastAPI, React, and TypeScript.</p>
      </footer>
    </div>
  );
}

