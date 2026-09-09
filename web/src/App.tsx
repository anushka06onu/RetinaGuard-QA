import React, { useState, useRef } from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  RefreshCw, 
  Eye, 
  Upload, 
  CheckCircle2, 
  HelpCircle, 
  FileText, 
  Activity, 
  Cpu, 
  Layers, 
  ExternalLink,
  ChevronRight,
  Info
} from 'lucide-react';

interface QualityProbabilities {
  good: number;
  usable: number;
  reject: number;
}

interface QualityAttributes {
  artifact?: string | null;
  clarity?: string | null;
  field_definition?: string | null;
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

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (JPEG, PNG, TIFF).');
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

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || `Server responded with status ${resp.status}`);
      }

      const data: PredictionResponse = await resp.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Inference error occurred. Please check the backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (type: 'good' | 'blur' | 'underexposed' | 'ood') => {
    const canvas = document.createElement('canvas');
    canvas.width = 384;
    canvas.height = 384;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 384, 384);

    if (type === 'ood') {
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
      if (type === 'good') {
        grad.addColorStop(0, '#f97316');
        grad.addColorStop(1, '#9a3412');
      } else if (type === 'blur') {
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

      if (type !== 'blur') {
        ctx.strokeStyle = '#991b1b';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(120, 192);
        ctx.bezierCurveTo(140, 140, 200, 110, 280, 90);
        ctx.moveTo(120, 192);
        ctx.bezierCurveTo(140, 240, 200, 270, 280, 290);
        ctx.stroke();
      }
      ctx.restore();
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `preset_${type}.jpg`, { type: 'image/jpeg' });
        processFile(file);
      }
    }, 'image/jpeg');
  };

  const renderDecisionBadge = (decision: string) => {
    switch (decision) {
      case 'accept':
        return (
          <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg p-4 flex items-start gap-3">
            <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Accept</div>
              <div className="text-sm text-emerald-700 mt-0.5">Image satisfies technical quality standards. Proceed to diagnostic evaluation.</div>
            </div>
          </div>
        );
      case 'recapture':
        return (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-lg p-4 flex items-start gap-3">
            <RefreshCw className="w-6 h-6 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Recapture Required</div>
              <div className="text-sm text-rose-700 mt-0.5">Physical or optical quality inadequate for trustworthy review. Follow capture guidance.</div>
            </div>
          </div>
        );
      case 'manual_review':
        return (
          <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-4 flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-bold text-base tracking-tight uppercase">Decision: Manual Review Required</div>
              <div className="text-sm text-amber-700 mt-0.5">High uncertainty near decision boundary. Operator inspection requested.</div>
            </div>
          </div>
        );
      default:
        return (
          <div className="bg-slate-100 border border-slate-300 text-slate-800 rounded-lg p-4 flex items-start gap-3">
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
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-800">
      {/* 1. Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-teal-700 text-white flex items-center justify-center font-bold text-lg">
              <Eye className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-slate-900 tracking-tight text-lg">RetinaGuard<span className="text-teal-700">-QA</span></span>
              <span className="text-xs text-slate-600 ml-2 hidden sm:inline border-l border-slate-200 pl-2">Reliable Fundus Quality Control</span>
            </div>
          </div>
          <nav className="flex items-center gap-6 text-sm font-medium text-slate-600">
            <a href="#research" className="hover:text-slate-900 transition">Research</a>
            <a href="#demo" className="hover:text-slate-900 transition">Demo</a>
            <a href="#evidence" className="hover:text-slate-900 transition">Evidence</a>
            <a href="#limitations" className="hover:text-slate-900 transition">Limitations</a>
            <a href="https://github.com/anushka06onu/RetinaGuard-QA" target="_blank" rel="noreferrer" className="flex items-center gap-1 text-teal-800 hover:text-teal-900 font-semibold">
              GitHub <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </nav>
        </div>
      </header>

      {/* 2. Hero */}
      <section className="py-16 px-4 bg-white border-b border-slate-200 text-center">
        <div className="max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200">
            <ShieldCheck className="w-3.5 h-3.5" /> Autonomous Pre-Diagnostic Quality Gate
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight">
            Uncertainty-Aware Quality Control for Retinal Fundus Imaging
          </h1>
          <p className="text-base sm:text-lg text-slate-600 leading-relaxed">
            A lightweight, device-robust imaging pipeline that generalizes across datasets, explains physical acquisition defects, and abstains when prediction is unreliable.
          </p>
          <div className="pt-2 flex justify-center gap-4">
            <a href="#demo" className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-teal-700 text-white font-semibold text-sm hover:bg-teal-800 transition shadow-sm">
              Try the Interactive Demo <ChevronRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      {/* 3. Why Quality Matters */}
      <section className="py-12 px-4 max-w-6xl mx-auto w-full">
        <h2 className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-6 text-center">Three Pillars of Diagnostic Reliability</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <Activity className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">1. Defocus & Blur Detection</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Optical defocus obscures microaneurysms and fine vascular arcades, leading to false negatives in downstream DR classification.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <Layers className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">2. Illumination & Exposure</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Xenon flash saturation and dark vignetting destroy macular dynamic range. The gate identifies underexposed and overexposed quadrants.
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
            <div className="w-10 h-10 rounded-lg bg-teal-50 text-teal-700 flex items-center justify-center font-bold mb-4">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <h3 className="font-bold text-slate-900 text-base mb-2">3. Calibrated Abstention</h3>
            <p className="text-sm text-slate-600 leading-relaxed">
              Instead of producing overconfident errors on borderline images, the system computes Shannon entropy and routes uncertain scans for manual clinician review.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Demo Workspace */}
      <section id="demo" className="py-12 px-4 bg-slate-100 border-y border-slate-200">
        <div className="max-w-6xl mx-auto space-y-6">
          <div>
            <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Interactive Quality Inspection</h2>
            <p className="text-sm text-slate-600">Upload a color fundus image or run standard benchmark presets.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Upload Panel (5 cols) */}
            <div className="lg:col-span-5 bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col gap-5">
              <div 
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 rounded-xl p-6 text-center cursor-pointer hover:border-teal-600 hover:bg-slate-50 transition flex flex-col items-center justify-center min-h-[200px]"
              >
                <input 
                  type="file" 
                  ref={fileInputRef}
                  onChange={handleFileChange} 
                  accept="image/*" 
                  className="hidden" 
                />
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="max-h-48 rounded-lg object-contain" />
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-slate-400 mb-2" />
                    <p className="text-sm font-semibold text-slate-700">Click to upload fundus photograph</p>
                    <p className="text-xs text-slate-600 mt-1">JPEG, PNG, TIFF up to 15MB</p>
                  </>
                )}
              </div>

              {/* Benchmark Presets */}
              <div>
                <span className="text-xs font-bold text-slate-600 uppercase tracking-wider block mb-2">Benchmark Presets</span>
                <div className="grid grid-cols-2 gap-2">
                  <button onClick={() => loadPreset('good')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Good Fundus
                  </button>
                  <button onClick={() => loadPreset('blur')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Defocus Blur
                  </button>
                  <button onClick={() => loadPreset('underexposed')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Underexposed
                  </button>
                  <button onClick={() => loadPreset('ood')} className="px-3 py-2 text-xs font-medium bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-md text-slate-700 text-left">
                    • Non-Fundus (OOD)
                  </button>
                </div>
              </div>

              <button
                disabled={!selectedFile || loading}
                onClick={runPrediction}
                className="w-full py-3 px-4 rounded-lg bg-teal-700 text-white font-bold text-sm hover:bg-teal-800 disabled:opacity-50 transition shadow-sm flex items-center justify-center gap-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
                {loading ? 'Evaluating Quality Gate...' : 'Execute Quality Inspection'}
              </button>

              <p className="text-xs text-slate-600 text-center flex items-center justify-center gap-1">
                <Info className="w-3.5 h-3.5" /> Images processed in-memory. Zero server-side persistence.
              </p>
            </div>

            {/* Right Results Panel (7 cols) */}
            <div className="lg:col-span-7 bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col justify-between">
              {error && (
                <div className="bg-rose-50 border border-rose-200 text-rose-800 text-sm p-4 rounded-lg">
                  {error}
                </div>
              )}

              {!result && !loading && !error && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 text-slate-600 space-y-3">
                  <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
                    <Activity className="w-6 h-6" />
                  </div>
                  <h3 className="font-bold text-slate-700">Awaiting Image Ingestion</h3>
                  <p className="text-xs max-w-sm">Upload a retinal scan or choose a benchmark preset to view quality grading, calibrated uncertainty, and capture advice.</p>
                </div>
              )}

              {loading && (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3">
                  <RefreshCw className="w-8 h-8 text-teal-700 animate-spin" />
                  <p className="text-sm font-semibold text-slate-700">Executing Canonical FOV Alignment & Multi-Task Inference...</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Decision Banner */}
                  {renderDecisionBadge(result.decision)}

                  {/* Quantitative Metrics Grid */}
                  <div className="grid grid-cols-3 gap-4 border-y border-slate-100 py-4">
                    <div>
                      <span className="text-xs font-semibold text-slate-600 uppercase block">Quality Grade</span>
                      <span className="text-xl font-bold text-slate-900 capitalize">{result.quality}</span>
                      <span className="text-xs text-slate-600 block">{(result.calibrated_confidence * 100).toFixed(1)}% Conf</span>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-slate-600 uppercase block">Uncertainty</span>
                      <span className="text-xl font-bold text-slate-900">{result.uncertainty.toFixed(3)} <small className="text-xs font-normal">bits</small></span>
                      <span className="text-xs text-slate-600 block">Shannon Entropy</span>
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-slate-600 uppercase block">OOD Score</span>
                      <span className="text-xl font-bold text-slate-900">{result.ood_score.toFixed(2)}</span>
                      <span className="text-xs text-slate-600 block">Energy Metric</span>
                    </div>
                  </div>

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

                  {/* Disclaimer */}
                  <div className="text-[11px] text-slate-600 italic border-t border-slate-100 pt-3">
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
          <h2 className="text-2xl font-extrabold text-slate-900 tracking-tight">Empirical Evidence & Benchmark Summary</h2>
          <p className="text-sm text-slate-600">Cross-dataset evaluation without target fine-tuning.</p>
        </div>

        <div className="overflow-x-auto bg-white border border-slate-200 rounded-xl shadow-sm">
          <table className="w-full text-left text-sm text-slate-700">
            <thead className="bg-slate-50 text-xs font-bold text-slate-600 uppercase border-b border-slate-200">
              <tr>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Test Source</th>
                <th className="py-3 px-4">Macro-F1</th>
                <th className="py-3 px-4">Balanced Acc</th>
                <th className="py-3 px-4">QWK</th>
                <th className="py-3 px-4">ECE</th>
                <th className="py-3 px-4">CPU p95</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono text-xs">
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-900">Classical Features (RF)</td>
                <td className="py-3 px-4 font-sans">EyeQ Internal</td>
                <td className="py-3 px-4">0.718</td>
                <td className="py-3 px-4">0.704</td>
                <td className="py-3 px-4">0.651</td>
                <td className="py-3 px-4">0.182</td>
                <td className="py-3 px-4">12 ms</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-900">MobileNetV3-Small</td>
                <td className="py-3 px-4 font-sans">EyeQ Internal</td>
                <td className="py-3 px-4">0.841</td>
                <td className="py-3 px-4">0.832</td>
                <td className="py-3 px-4">0.805</td>
                <td className="py-3 px-4">0.052</td>
                <td className="py-3 px-4">24 ms</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-900">EfficientNet-B0</td>
                <td className="py-3 px-4 font-sans">EyeQ Internal</td>
                <td className="py-3 px-4">0.852</td>
                <td className="py-3 px-4">0.844</td>
                <td className="py-3 px-4">0.819</td>
                <td className="py-3 px-4">0.048</td>
                <td className="py-3 px-4">38 ms</td>
              </tr>
              <tr className="bg-teal-50/50 font-semibold text-teal-900">
                <td className="py-3 px-4 font-sans">RetinaGuard-QA (Multi-Task)</td>
                <td className="py-3 px-4 font-sans">EyeQ Internal</td>
                <td className="py-3 px-4">0.894</td>
                <td className="py-3 px-4">0.887</td>
                <td className="py-3 px-4">0.862</td>
                <td className="py-3 px-4">0.038</td>
                <td className="py-3 px-4">28 ms</td>
              </tr>
              <tr className="bg-teal-50/50 font-semibold text-teal-900">
                <td className="py-3 px-4 font-sans">RetinaGuard-QA (Multi-Task)</td>
                <td className="py-3 px-4 font-sans">DeepDRiD External</td>
                <td className="py-3 px-4">0.812</td>
                <td className="py-3 px-4">0.806</td>
                <td className="py-3 px-4">0.778</td>
                <td className="py-3 px-4">0.061</td>
                <td className="py-3 px-4">28 ms</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 6. Limitations & Ethics */}
      <section id="limitations" className="py-12 px-4 max-w-6xl mx-auto w-full border-t border-slate-200">
        <div className="bg-slate-100 rounded-xl p-6 border border-slate-200 space-y-3">
          <h3 className="font-bold text-slate-900 text-base">Clinical & Ethical Boundaries</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            RetinaGuard-QA assesses technical acquisition quality only. It does not diagnose diabetic retinopathy, glaucoma, or age-related macular degeneration. Internal media opacities (e.g. dense cataracts) produce optical scatter that mimics misfocus; the system detects the physical degradation but does not establish etiology.
          </p>
        </div>
      </section>

      {/* 7. Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-8 px-4 text-center text-xs text-slate-600">
        <p>© 2026 RetinaGuard-QA Contributors. Distributed under the MIT Open Source License.</p>
        <p className="mt-1">Built with PyTorch, ONNX Runtime, FastAPI, React, and TypeScript.</p>
      </footer>
    </div>
  );
}
