// RetinaGuard-QA Frontend Controller

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropPrompt = document.getElementById('drop-zone-prompt');
  const previewContainer = document.getElementById('preview-container');
  const imagePreview = document.getElementById('image-preview');
  const btnClear = document.getElementById('btn-clear');
  const btnAnalyze = document.getElementById('btn-analyze');
  const presetButtons = document.querySelectorAll('.btn-preset');

  // Results Elements
  const idleState = document.getElementById('idle-state');
  const loaderState = document.getElementById('loader-state');
  const resultsContent = document.getElementById('results-content');
  const actionBanner = document.getElementById('action-banner');
  const actionPill = document.getElementById('action-pill');
  const actionTitle = document.getElementById('action-title');
  const actionSummary = document.getElementById('action-summary');
  const latencyBadge = document.getElementById('latency-badge');

  // Metrics Elements
  const resGrade = document.getElementById('res-grade');
  const resGradeConf = document.getElementById('res-grade-conf');
  const resScore = document.getElementById('res-score');
  const resEntropy = document.getElementById('res-entropy');
  const resEntropyStatus = document.getElementById('res-entropy-status');

  // Probabilities
  const barGood = document.getElementById('bar-good');
  const barUsable = document.getElementById('bar-usable');
  const barReject = document.getElementById('bar-reject');
  const pctGood = document.getElementById('pct-good');
  const pctUsable = document.getElementById('pct-usable');
  const pctReject = document.getElementById('pct-reject');

  // Lists
  const defectsList = document.getElementById('defects-list');
  const instructionsList = document.getElementById('instructions-list');

  // Export Buttons
  const btnExportPdf = document.getElementById('btn-export-pdf');
  const btnExportJson = document.getElementById('btn-export-json');

  let currentBlob = null;
  let currentAssessment = null;

  // Initialize Backend Health
  fetch('/api/v1/health')
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById('engine-status');
      if (data.status === 'online') {
        el.innerHTML = `<span class="pulse-dot"></span> ONNX Runtime (${data.runtime_device.toUpperCase()})`;
      }
    })
    .catch(() => {
      document.getElementById('engine-status').innerHTML = '<span class="pulse-dot" style="background:#ef4444"></span> Engine Offline';
    });

  // Drag and Drop Event Listeners
  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageFile(e.target.files[0]);
    }
  });

  btnClear.addEventListener('click', (e) => {
    e.stopPropagation();
    resetState();
  });

  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPEG, PNG, TIFF).');
      return;
    }
    currentBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Preset Image Generator (Canvas-Based Realistic Fundus & Defect Simulators)
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-preset');
      const canvas = document.createElement('canvas');
      canvas.width = 384;
      canvas.height = 384;
      const ctx = canvas.getContext('2d');

      // Black background
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, 384, 384);

      if (type === 'ood') {
        // Non-Fundus (Simulate Grayscale Chest X-Ray)
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(40, 40, 304, 304);
        ctx.fillStyle = '#64748b';
        ctx.fillRect(100, 100, 80, 180);
        ctx.fillRect(204, 100, 80, 180);
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 4;
        for (let i = 0; i < 6; i++) {
          ctx.beginPath();
          ctx.arc(192, 120 + i * 25, 60, 0, Math.PI);
          ctx.stroke();
        }
      } else {
        // Circular Fundus Mask
        ctx.save();
        ctx.beginPath();
        ctx.arc(192, 192, 165, 0, Math.PI * 2);
        ctx.closePath();
        ctx.clip();

        // Base retina color
        let baseGrad = ctx.createRadialGradient(192, 192, 20, 192, 192, 170);
        if (type === 'good') {
          baseGrad.addColorStop(0, '#f97316');
          baseGrad.addColorStop(0.6, '#ea580c');
          baseGrad.addColorStop(1, '#9a3412');
        } else if (type === 'blur') {
          baseGrad.addColorStop(0, '#ea580c');
          baseGrad.addColorStop(1, '#7c2d12');
        } else if (type === 'underexposed') {
          baseGrad.addColorStop(0, '#7c2d12');
          baseGrad.addColorStop(1, '#451a03');
        } else if (type === 'overexposed') {
          baseGrad.addColorStop(0, '#ffedd5');
          baseGrad.addColorStop(0.4, '#fed7aa');
          baseGrad.addColorStop(1, '#fb923c');
        }
        ctx.fillStyle = baseGrad;
        ctx.fillRect(0, 0, 384, 384);

        // Optic Disc
        ctx.beginPath();
        ctx.arc(120, 192, 28, 0, Math.PI * 2);
        ctx.fillStyle = (type === 'overexposed') ? '#ffffff' : '#fef08a';
        ctx.fill();

        // Fovea
        ctx.beginPath();
        ctx.arc(240, 192, 18, 0, Math.PI * 2);
        ctx.fillStyle = (type === 'underexposed') ? '#2e1065' : '#7c2d12';
        ctx.fill();

        // Retinal Blood Vessels
        if (type !== 'blur') {
          ctx.strokeStyle = '#991b1b';
          ctx.lineWidth = (type === 'overexposed') ? 1.5 : 3;
          ctx.beginPath();
          ctx.moveTo(120, 192);
          ctx.bezierCurveTo(140, 140, 200, 110, 280, 90);
          ctx.moveTo(120, 192);
          ctx.bezierCurveTo(140, 240, 200, 270, 280, 290);
          ctx.stroke();
        } else {
          // Defocus blur: soft faint lines
          ctx.filter = 'blur(6px)';
          ctx.strokeStyle = '#7f1d1d';
          ctx.lineWidth = 6;
          ctx.beginPath();
          ctx.moveTo(120, 192);
          ctx.bezierCurveTo(140, 140, 200, 110, 280, 90);
          ctx.stroke();
          ctx.filter = 'none';
        }

        ctx.restore();
      }

      canvas.toBlob((blob) => {
        handleImageFile(new File([blob], `preset_${type}.jpg`, { type: 'image/jpeg' }));
        // Automatically trigger analysis on preset selection
        analyzeImage();
      }, 'image/jpeg', 0.95);
    });
  });

  // Analyze Click
  btnAnalyze.addEventListener('click', analyzeImage);

  async function analyzeImage() {
    if (!currentBlob) return;

    idleState.classList.add('hidden');
    resultsContent.classList.add('hidden');
    loaderState.classList.remove('hidden');
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append('file', currentBlob);

    try {
      const resp = await fetch('/api/v1/inspect', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await resp.json();
      currentAssessment = data;
      renderResults(data);
    } catch (err) {
      alert(`Quality Assurance Error: ${err.message}`);
      idleState.classList.remove('hidden');
    } finally {
      loaderState.classList.add('hidden');
      btnAnalyze.disabled = false;
    }
  }

  function renderResults(data) {
    resultsContent.classList.remove('hidden');

    // Latency
    latencyBadge.textContent = `Latency: ${data.latency_ms.toFixed(1)} ms`;

    // Action Banner Theme
    actionBanner.className = 'action-banner';
    const decision = data.decision;
    if (decision === 'ACCEPT') {
      actionBanner.classList.add('banner-accept');
    } else if (decision === 'USABLE_WITH_WARNING') {
      actionBanner.classList.add('banner-warning');
    } else if (decision === 'RECAPTURE_WITH_GUIDANCE') {
      actionBanner.classList.add('banner-recapture');
    } else if (decision === 'MANUAL_REVIEW') {
      actionBanner.classList.add('banner-review');
    } else {
      actionBanner.classList.add('banner-ood');
    }

    actionPill.textContent = decision;
    actionTitle.textContent = data.action_title;
    actionSummary.textContent = data.action_summary;

    // Metrics Box
    resGrade.textContent = data.quality_grade;
    resGradeConf.textContent = `Confidence: ${(data.quality_grade_confidence * 100).toFixed(1)}%`;
    resScore.innerHTML = `${data.quality_score.toFixed(1)} <small>/100</small>`;

    resEntropy.innerHTML = `${data.predictive_entropy.toFixed(3)} <small>bits</small>`;
    resEntropyStatus.textContent = data.predictive_entropy > 0.85 ? 'High Uncertainty (Review Required)' : 'Calibrated Low Risk';

    // Probability Bars
    const pGood = (data.grade_probabilities.Good || 0) * 100;
    const pUsable = (data.grade_probabilities.Usable || 0) * 100;
    const pReject = (data.grade_probabilities.Reject || 0) * 100;

    barGood.style.width = `${pGood.toFixed(1)}%`;
    barUsable.style.width = `${pUsable.toFixed(1)}%`;
    barReject.style.width = `${pReject.toFixed(1)}%`;

    pctGood.textContent = `${pGood.toFixed(1)}%`;
    pctUsable.textContent = `${pUsable.toFixed(1)}%`;
    pctReject.textContent = `${pReject.toFixed(1)}%`;

    // Defect Checklist
    defectsList.innerHTML = '';
    if (data.detected_defects && data.detected_defects.length > 0) {
      data.detected_defects.forEach(d => {
        const item = document.createElement('div');
        item.className = 'defect-item';
        item.innerHTML = `
          <span class="defect-name">⚠️ ${d.name}</span>
          <span class="defect-prob">${(d.probability * 100).toFixed(1)}%</span>
        `;
        defectsList.appendChild(item);
      });
    } else {
      defectsList.innerHTML = `
        <div class="no-defects-msg">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          No severe physical or optical defects identified.
        </div>
      `;
    }

    // Operator Instructions
    instructionsList.innerHTML = '';
    if (data.actionable_instructions && data.actionable_instructions.length > 0) {
      data.actionable_instructions.forEach(inst => {
        const li = document.createElement('li');
        li.textContent = inst;
        instructionsList.appendChild(li);
      });
    }
  }

  // Export PDF Report
  btnExportPdf.addEventListener('click', async () => {
    if (!currentBlob) return;
    const formData = new FormData();
    formData.append('file', currentBlob);

    try {
      const resp = await fetch('/api/v1/report/pdf', {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) throw new Error('PDF generation failed');

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `RetinaGuard_Audit_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      alert(`Export Error: ${e.message}`);
    }
  });

  // Export JSON Report
  btnExportJson.addEventListener('click', async () => {
    if (!currentBlob) return;
    const formData = new FormData();
    formData.append('file', currentBlob);

    try {
      const resp = await fetch('/api/v1/report/json', {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) throw new Error('JSON export failed');

      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `RetinaGuard_Audit_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e) {
      alert(`Export Error: ${e.message}`);
    }
  });

  function resetState() {
    currentBlob = null;
    currentAssessment = null;
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    dropPrompt.classList.remove('hidden');
    btnAnalyze.disabled = true;
    resultsContent.classList.add('hidden');
    idleState.classList.remove('hidden');
    latencyBadge.textContent = 'Latency: -- ms';
    fileInput.value = '';
  }
});
