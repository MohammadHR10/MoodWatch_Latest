// Multimodal Emotion Analyzer Frontend
// Handles UI state for video (browser webcam + server-stream) and audio (mic recording + upload)

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // Elements
  const tabs = $$('.tab');
  const panels = $$('.tab-panel');

  // Video controls
  const startVideoBtn = $('#startVideoBtn');
  const stopVideoBtn = $('#stopVideoBtn');
  const videoTimerEl = $('#videoTimer');
  const videoFramesEl = $('#videoFrames');
  const videoEmotionEl = $('#videoEmotion');
  const videoConfidenceEl = $('#videoConfidence');
  const videoProgressBar = $('#videoProgress');
  const cameraSelect = $('#cameraSelect');
  const intervalSecondsEl = $('#intervalSeconds');
  const pauseSecondsEl = $('#pauseSeconds');
  const sessionCountEl = $('#sessionCount');
  const auDumpEl = $('#auDump');
  const auBarsEl = $('#auBars');
  const emotionReasonsEl = $('#emotionReasons');
  const emotionChangesEl = $('#emotionChanges');
  const serverStatus = $('#serverStatus');
  const backendBadge = $('#backendBadge');
  const videoRecBadge = $('#videoRecBadge');
  const audioRecBadge = $('#audioRecBadge');
  const auBadge = $('#auBadge');
  // Last-session results elements
  const lastResDominantEl = $('#lastResDominant');
  const lastResFacesEl = $('#lastResFaces');
  const lastResAUsEl = $('#lastResAUs');
  const lastResultJsonEl = $('#lastResultJson');

  // Browser webcam elements
  const browserWebcam = $('#browserWebcam');
  const serverStream = $('#serverStream');
  const webcamCanvas = $('#webcamCanvas');
  let browserStream = null;
  let browserAnalysisInterval = null;
  let browserFrameCount = 0;

  // Build/version marker for cache checks
  const BUILD_VERSION = '2025-12-10-01';
  try { console.log('[Frontend] Build', BUILD_VERSION); } catch {}
  // Placeholder for AU status
  let auPlaceholderEl;

  // Audio controls
  const startAudioBtn = $('#startAudioBtn');
  const stopAudioBtn = $('#stopAudioBtn');
  const audioTimerEl = $('#audioTimer');
  const audioStatusEl = $('#audioStatus');
  const audioEmotionEl = $('#audioEmotion');
  const audioConfidenceEl = $('#audioConfidence');
  const micSelect = $('#micSelect');
  const audioDurationEl = $('#audioDuration');
  const audioResultsContent = $('#audioResultsContent');
  const audioWaveCanvas = document.querySelector('#audioWave') || document.querySelector('#audioVisualizer');

  // Chart.js setup for emotion timeline
  let emotionChart;
  let auChart;
  let emotionPie;
  let emotionBars;      // current distribution (bar)
  let emotionMultiChart; // live multi-series timeline
  
  // Keep track of all chart instances to avoid reuse errors
  const chartInstances = {};
  
  // Store last valid AU data to persist after recording stops
  let lastAUData = null;

  // AU Analytics Dashboard data structures
  const auAnalytics = {
    history: [],        // Time series data: [{timestamp, aus: {AU01: 0.5, ...}}]
    stats: {},          // Per-AU statistics: {AU01: {sum, count, max, min, values: []}}
    totalFrames: 0,
    startTime: null
  };

  // Dashboard chart instances
  let auTimeSeriesChart = null;
  let auDistributionChart = null;
  let auTopChart = null;
  let auCorrelationChart = null;

  function initChart() {
    const ctx = document.getElementById('emotionChart');
    if (!ctx) return;

    // Destroy existing chart on this canvas before creating a new one
    if (chartInstances.emotionChart) {
      chartInstances.emotionChart.destroy();
    }
    
    chartInstances.emotionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Emotion confidence',
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,0.15)',
            fill: true,
            data: [],
            tension: 0.25,
            pointRadius: 0,
          },
        ],
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { display: false },
          y: { min: 0, max: 1, grid: { color: 'rgba(0,0,0,0.05)' } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: true },
        },
      },
    });

    const auCtx = document.getElementById('auChart');
    if (auCtx) {
      if (chartInstances.auChart) {
        chartInstances.auChart.destroy();
      }
      chartInstances.auChart = new Chart(auCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Intensity' } },
          },
          plugins: { legend: { display: true, position: 'bottom' } },
        },
      });
      auChart = chartInstances.auChart;
    }

    // Live current distribution (bar)
    const barsCtx = document.getElementById('emotionBars');
    if (barsCtx) {
      const labels = ['Happy','Sad','Surprise','Angry','Fear','Disgust','Neutral'];
      if (chartInstances.emotionBars) {
        chartInstances.emotionBars.destroy();
      }
      chartInstances.emotionBars = new Chart(barsCtx, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            label: 'Current score',
            data: new Array(labels.length).fill(0),
            backgroundColor: ['#22c55e','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#10b981','#94a3b8'],
          }],
        },
        options: {
          indexAxis: 'y',
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { min: 0, max: 1, ticks: { callback: (v)=> (v*100).toFixed(0)+'%' } },
          },
          plugins: { legend: { display: false } },
        },
      });
      emotionBars = chartInstances.emotionBars;
    }

    // Live multi-series timeline of emotion scores
    const multiCtx = document.getElementById('emotionMultiChart');
    if (multiCtx) {
      const KEYS = ['happiness','sadness','surprise','anger','fear','disgust'];
      const LABELS = { happiness:'Happy', sadness:'Sad', surprise:'Surprise', anger:'Angry', fear:'Fear', disgust:'Disgust' };
      const COLORS = { happiness:'#22c55e', sadness:'#3b82f6', surprise:'#f59e0b', anger:'#ef4444', fear:'#8b5cf6', disgust:'#10b981' };
      
      if (chartInstances.emotionMultiChart) {
        chartInstances.emotionMultiChart.destroy();
      }
      
      chartInstances.emotionMultiChart = new Chart(multiCtx, {
        type: 'line',
        data: {
          labels: [],
          datasets: KEYS.map((k)=> ({
            label: LABELS[k] || k,
            data: [],
            borderColor: COLORS[k] || '#999',
            backgroundColor: (COLORS[k] || '#999') + '26',
            tension: 0.2,
            pointRadius: 0,
            fill: false,
          })),
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Score' } },
          },
          plugins: { legend: { display: true, position: 'bottom' } },
        },
      });
      // Attach keys to chart instance for updates
      chartInstances.emotionMultiChart._emoKeys = KEYS;
      emotionMultiChart = chartInstances.emotionMultiChart;
    }

    const pieCtx = document.getElementById('emotionPie');
    if (pieCtx) {
      if (chartInstances.emotionPie) {
        chartInstances.emotionPie.destroy();
      }
      chartInstances.emotionPie = new Chart(pieCtx, {
        type: 'doughnut',
        data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: true, position: 'bottom' } },
        },
      });
    }
  }

  // Replace chart with multi-series timeline from session result
  function renderTimelineFromResult(timeline) {
    const ctx = document.getElementById('emotionChart');
    if (!ctx || !Array.isArray(timeline) || timeline.length === 0) return;

    // Collect available score keys
    const scoreKeys = new Set();
    const times = [];
    timeline.forEach((e) => {
      if (typeof e?.t === 'number') times.push(e.t);
      const s = e?.scores;
      if (s && typeof s === 'object') Object.keys(s).forEach((k) => scoreKeys.add(k));
    });
    // Preferred order and labels
    const PREFERRED = ['happiness', 'sadness', 'surprise', 'anger', 'fear', 'disgust'];
    const LABELS = {
      happiness: 'Happy', sadness: 'Sad', surprise: 'Surprise', anger: 'Angry', fear: 'Fear', disgust: 'Disgust'
    };
    const COLORS = {
      happiness: '#22c55e', // green
      sadness: '#3b82f6',   // blue
      surprise: '#f59e0b',  // amber
      anger: '#ef4444',     // red
      fear: '#8b5cf6',      // violet
      disgust: '#10b981',   // teal
    };

    const keys = PREFERRED.filter((k) => scoreKeys.has(k));
    const datasets = keys.map((k) => {
      const data = timeline.map((e) => {
        const v = e?.scores?.[k];
        return typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
      });
      const color = COLORS[k] || '#999999';
      return {
        label: LABELS[k] || k,
        data,
        borderColor: color,
        backgroundColor: color + '26', // ~0.15 alpha
        tension: 0.2,
        pointRadius: 0,
        fill: false,
      };
    });

    // Destroy old chart if any and recreate with new config
    try { emotionChart?.destroy?.(); } catch {}
    emotionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: times.map((t) => t.toFixed(1) + 's'),
        datasets,
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Time (s)' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Score' } },
        },
        plugins: {
          legend: { display: true, position: 'bottom' },
          tooltip: { enabled: true },
        },
      },
    });
  }

  // Build AU time series chart from timeline entries containing `aus`
  function renderAUTimelineFromResult(timeline) {
    const auCtx = document.getElementById('auChart');
    if (!auCtx || !Array.isArray(timeline) || !timeline.length) return;

    // Compute average intensity per AU across the session
    const auSums = {};
    const auCounts = {};
    const times = [];
    timeline.forEach((e) => {
      const t = typeof e?.t === 'number' ? e.t : null;
      if (t !== null) times.push(t);
      const aus = e?.aus || {};
      Object.entries(aus).forEach(([k, v]) => {
        if (typeof v === 'number') {
          auSums[k] = (auSums[k] || 0) + v;
          auCounts[k] = (auCounts[k] || 0) + 1;
        }
      });
    });
    const averages = Object.entries(auSums).map(([k, sum]) => [k, sum / (auCounts[k] || 1)]);
    // Pick top N AUs for clarity
    const TOP_N = 6;
    const topAUs = averages.sort((a, b) => b[1] - a[1]).slice(0, TOP_N).map((x) => x[0]);

    // Colors palette
    const palette = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#14b8a6', '#e11d48', '#0ea5e9'];
    const datasets = topAUs.map((auName, i) => {
      const color = palette[i % palette.length];
      const data = timeline.map((e) => {
        const v = e?.aus?.[auName];
        return typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
      });
      return {
        label: auName,
        data,
        borderColor: color,
        backgroundColor: color + '26',
        tension: 0.2,
        pointRadius: 0,
        fill: false,
      };
    });

    try { auChart?.destroy?.(); } catch {}
    auChart = new Chart(auCtx, {
      type: 'line',
      data: {
        labels: times.map((t) => t.toFixed(1) + 's'),
        datasets,
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'Time (s)' } },
          y: { min: 0, max: 1, title: { display: true, text: 'Intensity' } },
        },
        plugins: { legend: { display: true, position: 'bottom' } },
      },
    });
  }

  // Render donut chart of time per emotion and populate summary stats
  function renderEmotionSummary(summary = {}, segments = []) {
    const labels = Object.keys(summary.time_by_emotion || {});
    const values = labels.map((k) => summary.time_by_emotion[k]);
    const palette = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#14b8a6','#e11d48','#0ea5e9'];
    const colors = labels.map((_, i) => palette[i % palette.length]);

    if (emotionPie) {
      try { emotionPie.destroy(); } catch {}
      const ctx = document.getElementById('emotionPie');
      if (ctx) {
        emotionPie = new Chart(ctx, {
          type: 'doughnut',
          data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
          options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } },
        });
      }
    }

    // Quick stats
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('sumChanges', typeof summary.changes === 'number' ? String(summary.changes) : '—');
    set('sumAvgConf', typeof summary.avg_confidence === 'number' ? (summary.avg_confidence*100).toFixed(0)+'%' : '—');
    set('sumLowQ', typeof summary.low_quality_pct === 'number' ? summary.low_quality_pct.toFixed(0)+'%' : '—');
    set('sumDuration', typeof summary.total_duration === 'number' ? summary.total_duration.toFixed(1)+'s' : '—');

    // Segments table (top by duration)
    const tbody = document.querySelector('#segmentsTable tbody');
    if (tbody) {
      tbody.innerHTML = '';
      const top = (Array.isArray(segments) ? segments.slice() : []).sort((a,b)=> (b.duration||0) - (a.duration||0)).slice(0,8);
      top.forEach((s) => {
        const tr = document.createElement('tr');
        const cells = [s.start_t?.toFixed?.(1)+'s', s.end_t?.toFixed?.(1)+'s', String(s.label||'—'), (s.duration||0).toFixed(1)+'s', (typeof s.avg_confidence==='number'?(s.avg_confidence*100).toFixed(0)+'%':'—')];
        cells.forEach((c)=>{ const td = document.createElement('td'); td.textContent = c; tr.appendChild(td); });
        tbody.appendChild(tr);
      });
    }

    // Wire export button
    const btn = document.getElementById('exportCsvBtn');
    if (btn && !btn._hooked) {
      btn._hooked = true;
      btn.addEventListener('click', async () => {
        try {
          const res = await fetch('/get_timeline_export');
          if (!res.ok) throw new Error('Export failed with status ' + res.status);
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
          a.download = `session_timeline_${timestamp}.csv`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
          console.log('Session timeline exported successfully');
        } catch (e) {
          alert('Export failed: ' + e.message);
          console.error('Export error:', e);
        }
      });
    }
  }

  function pushChartPoint(conf) {
    if (!emotionChart) return;
    const maxPoints = 120; // last ~60s when polling every 500ms
    const labels = emotionChart.data.labels;
    const data = emotionChart.data.datasets[0].data;
    labels.push('');
    data.push(typeof conf === 'number' ? conf : 0);
    if (labels.length > maxPoints) {
      labels.shift();
      data.shift();
    }
    emotionChart.update('none');
  }

  // Diagnostics: check face backend availability and path
  async function updateBackendStatus() {
    try {
      const res = await fetch('/face_backend_status');
      if (!res.ok) return;
      const data = await res.json();
      if (backendBadge) {
        const ok = !!data.available;
        backendBadge.classList.remove('badge-warn', 'badge-ok');
        backendBadge.classList.add(ok ? 'badge-ok' : 'badge-warn');
        backendBadge.textContent = `Backend: ${data.backend || '—'}`;
        backendBadge.title = `Backend: ${data.backend}\n` +
          `FeatureExtraction: ${ok ? 'available' : 'not available'}\n` +
          (data.feature_extraction_path ? `Path: ${data.feature_extraction_path}` : '');
      }
    } catch (e) {
      // ignore
    }
  }

  async function checkBackend() {
    const out = $('#diagOutput');
    if (out) out.textContent = 'Checking backend...';
    try {
      const res = await fetch('/check_backend');
      const data = await res.json();
      if (out) {
        if (data.success) {
          out.textContent = `OK (code ${data.returncode})\nBackend: ${data.backend || ''}\n` +
            (data.path ? `Path: ${data.path}\n` : '') +
            (data.stdout ? `--- stdout ---\n${data.stdout}` : '');
        } else {
          out.textContent = `NOT OK${data.returncode !== undefined ? ` (code ${data.returncode})` : ''}\n` +
            (data.backend ? `Backend: ${data.backend}\n` : '') +
            (data.path ? `Path: ${data.path}\n` : '') +
            (data.error ? `Error: ${data.error}\n` : '') +
            (data.stderr ? `--- stderr ---\n${data.stderr}` : '');
        }
      }
      updateBackendStatus();
    } catch (e) {
      if (out) out.textContent = `Error: ${e}`;
    }
  }

  // Diagnostics: list available cameras from backend
  async function checkCameras() {
    const out = $('#diagOutput');
    if (out) out.textContent = 'Probing cameras...';
    try {
      const res = await fetch('/camera_status');
      const data = await res.json();
      if (out) out.textContent = JSON.stringify(data, null, 2);
      // Populate selector
      if (cameraSelect && Array.isArray(data.available_indices)) {
        cameraSelect.innerHTML = '';
        (data.available_indices.length ? data.available_indices : [0]).forEach((idx) => {
          const opt = document.createElement('option');
          opt.value = String(idx);
          opt.textContent = `Device ${idx}`;
          cameraSelect.appendChild(opt);
        });
      }
    } catch (e) {
      if (out) out.textContent = 'Camera check failed: ' + e;
    }
  }

  // Diagnostics: show macOS setup commands based on current error
  async function showMacSetup() {
    const out = $('#diagOutput');
    if (out) out.textContent = 'Collecting setup hints...';
    try {
      const res = await fetch('/openface_install_help');
      const data = await res.json();
      if (!data.success) {
        out && (out.textContent = 'Setup helper error: ' + (data.error || 'unknown error'));
        return;
      }
      const cmds = Array.isArray(data.commands) ? data.commands : [];
      const notes = Array.isArray(data.notes) ? data.notes : [];
      const fePath = data.feature_extraction_path || 'n/a';
      const sample = data.sample || {};
      const text = [
        `FeatureExtraction: ${fePath}`,
        notes.length ? ('Notes:\n- ' + notes.join('\n- ')) : '',
        'Run these in Terminal (zsh):',
        ...cmds.map((c) => '$ ' + c),
        sample.returncode !== null ? `\nLast check: code ${sample.returncode}\n${(sample.stderr||sample.stdout||'').slice(0,400)}` : ''
      ].filter(Boolean).join('\n\n');
      out && (out.textContent = text);
    } catch (e) {
      out && (out.textContent = 'Setup helper failed: ' + e);
    }
  }

  // AU visualization and reasoning
  function updateAUBars(aus) {
    if (!auBarsEl || !aus) return;
    
    // Filter and sort valid AU entries
    const entries = Object.entries(aus)
      .filter(([k, v]) => {
        // Must be a number and greater than 0
        return typeof v === 'number' && !isNaN(v) && v > 0.001;
      })
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12); // Show top 12 AUs
    
    // Store the last valid AU data if we have significant AUs
    if (entries.length > 0) {
      lastAUData = aus;
    }
    
    // Calculate AU summary statistics from ALL AUs (not just top 12)
    const allEntries = Object.entries(aus).filter(([, v]) => typeof v === 'number' && !isNaN(v));
    const activeAUs = allEntries.filter(([, v]) => v > 0.15); // Active threshold
    const avgIntensity = activeAUs.length > 0 
      ? activeAUs.reduce((sum, [, v]) => sum + v, 0) / activeAUs.length 
      : 0;
    const peakAU = entries.length > 0 ? entries[0] : null;
    
    // Always update summary stats (even if no bars to show)
    const activeCountEl = document.getElementById('auActiveCount');
    const avgIntensityEl = document.getElementById('auAvgIntensity');
    const peakNameEl = document.getElementById('auPeakName');
    const peakValueEl = document.getElementById('auPeakValue');
    
    if (activeCountEl) activeCountEl.textContent = activeAUs.length;
    if (avgIntensityEl) avgIntensityEl.textContent = Math.round(avgIntensity * 100) + '%';
    if (peakNameEl) {
      if (peakAU) {
        const shortName = peakAU[0].split('_')[0]; // Just show AU code like "AU12"
        peakNameEl.textContent = shortName;
        peakNameEl.style.fontSize = '12px';
      } else {
        peakNameEl.textContent = '—';
        peakNameEl.style.fontSize = '20px';
      }
    }
    if (peakValueEl) {
      peakValueEl.textContent = peakAU ? Math.round(peakAU[1] * 100) + '%' : '0%';
    }
    
    // Show placeholder if no significant AUs (but only if we haven't stored any data yet)
    if (entries.length === 0) {
      // If we have last valid AU data, don't show placeholder - keep showing the last data
      if (lastAUData && lastAUData !== aus) {
        // Re-render with the last valid data
        updateAUBars(lastAUData);
        return;
      }
      
      if (!auPlaceholderEl) {
        auPlaceholderEl = document.createElement('div');
        auPlaceholderEl.className = 'reason';
        auPlaceholderEl.style.cssText = 'text-align: center; padding: 20px; color: var(--subtext);';
        auPlaceholderEl.textContent = 'No significant facial muscle activity detected. Try making facial expressions.';
      }
      auBarsEl.innerHTML = '';
      auBarsEl.appendChild(auPlaceholderEl);
      return;
    }
    
    // Render AU bars
    auBarsEl.innerHTML = entries
      .map(([name, val]) => {
        const pct = Math.max(0, Math.min(100, Math.round(val * 100)));
        // Clean up AU name for better readability
        const cleanName = name.replace(/_/g, ' ');
        
        // Add intensity indicator with color
        let intensityLabel = '';
        let intensityColor = '';
        if (pct >= 70) {
          intensityLabel = 'Strong';
          intensityColor = '#10b981';
        } else if (pct >= 40) {
          intensityLabel = 'Moderate';
          intensityColor = '#3b82f6';
        } else if (pct >= 15) {
          intensityLabel = 'Mild';
          intensityColor = '#60a5fa';
        } else {
          intensityLabel = 'Weak';
          intensityColor = '#94a3b8';
        }
        
        return `<div class="au-bar">
          <div class="au-label" title="${cleanName}">${cleanName}</div>
          <div class="au-track"><div class="au-fill" style="width:${pct}%; background: linear-gradient(90deg, ${intensityColor}, #3b82f6);"></div></div>
          <div class="au-val" style="color: ${intensityColor};">${pct}% <span style="font-size:10px; opacity:0.7;">${intensityLabel}</span></div>
        </div>`;
      })
      .join('');
  }

  // Normalize backend emotion labels to UI categories
  function normalizeEmotion(e) {
    if (!e) return 'Neutral';
    const v = String(e).toLowerCase();
    if (v.includes('happy') || v === 'happiness') return 'Happy';
    if (v.includes('sad')) return 'Sad';
    if (v.includes('anger') || v.includes('angry')) return 'Angry';
    if (v.includes('surpris')) return 'Surprise';
    if (v.includes('disgust')) return 'Disgust';
    if (v.includes('fear') || v.includes('anx')) return 'Fear';
    if (v.includes('neutral')) return 'Neutral';
    return v.charAt(0).toUpperCase() + v.slice(1);
  }

  const EMOTION_AU_RULES = {
    Happy: {
      include: ['AU12_Lip_Corner_Puller', 'AU06_Cheek_Raiser'],
      inhibit: ['AU15_Lip_Corner_Depressor', 'AU04_Brow_Lowerer']
    },
    Sad: {
      include: ['AU15_Lip_Corner_Depressor', 'AU01_Inner_Brow_Raiser'],
      inhibit: ['AU12_Lip_Corner_Puller']
    },
    Angry: {
      include: ['AU04_Brow_Lowerer', 'AU07_Lid_Tightener'],
      inhibit: []
    },
    Surprise: {
      include: ['AU05_Upper_Lid_Raiser', 'AU02_Outer_Brow_Raiser', 'AU26_Jaw_Drop'],
      inhibit: []
    },
    Disgust: {
      include: ['AU09_Nose_Wrinkler', 'AU10_Upper_Lip_Raiser'],
      inhibit: ['AU12_Lip_Corner_Puller']
    },
    Fear: {
      include: ['AU01_Inner_Brow_Raiser', 'AU05_Upper_Lid_Raiser', 'AU20_Lip_Stretcher'],
      inhibit: []
    },
    Neutral: { include: [], inhibit: [] }
  };

  function explainEmotion(emotion, aus) {
    if (!emotionReasonsEl) return;
    const rules = EMOTION_AU_RULES[emotion] || EMOTION_AU_RULES.Neutral;
    const reasons = [];
    const get = (k) => (typeof aus?.[k] === 'number' ? aus[k] : 0);
    const fmt = (k) => `${k.split('_')[0]} ${(get(k) * 100).toFixed(0)}%`;

    // Only show if there are meaningful AUs
    const hasAnySignificantAU = Object.values(aus || {}).some(v => typeof v === 'number' && v > 0.15);
    
    if (!hasAnySignificantAU) {
      emotionReasonsEl.innerHTML = '<div class="reason" style="text-align: center; color: var(--subtext); font-style: italic;">Waiting for facial expressions...</div>';
      return;
    }

    if (rules.include?.length) {
      const hits = rules.include.filter((k) => get(k) > 0.25);
      if (hits.length) {
        reasons.push(`✓ Active AUs supporting ${emotion}: ${hits.map(fmt).join(', ')}`);
      }
    }
    
    if (rules.inhibit?.length) {
      const hits = rules.inhibit.filter((k) => get(k) > 0.25);
      if (hits.length) {
        reasons.push(`⚠ Conflicting AUs detected: ${hits.map(fmt).join(', ')}`);
      }
    }
    
    if (!reasons.length) {
      // Show top 3 active AUs instead of generic message
      const topAUs = Object.entries(aus || {})
        .filter(([, v]) => typeof v === 'number' && v > 0.15)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([k, v]) => `${k.split('_')[0]} ${(v * 100).toFixed(0)}%`);
      
      if (topAUs.length > 0) {
        reasons.push(`Most active AUs: ${topAUs.join(', ')}`);
      } else {
        reasons.push('Minimal facial muscle activity detected.');
      }
    }
    
    emotionReasonsEl.innerHTML = reasons.map((r) => `<div class="reason">${r}</div>`).join('');
  }

  // ========== AU Analytics Dashboard Functions ==========
  
  function initAUAnalyticsDashboard() {
    // Initialize charts
    initAUTimeSeriesChart();
    initAUDistributionChart();
    initAUTopChart();
    initAUCorrelationChart();
    
    // Event listeners
    const exportBtn = document.getElementById('exportAUData');
    const resetBtn = document.getElementById('resetAUStats');
    
    if (exportBtn) {
      exportBtn.addEventListener('click', exportAUAnalyticsData);
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', resetAUAnalytics);
    }
  }

  function initAUTimeSeriesChart() {
    const ctx = document.getElementById('auTimeSeriesChart');
    if (!ctx) return;
    
    if (auTimeSeriesChart) {
      auTimeSeriesChart.destroy();
    }
    
    auTimeSeriesChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: []
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        scales: {
          x: {
            title: { display: true, text: 'Time (s)', color: '#9ca3af' },
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            min: 0,
            max: 1,
            title: { display: true, text: 'Intensity', color: '#9ca3af' },
            ticks: { 
              color: '#9ca3af',
              callback: (v) => (v * 100).toFixed(0) + '%'
            },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
        },
        plugins: {
          legend: {
            display: true,
            position: 'bottom',
            labels: { 
              color: '#9ca3af',
              boxWidth: 12,
              font: { size: 11 }
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            callbacks: {
              label: (context) => {
                return `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(1)}%`;
              }
            }
          }
        },
      },
    });
  }

  function initAUDistributionChart() {
    const ctx = document.getElementById('auDistributionChart');
    if (!ctx) return;
    
    if (auDistributionChart) {
      auDistributionChart.destroy();
    }
    
    auDistributionChart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: []
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'right',
            labels: { 
              color: '#9ca3af',
              font: { size: 10 },
              boxWidth: 12
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            callbacks: {
              label: (context) => {
                const label = context.label || '';
                const value = context.parsed || 0;
                return `${label}: ${value.toFixed(1)}%`;
              }
            }
          }
        },
      },
    });
  }

  function initAUTopChart() {
    const ctx = document.getElementById('auTopChart');
    if (!ctx) return;
    
    if (auTopChart) {
      auTopChart.destroy();
    }
    
    auTopChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [],
        datasets: [{
          label: 'Average Intensity',
          data: [],
          backgroundColor: []
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            min: 0,
            max: 1,
            ticks: { 
              color: '#9ca3af',
              callback: (v) => (v * 100).toFixed(0) + '%'
            },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            ticks: { color: '#9ca3af', font: { size: 11 } },
            grid: { display: false }
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(0,0,0,0.8)',
            callbacks: {
              label: (context) => {
                return `Avg: ${(context.parsed.x * 100).toFixed(1)}%`;
              }
            }
          }
        },
      },
    });
  }

  function initAUCorrelationChart() {
    const ctx = document.getElementById('auCorrelationChart');
    if (!ctx) return;
    
    if (auCorrelationChart) {
      auCorrelationChart.destroy();
    }
    
    // This will be a custom-drawn heatmap
    auCorrelationChart = { ctx: ctx.getContext('2d'), canvas: ctx };
  }

  function updateAUAnalytics(aus) {
    if (!aus || typeof aus !== 'object') return;
    
    // Record data point
    const timestamp = auAnalytics.startTime ? (Date.now() - auAnalytics.startTime) / 1000 : 0;
    auAnalytics.history.push({ timestamp, aus: { ...aus } });
    auAnalytics.totalFrames++;
    
    // Keep last 5 minutes of data (assuming ~2 fps)
    if (auAnalytics.history.length > 600) {
      auAnalytics.history.shift();
    }
    
    // Update per-AU statistics
    Object.entries(aus).forEach(([auName, value]) => {
      if (typeof value !== 'number' || isNaN(value)) return;
      
      if (!auAnalytics.stats[auName]) {
        auAnalytics.stats[auName] = {
          sum: 0,
          count: 0,
          max: 0,
          min: Infinity,
          values: []
        };
      }
      
      const stat = auAnalytics.stats[auName];
      stat.sum += value;
      stat.count++;
      stat.max = Math.max(stat.max, value);
      stat.min = Math.min(stat.min, value);
      stat.values.push(value);
      
      // Keep last 100 values for std dev calculation
      if (stat.values.length > 100) {
        stat.values.shift();
      }
    });
    
    // Update dashboard every 10 frames to reduce overhead
    if (auAnalytics.totalFrames % 10 === 0) {
      updateAUDashboardCharts();
      updateAUStatsTable();
      updateAUStatsCards();
    }
  }

  function updateAUDashboardCharts() {
    updateAUTimeSeriesChartData();
    updateAUDistributionChartData();
    updateAUTopChartData();
    updateAUCorrelationHeatmap();
  }

  function updateAUTimeSeriesChartData() {
    if (!auTimeSeriesChart || auAnalytics.history.length === 0) return;
    
    // Get top 6 AUs by average intensity
    const topAUs = getTopAUs(6);
    const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#10b981'];
    
    // Build datasets
    const labels = auAnalytics.history.map(h => h.timestamp.toFixed(1));
    const datasets = topAUs.map((auName, i) => ({
      label: auName.split('_')[0],
      data: auAnalytics.history.map(h => h.aus[auName] || 0),
      borderColor: colors[i % colors.length],
      backgroundColor: (colors[i % colors.length]) + '40',
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2,
      fill: false,
    }));
    
    auTimeSeriesChart.data.labels = labels;
    auTimeSeriesChart.data.datasets = datasets;
    auTimeSeriesChart.update('none');
  }

  function updateAUDistributionChartData() {
    if (!auDistributionChart || Object.keys(auAnalytics.stats).length === 0) return;
    
    // Calculate average intensity for each AU
    const auAverages = Object.entries(auAnalytics.stats)
      .map(([name, stat]) => ({
        name: name.split('_')[0],
        avg: stat.sum / stat.count
      }))
      .filter(au => au.avg > 0.05)
      .sort((a, b) => b.avg - a.avg)
      .slice(0, 8);
    
    const colors = [
      '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', 
      '#8b5cf6', '#10b981', '#e11d48', '#0ea5e9'
    ];
    
    auDistributionChart.data.labels = auAverages.map(au => au.name);
    auDistributionChart.data.datasets[0].data = auAverages.map(au => au.avg * 100);
    auDistributionChart.data.datasets[0].backgroundColor = colors;
    auDistributionChart.update('none');
  }

  function updateAUTopChartData() {
    if (!auTopChart || Object.keys(auAnalytics.stats).length === 0) return;
    
    const topAUs = getTopAUs(10);
    const auData = topAUs.map(name => {
      const stat = auAnalytics.stats[name];
      return {
        name: name.split('_')[0],
        avg: stat.sum / stat.count
      };
    });
    
    const colors = auData.map((_, i) => {
      const intensity = auData[i].avg;
      if (intensity > 0.5) return '#10b981';
      if (intensity > 0.3) return '#3b82f6';
      if (intensity > 0.15) return '#f59e0b';
      return '#6b7280';
    });
    
    auTopChart.data.labels = auData.map(au => au.name);
    auTopChart.data.datasets[0].data = auData.map(au => au.avg);
    auTopChart.data.datasets[0].backgroundColor = colors;
    auTopChart.update('none');
  }

  function updateAUCorrelationHeatmap() {
    if (!auCorrelationChart || Object.keys(auAnalytics.stats).length === 0) return;
    
    const topAUs = getTopAUs(10);
    if (topAUs.length < 2) return;
    
    // Calculate correlation matrix
    const correlations = [];
    for (let i = 0; i < topAUs.length; i++) {
      correlations[i] = [];
      for (let j = 0; j < topAUs.length; j++) {
        correlations[i][j] = calculateCorrelation(topAUs[i], topAUs[j]);
      }
    }
    
    // Draw heatmap
    const ctx = auCorrelationChart.ctx;
    const canvas = auCorrelationChart.canvas;
    const cellSize = Math.min(canvas.width / (topAUs.length + 2), 30);
    const startX = 60;
    const startY = 30;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw cells
    for (let i = 0; i < topAUs.length; i++) {
      for (let j = 0; j < topAUs.length; j++) {
        const corr = correlations[i][j];
        const intensity = Math.abs(corr);
        const color = corr >= 0 
          ? `rgba(34, 197, 94, ${intensity})`  // Green for positive
          : `rgba(239, 68, 68, ${intensity})`; // Red for negative
        
        ctx.fillStyle = color;
        ctx.fillRect(startX + j * cellSize, startY + i * cellSize, cellSize - 1, cellSize - 1);
        
        // Draw value if strong correlation
        if (Math.abs(corr) > 0.3) {
          ctx.fillStyle = intensity > 0.6 ? '#fff' : '#9ca3af';
          ctx.font = '10px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            corr.toFixed(2),
            startX + j * cellSize + cellSize / 2,
            startY + i * cellSize + cellSize / 2
          );
        }
      }
    }
    
    // Draw labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    topAUs.forEach((au, i) => {
      const label = au.split('_')[0];
      // Row labels
      ctx.fillText(label, startX - 5, startY + i * cellSize + cellSize / 2);
      // Column labels (rotated)
      ctx.save();
      ctx.translate(startX + i * cellSize + cellSize / 2, startY - 5);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'right';
      ctx.fillText(label, 0, 0);
      ctx.restore();
    });
  }

  function updateAUStatsTable() {
    const tbody = document.getElementById('auStatsTableBody');
    if (!tbody || Object.keys(auAnalytics.stats).length === 0) return;
    
    // Sort AUs by average intensity
    const sortedAUs = Object.entries(auAnalytics.stats)
      .map(([name, stat]) => ({
        name,
        mean: stat.sum / stat.count,
        max: stat.max,
        stdDev: calculateStdDev(stat.values),
        frequency: stat.count / auAnalytics.totalFrames,
        trend: calculateTrend(stat.values)
      }))
      .sort((a, b) => b.mean - a.mean);
    
    tbody.innerHTML = sortedAUs.map(au => {
      const trendIcon = au.trend > 0.05 ? '📈' : au.trend < -0.05 ? '📉' : '➡️';
      const trendColor = au.trend > 0.05 ? '#10b981' : au.trend < -0.05 ? '#ef4444' : '#6b7280';
      
      return `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
          <td style="padding: 10px; font-weight: 500;">${au.name.split('_')[0]}</td>
          <td style="padding: 10px; text-align: right; color: #3b82f6;">${(au.mean * 100).toFixed(1)}%</td>
          <td style="padding: 10px; text-align: right; color: #10b981;">${(au.max * 100).toFixed(1)}%</td>
          <td style="padding: 10px; text-align: right; color: #9ca3af;">${(au.stdDev * 100).toFixed(1)}%</td>
          <td style="padding: 10px; text-align: right; color: #f59e0b;">${(au.frequency * 100).toFixed(0)}%</td>
          <td style="padding: 10px; text-align: center; font-size: 16px; color: ${trendColor};">${trendIcon}</td>
        </tr>
      `;
    }).join('');
  }

  function updateAUStatsCards() {
    const totalFramesEl = document.getElementById('auStatTotalFrames');
    const activeCountEl = document.getElementById('auStatActiveCount');
    const avgIntensityEl = document.getElementById('auStatAvgIntensity');
    const peakAUEl = document.getElementById('auStatPeakAU');
    const peakValueEl = document.getElementById('auStatPeakValue');
    
    if (totalFramesEl) totalFramesEl.textContent = auAnalytics.totalFrames;
    
    // Calculate overall metrics
    const allMeans = Object.values(auAnalytics.stats).map(s => s.sum / s.count);
    const activeCount = allMeans.filter(m => m > 0.15).length;
    const overallAvg = allMeans.length > 0 ? allMeans.reduce((a, b) => a + b, 0) / allMeans.length : 0;
    
    // Find peak AU
    let peakAU = null;
    let peakValue = 0;
    Object.entries(auAnalytics.stats).forEach(([name, stat]) => {
      if (stat.max > peakValue) {
        peakValue = stat.max;
        peakAU = name;
      }
    });
    
    if (activeCountEl) activeCountEl.textContent = activeCount;
    if (avgIntensityEl) avgIntensityEl.textContent = Math.round(overallAvg * 100) + '%';
    if (peakAUEl && peakAU) {
      peakAUEl.textContent = peakAU.split('_')[0];
      peakAUEl.style.fontSize = '24px';
    }
    if (peakValueEl) peakValueEl.textContent = Math.round(peakValue * 100) + '%';
  }

  // Helper functions
  function getTopAUs(count) {
    return Object.entries(auAnalytics.stats)
      .map(([name, stat]) => ({ name, avg: stat.sum / stat.count }))
      .sort((a, b) => b.avg - a.avg)
      .slice(0, count)
      .map(item => item.name);
  }

  function calculateCorrelation(au1, au2) {
    const values1 = auAnalytics.history.map(h => h.aus[au1] || 0);
    const values2 = auAnalytics.history.map(h => h.aus[au2] || 0);
    
    if (values1.length < 2) return 0;
    
    const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
    const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
    
    let numerator = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;
    
    for (let i = 0; i < values1.length; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;
      numerator += diff1 * diff2;
      sumSq1 += diff1 * diff1;
      sumSq2 += diff2 * diff2;
    }
    
    const denominator = Math.sqrt(sumSq1 * sumSq2);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  function calculateStdDev(values) {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squareDiffs = values.map(v => Math.pow(v - mean, 2));
    const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
  }

  function calculateTrend(values) {
    if (values.length < 10) return 0;
    
    // Simple linear regression slope
    const recentValues = values.slice(-20);
    const n = recentValues.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += recentValues[i];
      sumXY += i * recentValues[i];
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  function exportAUAnalyticsData() {
    const data = {
      metadata: {
        totalFrames: auAnalytics.totalFrames,
        duration: auAnalytics.history.length > 0 
          ? auAnalytics.history[auAnalytics.history.length - 1].timestamp 
          : 0,
        exportDate: new Date().toISOString()
      },
      statistics: auAnalytics.stats,
      history: auAnalytics.history
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `au-analytics-${new Date().getTime()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log('AU analytics data exported');
  }

  function resetAUAnalytics() {
    if (!confirm('Reset all AU analytics data? This cannot be undone.')) return;
    
    auAnalytics.history = [];
    auAnalytics.stats = {};
    auAnalytics.totalFrames = 0;
    auAnalytics.startTime = null;
    
    updateAUDashboardCharts();
    updateAUStatsTable();
    updateAUStatsCards();
    
    // Clear table
    const tbody = document.getElementById('auStatsTableBody');
    if (tbody) {
      tbody.innerHTML = '<tr><td colspan="6" style="padding: 20px; text-align: center; color: var(--subtext);">No AU data available yet. Start recording to see statistics.</td></tr>';
    }
    
    console.log('AU analytics reset');
  }

  // Track and render emotion changes during a session
  let lastEmotion = null;
  function trackEmotionChange(currEmotion) {
    if (!emotionChangesEl) return;
    if (currEmotion && currEmotion !== lastEmotion) {
      const t = new Date();
      const hh = String(t.getHours()).padStart(2, '0');
      const mm = String(t.getMinutes()).padStart(2, '0');
      const ss = String(t.getSeconds()).padStart(2, '0');
      const timeStr = `${hh}:${mm}:${ss}`;
      const div = document.createElement('div');
      div.className = 'change';
      div.innerHTML = `<span class="time">${timeStr}</span><span>→</span><span class=\"label\">${currEmotion}</span>`;
      emotionChangesEl.prepend(div);
      const nodes = emotionChangesEl.querySelectorAll('.change');
      if (nodes.length > 10) {
        nodes[nodes.length - 1].remove();
      }
      lastEmotion = currEmotion;
    }
  }

  // Init
  document.addEventListener('DOMContentLoaded', () => {
    updateBackendStatus();
    const btn = document.getElementById('checkBackendBtn');
    if (btn) btn.addEventListener('click', checkBackend);
  });

  // Format helpers
  const fmtTime = (s) => {
    s = Math.max(0, Math.floor(s));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
  };

  // Video state
  const videoState = {
    isRecording: false,
    startTs: 0,
    duration: 0,
    pollId: null,
    timerId: null,
    useBrowserWebcam: false,
  };

  // Browser webcam functions
  async function startBrowserWebcam() {
    try {
      browserStream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false 
      });
      browserWebcam.srcObject = browserStream;
      browserWebcam.style.display = 'block';
      serverStream.style.display = 'none';
      await browserWebcam.play();
      console.log('[Webcam] Browser webcam started');
      return true;
    } catch (err) {
      console.error('[Webcam] Failed to start browser webcam:', err);
      alert('Failed to access webcam: ' + err.message);
      return false;
    }
  }

  function stopBrowserWebcam() {
    if (browserStream) {
      browserStream.getTracks().forEach(t => t.stop());
      browserStream = null;
    }
    if (browserAnalysisInterval) {
      clearInterval(browserAnalysisInterval);
      browserAnalysisInterval = null;
    }
    browserWebcam.style.display = 'none';
  }

  async function captureAndAnalyzeFrame() {
    if (!browserStream || !browserWebcam.videoWidth) return;
    
    const canvas = webcamCanvas;
    canvas.width = browserWebcam.videoWidth;
    canvas.height = browserWebcam.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(browserWebcam, 0, 0);
    
    // Convert to base64 JPEG
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    const base64 = dataUrl.split(',')[1];
    
    try {
      const res = await fetch('/analyze_browser_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      });
      const data = await res.json();
      browserFrameCount++;
      
      if (data.success && data.face_detected) {
        videoEmotionEl.textContent = data.dominant_emotion || '—';
        videoConfidenceEl.textContent = typeof data.confidence === 'number' 
          ? `${(data.confidence * 100).toFixed(0)}%` : '—';
        videoFramesEl.textContent = `${browserFrameCount} frames`;
        
        // Update AU display
        if (data.aus && Object.keys(data.aus).length > 0) {
          auBadge.textContent = `AUs: ${Object.keys(data.aus).length}`;
          updateAUBars(data.aus);
          updateAUAnalytics(data.aus);
        }
        
        // Update emotion bars
        if (data.emotions && emotionBars) {
          const labels = emotionBars.data.labels;
          const newData = labels.map(l => data.emotions[l.toLowerCase()] || 0);
          emotionBars.data.datasets[0].data = newData;
          emotionBars.update('none');
        }
      } else if (data.success) {
        videoEmotionEl.textContent = 'No Face';
      }
    } catch (err) {
      console.error('[Webcam] Analysis error:', err);
    }
  }

  async function startVideoRecording() {
    if (videoState.isRecording) return;
    const cameraMode = cameraSelect ? cameraSelect.value : 'browser';
    const intervalSeconds = parseInt(intervalSecondsEl.value || '10', 10);

    startVideoBtn.disabled = true;
    stopVideoBtn.disabled = false;
    videoRecBadge.className = 'badge badge-live';
    videoRecBadge.textContent = 'Video: Recording';
    serverStatus.className = 'badge badge-live';
    serverStatus.textContent = 'Live';
    
    // Initialize AU Analytics tracking
    auAnalytics.startTime = Date.now();
    browserFrameCount = 0;

    // Check if using browser webcam (for cloud deployment)
    if (cameraMode === 'browser') {
      videoState.useBrowserWebcam = true;
      const started = await startBrowserWebcam();
      if (!started) {
        stopVideoUI();
        startVideoBtn.disabled = false;
        return;
      }
      
      videoState.isRecording = true;
      videoState.startTs = Date.now();
      videoState.duration = intervalSeconds;
      
      // Timer UI
      videoState.timerId = setInterval(() => {
        const elapsedSec = (Date.now() - videoState.startTs) / 1000;
        videoTimerEl.textContent = fmtTime(elapsedSec);
        const pct = Math.min(100, (elapsedSec / videoState.duration) * 100);
        videoProgressBar.style.width = `${pct.toFixed(1)}%`;
        
        // Auto-stop after duration
        if (elapsedSec >= videoState.duration) {
          stopVideoRecording();
        }
      }, 300);
      
      // Analyze frames at ~2-3 FPS (every 400ms)
      browserAnalysisInterval = setInterval(captureAndAnalyzeFrame, 400);
      startVideoBtn.disabled = false;
      return;
    }

    // Server-side camera mode (local dev only)
    videoState.useBrowserWebcam = false;
    const device = parseInt(cameraMode || '0', 10);
    const reqBody = { duration: intervalSeconds };

    try {
      // Show server stream
      browserWebcam.style.display = 'none';
      serverStream.style.display = 'block';
      serverStream.src = `/video_feed?device=${encodeURIComponent(device)}`;
      
      const res = await fetch('/start_recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reqBody),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Failed to start');

      videoState.isRecording = true;
      videoState.startTs = Date.now();
      videoState.duration = data.duration || intervalSeconds;

      // Timer UI
      videoState.timerId = setInterval(() => {
        const elapsedSec = (Date.now() - videoState.startTs) / 1000;
        videoTimerEl.textContent = fmtTime(elapsedSec);
        const pct = Math.min(100, (elapsedSec / videoState.duration) * 100);
        videoProgressBar.style.width = `${pct.toFixed(1)}%`;
      }, 300);

      // Poll telemetry
      videoState.pollId = setInterval(updateVideoTelemetry, 500);
    } catch (err) {
      console.error(err);
      stopVideoUI();
      const out = document.getElementById('diagOutput');
      if (out) out.textContent = 'Failed to start video recording: ' + err.message;
      alert('Failed to start video recording: ' + err.message);
    } finally {
      startVideoBtn.disabled = false;
    }
  }

  async function updateVideoTelemetry() {
    try {
      const res = await fetch('/get_realtime_data');
      const data = await res.json();
      if (!data || !videoState.isRecording) return;

  const { emotion, emotion_confidence, frame_count, action_units, progress, au_count, landmark_quality } = data;
      const emotionScores = data.emotion_scores || null;
      const timeline = Array.isArray(data.timeline) ? data.timeline : [];
      const top_aus = data.top_aus || [];
      const lastError = data.error;
      const normEmotion = normalizeEmotion(emotion);
      videoEmotionEl.textContent = normEmotion || '—';
      videoConfidenceEl.textContent = typeof emotion_confidence === 'number' ? `${(emotion_confidence * 100).toFixed(0)}%` : '—';
      videoFramesEl.textContent = `${frame_count || 0} frames`;
      if (typeof progress === 'number' && videoState.duration > 0) {
        videoProgressBar.style.width = `${(progress * 100).toFixed(1)}%`;
      }

      if (lastError) {
        const out = document.getElementById('diagOutput');
        if (out && !out.textContent.includes(lastError)) {
          out.textContent = `Analysis error: ${lastError}`;
        }
        updateBackendStatus();
      }

      if (action_units) {
        try { auDumpEl.textContent = JSON.stringify(action_units, null, 2); } catch { /* ignore */ }
        updateAUBars(action_units);
        explainEmotion(normEmotion || 'Neutral', action_units);
        
        // Feed data to AU Analytics Dashboard
        updateAUAnalytics(action_units);
        
        if (typeof au_count === 'number') {
          // Lightweight console debug for AU pipeline
          if (au_count === 0) {
            console.debug('Realtime: 0 AUs in payload — waiting for face/lighting.');
          }
          if (auBadge) {
            auBadge.textContent = `AUs: ${au_count}`;
            auBadge.className = `badge ${au_count > 0 ? 'badge-live' : 'badge-muted'}`;
          }
        }
      }

      // Live current emotion distribution bar chart
      if (emotionBars && emotionScores) {
        const order = ['happiness','sadness','surprise','anger','fear','disgust','neutral'];
        const vals = order.map((k) => {
          const v = emotionScores?.[k];
          return typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
        });
        emotionBars.data.datasets[0].data = vals;
        emotionBars.update('none');
      }

      // Live multi-series emotion timeline chart (all emotions as separate series)
      if (emotionMultiChart && emotionScores && timeline.length > 0) {
        const MAX_POINTS = 120; // ~60s if polling every 500ms
        // Initialize chart if needed
        if (!emotionMultiChart.data.labels || emotionMultiChart.data.labels.length === 0) {
          const order = ['happiness', 'sadness', 'surprise', 'anger', 'fear', 'disgust', 'neutral'];
          const colors = {
            happiness: '#22c55e', sadness: '#3b82f6', surprise: '#f59e0b',
            anger: '#ef4444', fear: '#8b5cf6', disgust: '#10b981', neutral: '#9ca3af',
          };
          emotionMultiChart.data.labels = [];
          emotionMultiChart.data.datasets = order.map((emoKey, i) => ({
            label: emoKey.charAt(0).toUpperCase() + emoKey.slice(1),
            data: [],
            borderColor: colors[emoKey] || '#999',
            backgroundColor: (colors[emoKey] || '#999') + '26',
            tension: 0.2,
            pointRadius: 0,
            fill: false,
          }));
          emotionMultiChart._emoKeys = order;
        }

        // Add new data point from latest timeline entry
        const latest = timeline[timeline.length - 1];
        if (latest) {
          const timeLabel = latest.t.toFixed(1) + 's';
          emotionMultiChart.data.labels.push(timeLabel);
          
          (emotionMultiChart._emoKeys || []).forEach((k, i) => {
            const v = latest.scores?.[k];
            const val = typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
            const ds = emotionMultiChart.data.datasets[i];
            ds.data.push(val);
            if (ds.data.length > MAX_POINTS) ds.data.shift();
          });
          
          if (emotionMultiChart.data.labels.length > MAX_POINTS) {
            emotionMultiChart.data.labels.shift();
          }
          emotionMultiChart.update('none');
        }
      }

  // Live AU timeline chart (top 6-8 AUs)
      if (auChart && timeline.length > 0) {
        const MAX_POINTS = 120;
        // Initialize if needed
        if (!auChart.data.labels || auChart.data.labels.length === 0) {
          const palette = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#14b8a6', '#e11d48', '#0ea5e9'];
          const topAUNames = top_aus.slice(0, 8).map(au => au.name);
          auChart.data.labels = [];
          auChart.data.datasets = topAUNames.map((auName, i) => ({
            label: auName,
            data: [],
            borderColor: palette[i % palette.length],
            backgroundColor: (palette[i % palette.length]) + '26',
            tension: 0.2,
            pointRadius: 0,
            fill: false,
          }));
          auChart._auNames = topAUNames;
        }

        // Add new data point
        const latest = timeline[timeline.length - 1];
        if (latest) {
          const timeLabel = latest.t.toFixed(1) + 's';
          auChart.data.labels.push(timeLabel);
          
          (auChart._auNames || []).forEach((auName, i) => {
            const v = latest.aus?.[auName];
            const val = typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
            const ds = auChart.data.datasets[i];
            ds.data.push(val);
            if (ds.data.length > MAX_POINTS) ds.data.shift();
          });
          
          if (auChart.data.labels.length > MAX_POINTS) {
            auChart.data.labels.shift();
          }
          auChart.update('none');
        }
      }

      // Throttled AU heatmap for ALL AUs (efficient rendering)
      try {
        if (!window._auHeatmapLast || (Date.now() - window._auHeatmapLast) > 900) {
          renderAUHeatmap(timeline);
          window._auHeatmapLast = Date.now();
        }
      } catch (_) {}

      // Show face quality hints when detection is weak
      if (typeof landmark_quality === 'number') {
        const out = document.getElementById('diagOutput');
        if (landmark_quality < 0.45) {
          const msg = `Face quality low (${(landmark_quality*100).toFixed(0)}%). Move closer, improve lighting, face camera.`;
          if (out && !out.textContent.includes(msg)) out.textContent = msg;
        }
      }

      pushChartPoint(typeof emotion_confidence === 'number' ? emotion_confidence : 0);

      // Record emotion change
      trackEmotionChange(normEmotion);

      // Auto-stop if server reports finished
      if (typeof progress === 'number' && progress >= 1) {
        await stopVideoRecording();
      }
    } catch (e) {
      console.warn('Telemetry poll failed', e);
    }
  }

  // --- AU Heatmap (All AUs) ---
  // Py-feat detects 20 AUs (AU27 not included, added AU11, AU24, AU28, AU43)
  const CANONICAL_AUS = [
    'AU01_Inner_Brow_Raiser','AU02_Outer_Brow_Raiser','AU04_Brow_Lowerer',
    'AU05_Upper_Lid_Raiser','AU06_Cheek_Raiser','AU07_Lid_Tightener',
    'AU09_Nose_Wrinkler','AU10_Upper_Lip_Raiser','AU11_Nasolabial_Deepener',
    'AU12_Lip_Corner_Puller','AU14_Dimpler','AU15_Lip_Corner_Depressor',
    'AU17_Chin_Raiser','AU20_Lip_Stretcher','AU23_Lip_Tightener',
    'AU24_Lip_Pressor','AU25_Lips_Part','AU26_Jaw_Drop','AU28_Lip_Suck',
    'AU43_Eyes_Closed'
  ];

  function colorForValue(v) {
    // v in [0,1] -> HSL hue 220 (blue) to 0 (red)
    const hue = Math.max(0, 220 - Math.round(220 * Math.min(1, Math.max(0, v))));
    const sat = 85; const light = 50;
    return `hsl(${hue} ${sat}% ${light}%)`;
  }

  function renderAUHeatmap(timeline) {
    const canvas = document.getElementById('auHeatmapCanvas');
    if (!canvas || !Array.isArray(timeline) || timeline.length === 0) return;
    const ctx = canvas.getContext('2d');
    const paddingLeft = 140; // space for AU labels
    const paddingRight = 8;
    const paddingTop = 8;
    const paddingBottom = 20;
    const w = canvas.clientWidth || canvas.width;
    const h = canvas.height;
    if (canvas.width !== w) canvas.width = w; // ensure crisp drawing on resize

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Determine time range (use provided timeline range)
    const tMin = timeline[0]?.t ?? 0;
    const tMax = timeline[timeline.length - 1]?.t ?? (tMin + 60);
    const cols = 60; // ~60s resolution
    const rows = CANONICAL_AUS.length;
    const cellW = Math.max(1, Math.floor((w - paddingLeft - paddingRight) / cols));
    const cellH = Math.max(2, Math.floor((h - paddingTop - paddingBottom) / rows));

    // Prepare accumulation matrices
    const sums = Array.from({ length: rows }, () => new Float32Array(cols));
    const counts = Array.from({ length: rows }, () => new Uint16Array(cols));

    const range = Math.max(0.001, (tMax - tMin));
    const binW = range / cols;

    // Accumulate values into bins
    for (let i = 0; i < timeline.length; i++) {
      const e = timeline[i];
      const t = typeof e?.t === 'number' ? e.t : null;
      if (t == null) continue;
      let col = Math.floor((t - tMin) / binW);
      if (col < 0) col = 0; if (col >= cols) col = cols - 1;
      const aus = e?.aus || {};
      for (let r = 0; r < rows; r++) {
        const name = CANONICAL_AUS[r];
        const v = typeof aus[name] === 'number' ? aus[name] : 0;
        sums[r][col] += v;
        counts[r][col]++;
      }
    }

    // Draw grid
    ctx.font = '12px system-ui, -apple-system, Segoe UI, Roboto';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Labels background
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, paddingLeft, canvas.height - paddingBottom);

    // Draw cells
    for (let r = 0; r < rows; r++) {
      const y = paddingTop + r * cellH;
      // AU label
      ctx.fillStyle = '#e5e7eb';
      const label = CANONICAL_AUS[r].replace(/^AU(\d+)_/, 'AU$1 ');
      ctx.fillText(label, 6, y + cellH / 2);
      for (let c = 0; c < cols; c++) {
        const x = paddingLeft + c * cellW;
        const avg = counts[r][c] ? (sums[r][c] / counts[r][c]) : 0;
        ctx.fillStyle = colorForValue(avg);
        ctx.fillRect(x, y, cellW + 1, cellH + 1);
      }
    }

    // Time axis ticks
    ctx.fillStyle = '#9ca3af';
    ctx.textBaseline = 'alphabetic';
    for (let mark = 0; mark <= 60; mark += 15) {
      const x = paddingLeft + Math.floor((mark / 60) * cols) * cellW;
      ctx.fillRect(x, canvas.height - paddingBottom + 2, 1, 6);
      const label = `${mark}s`;
      ctx.fillText(label, x - 6, canvas.height - 2);
    }
  }

  async function stopVideoRecording() {
    if (!videoState.isRecording) return;
    
    // Stop browser webcam if active
    if (videoState.useBrowserWebcam) {
      stopBrowserWebcam();
    } else {
      // Server-side recording
      try {
        stopVideoBtn.disabled = true;
        await fetch('/stop_recording', { method: 'POST' });
      } catch (e) {
        console.warn('Stop request failed', e);
      }
    }
    stopVideoUI();
  }

  function stopVideoUI() {
    videoState.isRecording = false;
    videoState.useBrowserWebcam = false;
    if (videoState.pollId) clearInterval(videoState.pollId);
    if (videoState.timerId) clearInterval(videoState.timerId);
    if (browserAnalysisInterval) {
      clearInterval(browserAnalysisInterval);
      browserAnalysisInterval = null;
    }
    videoState.pollId = null;
    videoState.timerId = null;
    startVideoBtn.disabled = false;
    stopVideoBtn.disabled = true;
    videoRecBadge.className = 'badge badge-muted';
    videoRecBadge.textContent = 'Video: Stopped';
    serverStatus.className = 'badge badge-idle';
    serverStatus.textContent = 'Idle';
    videoProgressBar.style.width = '0%';
    videoTimerEl.textContent = '00:00';
    // Reset emotion change tracker for next session
    lastEmotion = null;
    if (emotionChangesEl) emotionChangesEl.innerHTML = '';
    
    // Keep the last AU data visible after stopping (don't clear AU bars)
    // The AU breakdown section will show the final state from the recording

    // Fetch the latest aggregated session result and render
    fetchLatestResult();
    
    // Fetch and display session summary
    fetchAndDisplaySessionSummary();
  }

  async function fetchAndDisplaySessionSummary() {
    try {
      const res = await fetch('/get_session_summary');
      if (!res.ok) {
        console.warn('Failed to fetch session summary:', res.status);
        return;
      }
      const summary = await res.json();
      
      // Render the emotion distribution pie chart
      const emotionPieCtx = document.getElementById('emotionPie');
      if (emotionPieCtx && summary.emotion_distribution) {
        const labels = Object.keys(summary.emotion_distribution);
        const values = labels.map(k => summary.emotion_distribution[k]);
        const palette = ['#22c55e','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#10b981','#e11d48','#0ea5e9'];
        const colors = labels.map((_, i) => palette[i % palette.length]);
        
        try { emotionPie?.destroy?.(); } catch {}
        emotionPie = new Chart(emotionPieCtx, {
          type: 'doughnut',
          data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{ data: values, backgroundColor: colors }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } }
          },
        });
      }
      
      // Display summary stats
      const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
      set('sumChanges', typeof summary.mood_changes === 'number' ? String(summary.mood_changes) : '—');
      set('sumAvgConf', typeof summary.emotion_avg_confidence === 'number' ? (summary.emotion_avg_confidence*100).toFixed(0)+'%' : '—');
      const lowQPct = summary.quality_metrics?.low_quality_frames || 0;
      set('sumLowQ', typeof lowQPct === 'number' ? lowQPct.toFixed(0)+'%' : '—');
      set('sumDuration', typeof summary.duration_seconds === 'number' ? summary.duration_seconds.toFixed(1)+'s' : '—');
      
      // Display top AUs
      if (summary.top_aus_overall && Array.isArray(summary.top_aus_overall)) {
        const topAUsEl = document.getElementById('topAUsListSummary');
        if (topAUsEl) {
          topAUsEl.innerHTML = summary.top_aus_overall.slice(0, 8).map(au => 
            `<div style="display:flex;justify-content:space-between;font-size:0.9rem;">
              <span>${au.name}</span>
              <span style="font-weight:bold;">${(au.mean*100).toFixed(0)}%</span>
            </div>`
          ).join('');
        }
      }

      // Display AU statistics table if needed
      if (summary.au_statistics) {
        const statsTable = document.querySelector('#auStatsTable tbody');
        if (statsTable) {
          statsTable.innerHTML = '';
          Object.entries(summary.au_statistics).slice(0, 10).forEach(([auName, stats]) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>${auName}</td>
              <td>${(stats.mean*100).toFixed(0)}%</td>
              <td>${(stats.max*100).toFixed(0)}%</td>
              <td>${(stats.min*100).toFixed(0)}%</td>
            `;
            statsTable.appendChild(tr);
          });
        }
      }

    } catch (e) {
      console.warn('Failed to display session summary:', e);
    }
  }

  async function fetchLatestResult() {
    try {
      const res = await fetch('/get_results');
      if (!res.ok) return;
      const data = await res.json();
      const results = Array.isArray(data?.results) ? data.results : [];
      if (!results.length) return;
      const last = results[results.length - 1];
      // Update quick stats
      if (lastResDominantEl) lastResDominantEl.textContent = last.dominant_emotion || '—';
      if (lastResFacesEl) lastResFacesEl.textContent = (last.faces_detected ?? '—');
      const auCount = last.action_units ? Object.keys(last.action_units).length : 0;
      if (lastResAUsEl) lastResAUsEl.textContent = auCount;
      // Raw JSON
      if (lastResultJsonEl) {
        try { lastResultJsonEl.textContent = JSON.stringify(last, null, 2); } catch {}
      }

      // Render static timeline chart if present
      if (Array.isArray(last.timeline) && last.timeline.length > 0) {
        renderTimelineFromResult(last.timeline);
        renderAUTimelineFromResult(last.timeline);
      }

      // Render session summary/segments if present
      if (last.emotion_summary || last.emotion_segments) {
        renderEmotionSummary(last.emotion_summary || {}, last.emotion_segments || []);
      }
    } catch (e) {
      // ignore transient errors
    }
  }

  // Audio state and waveform
  const audioState = {
    isRecording: false,
    mediaRecorder: null,
    audioChunks: [],
    stream: null,
    analyser: null,
    rafId: null,
    audioCtx: null,
    source: null,
  };

  async function populateMicDevices() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        console.warn('MediaDevices API not available');
        return;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter((d) => d.kind === 'audioinput');
      if (!micSelect) return;
      micSelect.innerHTML = '';
      mics.forEach((m, idx) => {
        const opt = document.createElement('option');
        opt.value = m.deviceId;
        opt.textContent = m.label || `Microphone ${idx + 1}`;
        micSelect.appendChild(opt);
      });
    } catch (e) {
      console.warn('Could not enumerate microphones', e);
    }
  }

  function drawWaveform() {
    if (!audioState.analyser || !audioWaveCanvas) return;
    const canvas = audioWaveCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.clientWidth;
    const height = canvas.height;
    const bufferLength = audioState.analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!audioState.isRecording) return;
      audioState.analyser.getByteTimeDomainData(dataArray);
      ctx.clearRect(0, 0, width, height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#10b981';
      ctx.beginPath();
      const sliceWidth = width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // 0..2
        const y = (v * height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      audioState.rafId = requestAnimationFrame(draw);
    }
    draw();
  }

  async function startAudioRecording() {
    if (audioState.isRecording) return;
    audioState.audioChunks = [];
    
    const duration = parseInt(audioDurationEl?.value || '10', 10);
    const deviceId = micSelect?.value || undefined;

    try {
      if (startAudioBtn) startAudioBtn.disabled = true;
      if (stopAudioBtn) stopAudioBtn.disabled = false;
      if (audioStatusEl) audioStatusEl.textContent = 'Recording…';

      const constraints = { audio: deviceId ? { deviceId: { exact: deviceId } } : true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      audioState.stream = stream;

      // MediaRecorder
      const mr = new MediaRecorder(stream);
      audioState.mediaRecorder = mr;
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioState.audioChunks.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(audioState.audioChunks, { type: 'audio/webm' });
        uploadAudioForAnalysis(blob);
      };

      // Waveform
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      audioState.audioCtx = ctx;
      const source = ctx.createMediaStreamSource(stream);
      audioState.source = source;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      audioState.analyser = analyser;
      audioState.isRecording = true;
      drawWaveform();

      mr.start(100); // collect in small chunks

      // Timer
      const startTs = Date.now();
      const tid = setInterval(() => {
        const elapsedSec = (Date.now() - startTs) / 1000;
        if (audioTimerEl) audioTimerEl.textContent = fmtTime(elapsedSec);
        if (elapsedSec >= duration) {
          clearInterval(tid);
          stopAudioRecording();
        }
      }, 300);
    } catch (e) {
      console.error(e);
      if (audioStatusEl) audioStatusEl.textContent = 'Mic error';
      if (startAudioBtn) startAudioBtn.disabled = false;
      if (stopAudioBtn) stopAudioBtn.disabled = true;
      alert('Failed to start mic: ' + e.message);
    }
  }

  function stopAudioRecording() {
    if (!audioState.isRecording) return;
    try {
      audioState.isRecording = false;
      if (audioState.mediaRecorder && audioState.mediaRecorder.state !== 'inactive') {
        audioState.mediaRecorder.stop();
      }
      if (audioState.stream) {
        audioState.stream.getTracks().forEach((t) => t.stop());
      }
      if (audioState.audioCtx) {
        audioState.audioCtx.close();
      }
      if (audioState.rafId) cancelAnimationFrame(audioState.rafId);
      if (audioStatusEl) audioStatusEl.textContent = 'Processing…';
    } finally {
      if (startAudioBtn) startAudioBtn.disabled = false;
      if (stopAudioBtn) stopAudioBtn.disabled = true;
      if (audioTimerEl) audioTimerEl.textContent = '00:00';
    }
  }

  async function uploadAudioForAnalysis(blob) {
    const progressDiv = document.getElementById('audioProgress');
    
    try {
      if (progressDiv) progressDiv.style.display = 'block';
      
      const fd = new FormData();
      const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
      fd.append('audio_file', file);
      const res = await fetch('/analyze_audio', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Audio analysis failed');
      
      if (audioStatusEl) audioStatusEl.textContent = 'Done ✓';
      if (progressDiv) progressDiv.style.display = 'none';
      
      displayAudioResults(data.result || {});
    } catch (e) {
      console.error(e);
      if (audioStatusEl) audioStatusEl.textContent = 'Error';
      if (progressDiv) progressDiv.style.display = 'none';
      alert('Audio analysis error: ' + e.message);
    }
  }

  // Upload selected audio file (from file input)
  let uploadInFlight = false;
  async function handleUploadSelectedAudio() {
    if (uploadInFlight) return;
    const input = document.getElementById('audioFileUpload');
    const btn = document.getElementById('uploadAudioBtn');
    const progressDiv = document.getElementById('audioProgress');
    const resultDisplay = document.getElementById('audioResultDisplay');
    const placeholder = document.getElementById('audioPlaceholder');
    const errorDiv = document.getElementById('audioError');
    
    if (!input || !input.files || !input.files[0]) {
      alert('Please choose an audio file first.');
      return;
    }
    
    try {
      uploadInFlight = true;
      
      // Update UI - show progress
      if (btn) { 
        btn.disabled = true; 
        btn.innerHTML = '<span style="margin-right: 8px;">⏳</span>Analyzing...';
      }
      if (progressDiv) progressDiv.style.display = 'block';
      if (resultDisplay) resultDisplay.style.display = 'none';
      if (placeholder) placeholder.style.display = 'none';
      if (errorDiv) errorDiv.style.display = 'none';
      
      // Make request
      const fd = new FormData();
      fd.append('audio_file', input.files[0]);
      const res = await fetch('/analyze_audio', { method: 'POST', body: fd });
      const data = await res.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Audio analysis failed');
      }
      
      // Hide progress
      if (progressDiv) progressDiv.style.display = 'none';
      
      // Display results
      displayAudioResults(data.result || {});
      
    } catch (e) {
      console.error('Audio analysis error:', e);
      
      // Hide progress
      if (progressDiv) progressDiv.style.display = 'none';
      
      // Show error
      if (errorDiv) {
        errorDiv.style.display = 'block';
        const errorMsg = document.getElementById('audioErrorMessage');
        if (errorMsg) {
          errorMsg.textContent = e.message || 'An unexpected error occurred during analysis.';
        }
      }
      
      // Hide other sections
      if (resultDisplay) resultDisplay.style.display = 'none';
      if (placeholder) placeholder.style.display = 'none';
      
    } finally {
      uploadInFlight = false;
      if (btn) { 
        btn.disabled = false; 
        btn.innerHTML = '<span style="margin-right: 8px;">🎯</span>Analyze Audio';
      }
    }
  }

  function displayAudioResults(result) {
    const resultDisplay = document.getElementById('audioResultDisplay');
    const placeholder = document.getElementById('audioPlaceholder');
    const errorDiv = document.getElementById('audioError');
    
    if (!resultDisplay) return;
    
    // Hide placeholder and error
    if (placeholder) placeholder.style.display = 'none';
    if (errorDiv) errorDiv.style.display = 'none';
    
    // Check for error in result
    if (result.error) {
      console.error('Audio analysis error:', result.error);
      if (errorDiv) {
        errorDiv.textContent = 'Error: ' + result.error;
        errorDiv.style.display = 'block';
      }
      resultDisplay.style.display = 'none';
      return;
    }
    
    // Show results
    resultDisplay.style.display = 'block';
    
    // Populate emotion cards
    const primaryEmotion = document.getElementById('audioPrimaryEmotion');
    const confidence = document.getElementById('audioConfidenceScore');
    const mood = document.getElementById('audioMood');
    const energy = document.getElementById('audioEnergy');
    const transcript = document.getElementById('audioTranscript');
    const overallVibe = document.getElementById('audioOverallVibe');
    const tone = document.getElementById('audioTone');
    const secondaryEmotions = document.getElementById('audioSecondaryEmotions');
    const keyPhrases = document.getElementById('audioKeyPhrases');
    const rawJSON = document.getElementById('audioRawJSON');
    
    if (primaryEmotion) {
      primaryEmotion.textContent = result.primary_emotion || '—';
      // Add emoji based on emotion
      const emotionEmoji = {
        'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Calm': '😌',
        'Fear': '😰', 'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐'
      };
      const emoji = emotionEmoji[result.primary_emotion] || '';
      if (emoji) primaryEmotion.textContent = emoji + ' ' + result.primary_emotion;
    }
    
    if (confidence) {
      const conf = result.confidence || 0;
      confidence.textContent = Math.round(conf * 100) + '%';
    }
    
    if (mood) {
      mood.textContent = result.mood_category || '—';
    }
    
    if (energy) {
      const energyLevel = result.energy_level || '—';
      energy.textContent = energyLevel;
      // Add emoji for energy
      const energyEmoji = { 'High': '⚡', 'Medium': '🔋', 'Low': '🔋' };
      const eEmoji = energyEmoji[energyLevel] || '';
      if (eEmoji) energy.textContent = eEmoji + ' ' + energyLevel;
    }
    
    if (transcript) {
      transcript.textContent = result.transcript || 'No transcript available';
    }
    
    if (overallVibe) {
      overallVibe.textContent = result.overall_vibe || result.explanation || '—';
    }
    
    if (tone) {
      tone.textContent = result.tone || '—';
    }
    
    if (secondaryEmotions) {
      const secondary = result.secondary_emotions || [];
      secondaryEmotions.textContent = Array.isArray(secondary) && secondary.length > 0 
        ? secondary.join(', ') 
        : 'None detected';
    }
    
    if (keyPhrases) {
      const phrases = result.key_phrases || [];
      keyPhrases.textContent = Array.isArray(phrases) && phrases.length > 0 
        ? phrases.map(p => `"${p}"`).join(', ') 
        : 'None identified';
    }
    
    if (rawJSON) {
      rawJSON.textContent = JSON.stringify(result, null, 2);
    }
  }

  // Update file name display when file is selected
  document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audioFileUpload');
    const fileNameDisplay = document.getElementById('audioFileName');
    
    if (fileInput && fileNameDisplay) {
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          const file = e.target.files[0];
          const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
          fileNameDisplay.textContent = `Selected: ${file.name} (${sizeMB} MB)`;
          fileNameDisplay.style.color = '#3b82f6';
        } else {
          fileNameDisplay.textContent = '';
        }
      });
    }
  });

  // Tabs behavior
  function setupTabs() {
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        tabs.forEach((t) => t.classList.remove('active'));
        panels.forEach((p) => p.classList.remove('active'));
        tab.classList.add('active');
        const id = tab.getAttribute('data-tab');
        const panel = document.getElementById(id);
        if (panel) panel.classList.add('active');
        // Only probe cameras when video tab is activated to avoid noise
        if (id === 'videoTab') {
          checkCameras();
        }
      });
    });
  }

  // Audio input mode switching (Live Recording vs File Upload)
  function setupAudioModeTabs() {
    const recordModeBtn = document.getElementById('audioModeRecord');
    const uploadModeBtn = document.getElementById('audioModeUpload');
    const liveRecordingSection = document.getElementById('liveRecordingSection');
    const fileUploadSection = document.getElementById('fileUploadSection');

    if (!recordModeBtn || !uploadModeBtn) return;

    function switchToRecordMode() {
      recordModeBtn.classList.add('primary');
      recordModeBtn.classList.remove('btn-outline');
      uploadModeBtn.classList.remove('primary');
      uploadModeBtn.classList.add('btn-outline');
      if (liveRecordingSection) liveRecordingSection.style.display = 'block';
      if (fileUploadSection) fileUploadSection.style.display = 'none';
      // Request mic permissions early to populate device list
      populateMicDevices();
    }

    function switchToUploadMode() {
      uploadModeBtn.classList.add('primary');
      uploadModeBtn.classList.remove('btn-outline');
      recordModeBtn.classList.remove('primary');
      recordModeBtn.classList.add('btn-outline');
      if (liveRecordingSection) liveRecordingSection.style.display = 'none';
      if (fileUploadSection) fileUploadSection.style.display = 'block';
    }

    recordModeBtn.addEventListener('click', switchToRecordMode);
    uploadModeBtn.addEventListener('click', switchToUploadMode);

    // Default to recording mode
    switchToRecordMode();
  }

  function setupEvents() {
    startVideoBtn?.addEventListener('click', startVideoRecording);
    stopVideoBtn?.addEventListener('click', stopVideoRecording);
    cameraSelect?.addEventListener('change', () => {
      const device = cameraSelect ? parseInt(cameraSelect.value || '0', 10) : 0;
      const img = document.getElementById('serverStream');
      if (img) img.src = `/video_feed?device=${encodeURIComponent(device)}`;
    });
    startAudioBtn?.addEventListener('click', startAudioRecording);
    stopAudioBtn?.addEventListener('click', stopAudioRecording);
    navigator.mediaDevices?.addEventListener?.('devicechange', populateMicDevices);
  const uploadBtn = document.getElementById('uploadAudioBtn');
  if (uploadBtn && !uploadBtn._hooked) {
    uploadBtn.addEventListener('click', handleUploadSelectedAudio);
    uploadBtn._hooked = true;
  }
    const camBtn = document.getElementById('checkCamerasBtn');
    camBtn?.addEventListener('click', checkCameras);
    const macBtn = document.getElementById('showMacSetupBtn');
    macBtn?.addEventListener('click', showMacSetup);
    
    // Setup audio mode tabs
    setupAudioModeTabs();
  }

  function init() {
    setupTabs();
    setupEvents();
    initChart();
    initAUAnalyticsDashboard();
    populateMicDevices();
    // Defer camera probe until video tab is activated or explicitly requested
    // to avoid unnecessary noise when user is on the audio tab
    updateBackendStatus(); // Only update backend status, not camera probing
  }

  document.addEventListener('DOMContentLoaded', init);
})();
