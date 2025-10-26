// Multimodal Emotion Analyzer Frontend
// Handles UI state for video (server-stream) and audio (mic recording + upload)

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
  const auBadge = $('#auBadge');
  // Last-session results elements
  const lastResDominantEl = $('#lastResDominant');
  const lastResFacesEl = $('#lastResFaces');
  const lastResAUsEl = $('#lastResAUs');
  const lastResultJsonEl = $('#lastResultJson');

  // Build/version marker for cache checks
  const BUILD_VERSION = '2025-10-25-03';
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
  const audioWaveCanvas = $('#audioWave');

  // Chart.js setup for emotion timeline
  let emotionChart;
  let auChart;
  let emotionPie;
  let emotionBars;      // current distribution (bar)
  let emotionMultiChart; // live multi-series timeline
  function initChart() {
    const ctx = document.getElementById('emotionChart');
    if (!ctx) return;
    emotionChart = new Chart(ctx, {
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
      auChart = new Chart(auCtx, {
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
    }

    // Live current distribution (bar)
    const barsCtx = document.getElementById('emotionBars');
    if (barsCtx) {
      const labels = ['Happy','Sad','Surprise','Angry','Fear','Disgust','Neutral'];
      emotionBars = new Chart(barsCtx, {
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
    }

    // Live multi-series timeline of emotion scores
    const multiCtx = document.getElementById('emotionMultiChart');
    if (multiCtx) {
      const KEYS = ['happiness','sadness','surprise','anger','fear','disgust'];
      const LABELS = { happiness:'Happy', sadness:'Sad', surprise:'Surprise', anger:'Angry', fear:'Fear', disgust:'Disgust' };
      const COLORS = { happiness:'#22c55e', sadness:'#3b82f6', surprise:'#f59e0b', anger:'#ef4444', fear:'#8b5cf6', disgust:'#10b981' };
      emotionMultiChart = new Chart(multiCtx, {
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
      emotionMultiChart._emoKeys = KEYS;
    }

    const pieCtx = document.getElementById('emotionPie');
    if (pieCtx) {
      emotionPie = new Chart(pieCtx, {
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
          const res = await fetch('/export_last_session_csv');
          if (!res.ok) throw new Error('Export failed');
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'session_timeline.csv';
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        } catch (e) {
          alert('Export failed: ' + e.message);
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
    const entries = Object.entries(aus)
      .filter(([, v]) => typeof v === 'number' && !isNaN(v))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    if (entries.length === 0) {
      if (!auPlaceholderEl) {
        auPlaceholderEl = document.createElement('div');
        auPlaceholderEl.className = 'reason';
        auPlaceholderEl.textContent = 'No facial Action Units detected yet. Ensure your face is well-lit and within frame.';
      }
      auBarsEl.innerHTML = '';
      auBarsEl.appendChild(auPlaceholderEl);
      return;
    }
    auBarsEl.innerHTML = entries
      .map(([name, val]) => {
        const pct = Math.max(0, Math.min(100, Math.round(val * 100)));
        return `<div class="au-bar">
          <div class="au-label">${name}</div>
          <div class="au-track"><div class="au-fill" style="width:${pct}%"></div></div>
          <div class="au-val">${pct}%</div>
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
    const fmt = (k) => `${k} ${(get(k) * 100).toFixed(0)}%`;

    if (rules.include?.length) {
      const hits = rules.include.filter((k) => get(k) > 0.25);
      if (hits.length) {
        reasons.push(`Supporting AUs: ${hits.map(fmt).join(', ')}`);
      }
    }
    if (rules.inhibit?.length) {
      const hits = rules.inhibit.filter((k) => get(k) > 0.25);
      if (hits.length) {
        reasons.push(`Inhibitors present: ${hits.map(fmt).join(', ')}`);
      }
    }
    if (!reasons.length) reasons.push('No strong AU signals; likely neutral or low intensity.');
    emotionReasonsEl.innerHTML = reasons.map((r) => `<div class="reason">${r}</div>`).join('');
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
  };

  async function startVideoRecording() {
    if (videoState.isRecording) return;
    const device = cameraSelect ? parseInt(cameraSelect.value || '0', 10) : 0;
    const intervalSeconds = parseInt(intervalSecondsEl.value || '10', 10);
    // We pass one interval as duration per session
    const reqBody = { duration: intervalSeconds };

    startVideoBtn.disabled = true;
    stopVideoBtn.disabled = false;
    videoRecBadge.className = 'badge badge-live';
    videoRecBadge.textContent = 'Video: Recording';
    serverStatus.className = 'badge badge-live';
    serverStatus.textContent = 'Live';

    try {
      // Ensure stream uses selected camera
      const img = document.getElementById('serverStream');
      if (img) img.src = `/video_feed?device=${encodeURIComponent(device)}`;
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

      // Live multi-series timeline chart
      if (emotionMultiChart && emotionScores) {
        const MAX_POINTS = 120; // ~60s if polling every 500ms
        const elapsedSec = (Date.now() - (videoState.startTs || Date.now())) / 1000;
        emotionMultiChart.data.labels.push(elapsedSec.toFixed(1) + 's');
        // push per series values in chart's key order
        (emotionMultiChart._emoKeys || []).forEach((k, i) => {
          const v = emotionScores?.[k];
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

  async function stopVideoRecording() {
    if (!videoState.isRecording) return;
    try {
      stopVideoBtn.disabled = true;
      await fetch('/stop_recording', { method: 'POST' });
    } catch (e) {
      console.warn('Stop request failed', e);
    }
    stopVideoUI();
  }

  function stopVideoUI() {
    videoState.isRecording = false;
    if (videoState.pollId) clearInterval(videoState.pollId);
    if (videoState.timerId) clearInterval(videoState.timerId);
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

    // Fetch the latest aggregated session result and render
    fetchLatestResult();
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
      const devices = await navigator.mediaDevices.enumerateDevices();
      const mics = devices.filter((d) => d.kind === 'audioinput');
      micSelect.innerHTML = '';
      mics.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.deviceId;
        opt.textContent = m.label || `Microphone ${micSelect.length + 1}`;
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
    const duration = parseInt(audioDurationEl.value || '10', 10);
    const deviceId = micSelect.value || undefined;

    try {
      startAudioBtn.disabled = true;
      stopAudioBtn.disabled = false;
      audioStatusEl.textContent = 'Recording…';

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
        audioTimerEl.textContent = fmtTime(elapsedSec);
        if (elapsedSec >= duration) {
          clearInterval(tid);
          stopAudioRecording();
        }
      }, 300);
    } catch (e) {
      console.error(e);
      audioStatusEl.textContent = 'Mic error';
      startAudioBtn.disabled = false;
      stopAudioBtn.disabled = true;
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
      audioStatusEl.textContent = 'Processing…';
    } finally {
      startAudioBtn.disabled = false;
      stopAudioBtn.disabled = true;
      audioTimerEl.textContent = '00:00';
    }
  }

  async function uploadAudioForAnalysis(blob) {
    try {
      const fd = new FormData();
      const file = new File([blob], 'recording.webm', { type: 'audio/webm' });
      fd.append('audio_file', file);
      const res = await fetch('/analyze_audio', { method: 'POST', body: fd });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Audio analysis failed');
      audioStatusEl.textContent = 'Done';
      // Update telemetry
      const result = data.result || {};
      audioEmotionEl.textContent = result.sentiment || result.emotion || '—';
      const conf = result.confidence;
      audioConfidenceEl.textContent = typeof conf === 'number' ? `${(conf * 100).toFixed(0)}%` : '—';
      try { audioResultsContent.textContent = JSON.stringify(result, null, 2); } catch {}
    } catch (e) {
      console.error(e);
      audioStatusEl.textContent = 'Error';
      alert('Audio analysis error: ' + e.message);
    }
  }

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
      });
    });
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
    const camBtn = document.getElementById('checkCamerasBtn');
    camBtn?.addEventListener('click', checkCameras);
    const macBtn = document.getElementById('showMacSetupBtn');
    macBtn?.addEventListener('click', showMacSetup);
  }

  function init() {
    setupTabs();
    setupEvents();
    initChart();
    populateMicDevices();
    // Initial camera probe to populate choices
    checkCameras();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
