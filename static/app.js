/*
  Deprecated stub: /static/app.js
  The active frontend script is /static/js/app.js (loaded by templates/index.html).
  This file remains only to avoid 404s from old bookmarks or documentation.
  It performs no operations.
*/

/* eslint-disable no-console */
console.warn('[deprecation] /static/app.js is no longer used. Please reference /static/js/app.js.');
/* eslint-enable no-console */

// Intentionally left blank.

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupVideoTab();
    setupAudioTab();
});

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');

            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// ============================================================================
// VIDEO TAB
// ============================================================================

function setupVideoTab() {
    const startBtn = document.getElementById('videoStartBtn');
    const stopBtn = document.getElementById('videoStopBtn');
    const durationInput = document.getElementById('videoDuration');
    const pauseInput = document.getElementById('videoPause');
    const sessionsInput = document.getElementById('videoSessions');

    startBtn.addEventListener('click', startVideoRecording);
    stopBtn.addEventListener('click', stopVideoRecording);

    durationInput.addEventListener('change', (e) => {
        state.video.duration = parseInt(e.target.value);
    });

    pauseInput.addEventListener('change', (e) => {
        state.video.pauseTime = parseInt(e.target.value);
    });

    sessionsInput.addEventListener('change', (e) => {
        state.video.sessions = parseInt(e.target.value);
    });

    // Initialize emotion chart
    initializeEmotionChart();
}

async function startVideoRecording() {
    try {
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user' },
            audio: false
        });

        state.video.stream = stream;
        state.video.recording = true;
        state.video.currentSession = 0;
        state.video.emotions = [];

        // Display video preview
        const videoPreview = document.getElementById('videoPreview');
        videoPreview.srcObject = stream;

        // Setup media recorder
        const mimeType = 'video/webm;codecs=vp8,opus';
        state.video.mediaRecorder = new MediaRecorder(stream, { mimeType });
        state.video.mediaRecorder.start();

        // Update UI
        updateVideoUI('recording');
        document.getElementById('videoStartBtn').disabled = true;
        document.getElementById('videoStopBtn').disabled = false;
        document.getElementById('videoDuration').disabled = true;
        document.getElementById('videoPause').disabled = true;
        document.getElementById('videoSessions').disabled = true;

        // Start recording session
        startVideoSession();

    } catch (error) {
        console.error('Error accessing camera:', error);
        showError('Could not access camera. Please check permissions.');
    }
}

function startVideoSession() {
    state.video.currentSession++;
    state.video.startTime = Date.now();

    // Send start_recording request to Flask
    fetch('/start_recording', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            duration: state.video.duration,
            session: state.video.currentSession
        })
    });

    // Start polling for emotion updates
    state.video.updateInterval = setInterval(updateVideoEmotion, 1000);

    // Set timer for session end
    setTimeout(() => {
        if (state.video.recording) {
            endVideoSession();
        }
    }, state.video.duration * 1000);
}

function endVideoSession() {
    clearInterval(state.video.updateInterval);

    if (state.video.currentSession < state.video.sessions) {
        // Pause before next session
        updateVideoUI('pausing');
        
        setTimeout(() => {
            if (state.video.recording) {
                startVideoSession();
            }
        }, state.video.pauseTime * 1000);
    } else {
        // All sessions complete
        stopVideoRecording();
    }
}

function stopVideoRecording() {
    state.video.recording = false;

    // Stop media recorder
    if (state.video.mediaRecorder && state.video.mediaRecorder.state !== 'inactive') {
        state.video.mediaRecorder.stop();
    }

    // Stop stream
    if (state.video.stream) {
        state.video.stream.getTracks().forEach(track => track.stop());
    }

    // Stop updates
    clearInterval(state.video.updateInterval);

    // Send stop request to Flask
    fetch('/stop_recording', { method: 'POST' });

    // Update UI
    updateVideoUI('completed');
    document.getElementById('videoStartBtn').disabled = false;
    document.getElementById('videoStopBtn').disabled = true;
    document.getElementById('videoDuration').disabled = false;
    document.getElementById('videoPause').disabled = false;
    document.getElementById('videoSessions').disabled = false;

    // Show results
    showVideoResults();
}

async function updateVideoEmotion() {
    try {
        const response = await fetch('/get_realtime_data');
        const data = await response.json();

        if (data.emotion && data.emotion !== 'Neutral' && data.emotion !== 'Processing...') {
            const emotionDisplay = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
            document.getElementById('videoEmotion').textContent = emotionDisplay;
            document.getElementById('videoEmotion').className = `emotion-badge ${data.emotion.toLowerCase()}`;
            document.getElementById('videoConfidence').textContent = `${(data.emotion_confidence * 100).toFixed(1)}%`;

            // Add to emotion history
            state.video.emotions.push({
                time: new Date(),
                emotion: data.emotion,
                confidence: data.emotion_confidence
            });

            // Update chart
            updateEmotionChart(data.emotion);
        }

        // Update timer
        const elapsed = data.elapsed || 0;
        const minutes = Math.floor(elapsed / 60);
        const seconds = Math.floor(elapsed % 60);
        document.getElementById('videoTimer').textContent = 
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

    } catch (error) {
        console.error('Error updating emotion:', error);
    }
}

function updateVideoUI(status) {
    const statusElement = document.getElementById('videoStatus');
    const statusClass = status === 'recording' ? 'recording' : 
                        status === 'completed' ? 'completed' : '';

    statusElement.textContent = 
        status === 'recording' ? 'Recording...' :
        status === 'pausing' ? 'Pausing...' :
        status === 'completed' ? 'Completed' : 'Ready';

    statusElement.className = `status ${statusClass}`;
}

function initializeEmotionChart() {
    const ctx = document.getElementById('emotionChart')?.getContext('2d');
    if (!ctx) return;

    state.video.emotionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Emotion Timeline',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

function updateEmotionChart(emotion) {
    if (!state.video.emotionChart) return;

    const confidenceMap = {
        'happiness': 0.8,
        'sadness': 0.3,
        'anger': 0.5,
        'fear': 0.4,
        'surprise': 0.7,
        'disgust': 0.6,
        'neutral': 0.2
    };

    const value = confidenceMap[emotion.toLowerCase()] || 0.5;
    const time = new Date().toLocaleTimeString();

    state.video.emotionChart.data.labels.push(time);
    state.video.emotionChart.data.datasets[0].data.push(value);

    if (state.video.emotionChart.data.labels.length > 20) {
        state.video.emotionChart.data.labels.shift();
        state.video.emotionChart.data.datasets[0].data.shift();
    }

    state.video.emotionChart.update();
}

async function showVideoResults() {
    try {
        const response = await fetch('/get_results');
        const results = await response.json();

        if (results.success) {
            const resultsDiv = document.getElementById('videoResults');
            const contentDiv = document.getElementById('videoResultsContent');

            let html = '<div class="result-item"><strong>Session Summary:</strong></div>';
            
            for (const [key, value] of Object.entries(results.data || {})) {
                html += `
                    <div class="result-item">
                        <div class="result-label">${formatKey(key)}</div>
                        <div class="result-value">${formatValue(value)}</div>
                    </div>
                `;
            }

            contentDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// ============================================================================
// AUDIO TAB
// ============================================================================

function setupAudioTab() {
    const startBtn = document.getElementById('audioStartBtn');
    const stopBtn = document.getElementById('audioStopBtn');
    const durationInput = document.getElementById('audioDuration');

    startBtn.addEventListener('click', startAudioRecording);
    stopBtn.addEventListener('click', stopAudioRecording);

    durationInput.addEventListener('change', (e) => {
        state.audio.duration = parseInt(e.target.value);
    });
}

async function startAudioRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        state.audio.stream = stream;
        state.audio.recording = true;
        state.audio.startTime = Date.now();

        // Setup audio context for visualization
        state.audio.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = state.audio.audioContext.createMediaStreamSource(stream);
        state.audio.analyser = state.audio.audioContext.createAnalyser();
        source.connect(state.audio.analyser);

        // Setup media recorder
        state.audio.mediaRecorder = new MediaRecorder(stream);
        state.audio.mediaRecorder.start();

        // Update UI
        updateAudioUI('recording');
        document.getElementById('audioStartBtn').disabled = true;
        document.getElementById('audioStopBtn').disabled = false;
        document.getElementById('audioDuration').disabled = true;

        // Start visualization
        visualizeAudio();

        // Start polling for emotion updates
        state.audio.updateInterval = setInterval(updateAudioEmotion, 1000);

        // Send start_recording request to Flask
        fetch('/start_recording', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                duration: state.audio.duration,
                mode: 'audio'
            })
        });

        // Set timer for recording end
        setTimeout(() => {
            if (state.audio.recording) {
                stopAudioRecording();
            }
        }, state.audio.duration * 1000);

    } catch (error) {
        console.error('Error accessing microphone:', error);
        showError('Could not access microphone. Please check permissions.');
    }
}

function stopAudioRecording() {
    state.audio.recording = false;

    // Stop media recorder
    if (state.audio.mediaRecorder && state.audio.mediaRecorder.state !== 'inactive') {
        state.audio.mediaRecorder.stop();
    }

    // Stop stream
    if (state.audio.stream) {
        state.audio.stream.getTracks().forEach(track => track.stop());
    }

    // Stop audio context
    if (state.audio.audioContext) {
        state.audio.audioContext.close();
    }

    // Stop updates
    clearInterval(state.audio.updateInterval);
    cancelAnimationFrame(state.audio.animationId);

    // Send stop request to Flask
    fetch('/stop_recording', { method: 'POST' });

    // Update UI
    updateAudioUI('completed');
    document.getElementById('audioStartBtn').disabled = false;
    document.getElementById('audioStopBtn').disabled = true;
    document.getElementById('audioDuration').disabled = false;

    // Show results
    showAudioResults();
}

function visualizeAudio() {
    if (!state.audio.recording || !state.audio.analyser) return;

    const dataArray = new Uint8Array(state.audio.analyser.frequencyBinCount);
    state.audio.analyser.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
    const percentage = (average / 255) * 100;

    const audioLevel = document.getElementById('audioLevel');
    audioLevel.style.width = percentage + '%';

    state.audio.animationId = requestAnimationFrame(visualizeAudio);
}

async function updateAudioEmotion() {
    try {
        const response = await fetch('/get_realtime_data');
        const data = await response.json();

        if (data.emotion && data.emotion !== 'Neutral' && data.emotion !== 'Processing...') {
            const emotionDisplay = data.emotion.charAt(0).toUpperCase() + data.emotion.slice(1);
            document.getElementById('audioEmotion').textContent = emotionDisplay;
            document.getElementById('audioEmotion').className = `emotion-badge ${data.emotion.toLowerCase()}`;
            document.getElementById('audioConfidence').textContent = `${(data.emotion_confidence * 100).toFixed(1)}%`;

            state.audio.emotions.push({
                time: new Date(),
                emotion: data.emotion,
                confidence: data.emotion_confidence
            });
        }

        // Update timer
        const elapsed = (Date.now() - state.audio.startTime) / 1000;
        const minutes = Math.floor(elapsed / 60);
        const seconds = Math.floor(elapsed % 60);
        document.getElementById('audioTimer').textContent = 
            `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

    } catch (error) {
        console.error('Error updating audio emotion:', error);
    }
}

function updateAudioUI(status) {
    const statusElement = document.getElementById('audioStatus');
    const statusClass = status === 'recording' ? 'recording' : 
                        status === 'completed' ? 'completed' : '';

    statusElement.textContent = 
        status === 'recording' ? 'Recording...' :
        status === 'completed' ? 'Completed' : 'Ready';

    statusElement.className = `status ${statusClass}`;
}

async function showAudioResults() {
    try {
        const response = await fetch('/get_results');
        const results = await response.json();

        if (results.success) {
            const resultsDiv = document.getElementById('audioResults');
            const contentDiv = document.getElementById('audioResultsContent');

            let html = '<div class="result-item"><strong>Audio Analysis Summary:</strong></div>';

            for (const [key, value] of Object.entries(results.data || {})) {
                html += `
                    <div class="result-item">
                        <div class="result-label">${formatKey(key)}</div>
                        <div class="result-value">${formatValue(value)}</div>
                    </div>
                `;
            }

            contentDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatKey(key) {
    return key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

function formatValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(2);
    }
    if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
    }
    return String(value);
}

function showError(message) {
    alert(message);
}

function downloadResults(type) {
    const data = type === 'video' ? state.video.emotions : state.audio.emotions;
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${type}-emotions-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
}
