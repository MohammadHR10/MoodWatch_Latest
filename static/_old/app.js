// VoiceVibe Audio Analyzer JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileWrapper = document.getElementById('fileInputWrapper');
    const fileInput = document.getElementById('audioInput');
    const fileText = document.getElementById('fileText');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');

    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }

    // Drag and drop functionality
    if (fileWrapper) {
        setupDragAndDrop();
    }

    // Form submission handler
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFormSubmit);
    }

    // Initialize tooltips and animations
    initializeAnimations();
});

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        displaySelectedFile(file);
        validateFile(file);
    }
}

function displaySelectedFile(file) {
    const fileText = document.getElementById('fileText');
    const fileWrapper = document.getElementById('fileInputWrapper');
    
    if (fileText && fileWrapper) {
        fileText.textContent = `📁 ${file.name}`;
        fileWrapper.style.borderColor = '#4caf50';
        fileWrapper.style.background = '#f0fff0';
        
        // Add file size info
        const sizeInfo = document.createElement('div');
        sizeInfo.className = 'file-size-info';
        sizeInfo.style.fontSize = '0.8rem';
        sizeInfo.style.color = '#666';
        sizeInfo.style.marginTop = '5px';
        sizeInfo.textContent = `Size: ${formatFileSize(file.size)}`;
        
        // Remove existing size info
        const existingSize = document.querySelector('.file-size-info');
        if (existingSize) {
            existingSize.remove();
        }
        
        document.querySelector('.file-input-content').appendChild(sizeInfo);
    }
}

function validateFile(file) {
    const allowedTypes = ['.mp3', '.wav', '.m4a', '.flac', '.ogg'];
    const maxSize = 100 * 1024 * 1024; // 100MB
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    
    let isValid = true;
    let errorMessage = '';
    
    // Check file type
    if (!allowedTypes.includes(fileExt)) {
        isValid = false;
        errorMessage = `Invalid file type. Allowed types: ${allowedTypes.join(', ')}`;
    }
    
    // Check file size
    if (file.size > maxSize) {
        isValid = false;
        errorMessage = `File too large. Maximum size: ${formatFileSize(maxSize)}`;
    }
    
    // Check minimum size
    if (file.size < 1024) {
        isValid = false;
        errorMessage = 'File too small. Please upload a valid audio file.';
    }
    
    // Display validation result
    displayValidationResult(isValid, errorMessage);
    
    return isValid;
}

function displayValidationResult(isValid, errorMessage) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const existingError = document.querySelector('.validation-error');
    
    // Remove existing error message
    if (existingError) {
        existingError.remove();
    }
    
    if (!isValid) {
        // Show error
        const errorDiv = document.createElement('div');
        errorDiv.className = 'validation-error';
        errorDiv.style.color = '#f44336';
        errorDiv.style.background = '#ffebee';
        errorDiv.style.padding = '10px';
        errorDiv.style.borderRadius = '5px';
        errorDiv.style.border = '1px solid #ffcdd2';
        errorDiv.style.marginTop = '10px';
        errorDiv.textContent = `⚠️ ${errorMessage}`;
        
        document.querySelector('.upload-form').appendChild(errorDiv);
        
        // Disable submit button
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
        }
    } else {
        // Enable submit button
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    }
}

function setupDragAndDrop() {
    const fileWrapper = document.getElementById('fileInputWrapper');
    const fileInput = document.getElementById('audioInput');
    
    if (!fileWrapper || !fileInput) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileWrapper.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        fileWrapper.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileWrapper.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    fileWrapper.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        fileWrapper.classList.add('dragover');
    }

    function unhighlight(e) {
        fileWrapper.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const file = files[0];
            fileInput.files = dt.files;
            displaySelectedFile(file);
            validateFile(file);
        }
    }
}

function handleFormSubmit(event) {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileInput = document.getElementById('audioInput');
    
    // Check if file is selected
    if (!fileInput.files || fileInput.files.length === 0) {
        event.preventDefault();
        showNotification('Please select an audio file first', 'error');
        return false;
    }
    
    // Show loading state
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '🔄 Analyzing...';
        analyzeBtn.style.background = '#ccc';
    }
    
    // Show progress indicator
    showProgressIndicator();
}

function showProgressIndicator() {
    const progressDiv = document.createElement('div');
    progressDiv.id = 'progressIndicator';
    progressDiv.innerHTML = `
        <div style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        ">
            <div style="
                background: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            ">
                <div style="
                    font-size: 2rem;
                    margin-bottom: 15px;
                    animation: spin 1s linear infinite;
                ">🔄</div>
                <h3 style="color: #333; margin-bottom: 10px;">Analyzing Audio</h3>
                <p style="color: #666;">This may take a few moments...</p>
                <div style="
                    width: 200px;
                    height: 4px;
                    background: #e0e0e0;
                    border-radius: 2px;
                    margin: 15px auto;
                    overflow: hidden;
                ">
                    <div style="
                        width: 100%;
                        height: 100%;
                        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
                        animation: progress 2s ease-in-out infinite;
                    "></div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(progressDiv);
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
        max-width: 300px;
    `;
    
    const colors = {
        'info': '#2196f3',
        'success': '#4caf50',
        'error': '#f44336',
        'warning': '#ff9800'
    };
    
    notification.style.background = colors[type] || colors.info;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease-in';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    const size = (bytes / Math.pow(1024, i)).toFixed(1);
    
    return `${size} ${sizes[i]}`;
}

function initializeAnimations() {
    // Add CSS animations dynamically
    const style = document.createElement('style');
    style.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes progress {
            0% { transform: translateX(-100%); }
            50% { transform: translateX(0%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
            transition: transform 0.2s ease;
        }
        
        .result-section {
            animation: fadeInUp 0.6s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    
    document.head.appendChild(style);
}

// API interaction functions (for future use)
function analyzeAudioAPI(formData) {
    return fetch('/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .catch(error => {
        console.error('API Error:', error);
        throw error;
    });
}

// Export functions for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        validateFile,
        formatFileSize,
        showNotification
    };
}
