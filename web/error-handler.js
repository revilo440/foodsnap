// Enhanced Error Handling for FoodSnap

class ErrorHandler {
    constructor() {
        this.errorMessages = {
            'network': {
                title: '🌐 Connection Issue',
                message: 'Unable to connect to the analysis server. Please check your internet connection and try again.',
                icon: '📡',
                suggestions: [
                    'Check your internet connection',
                    'Try refreshing the page',
                    'Use a demo image to test'
                ]
            },
            'timeout': {
                title: '⏱️ Request Timeout',
                message: 'The analysis is taking longer than expected. The server might be under heavy load.',
                icon: '⏳',
                suggestions: [
                    'Try again in a few moments',
                    'Use a smaller image',
                    'Check the server status'
                ]
            },
            'invalid_image': {
                title: '🖼️ Invalid Image',
                message: 'The uploaded file doesn\'t appear to be a valid image or is corrupted.',
                icon: '❌',
                suggestions: [
                    'Use JPG, PNG, or WebP format',
                    'Ensure file size is under 10MB',
                    'Try a different image'
                ]
            },
            'server_error': {
                title: '🔧 Server Error',
                message: 'The analysis server encountered an unexpected error.',
                icon: '⚠️',
                suggestions: [
                    'Wait a moment and try again',
                    'Contact support if the issue persists',
                    'Try using a demo image'
                ]
            },
            'model_error': {
                title: '🤖 AI Model Error',
                message: 'The AI model failed to process your image properly.',
                icon: '🤯',
                suggestions: [
                    'Try a clearer food image',
                    'Ensure the image contains food',
                    'Use a demo image to verify the system works'
                ]
            }
        };
    }

    determineErrorType(error, statusCode) {
        if (!navigator.onLine) return 'network';
        if (statusCode === 408 || error.message?.includes('timeout')) return 'timeout';
        if (statusCode === 415 || error.message?.includes('image')) return 'invalid_image';
        if (statusCode >= 500) return 'server_error';
        if (error.message?.includes('model') || error.message?.includes('LLM')) return 'model_error';
        return 'server_error';
    }

    displayError(error, statusCode = null) {
        const errorType = this.determineErrorType(error, statusCode);
        const errorInfo = this.errorMessages[errorType];
        
        const errorHTML = `
            <div class="error-container animated">
                <div class="error-icon">${errorInfo.icon}</div>
                <h2 class="error-title">${errorInfo.title}</h2>
                <p class="error-message">${errorInfo.message}</p>
                
                <div class="error-suggestions">
                    <h4>What you can try:</h4>
                    <ul>
                        ${errorInfo.suggestions.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
                
                <div class="error-actions">
                    <button onclick="location.reload()" class="error-btn primary">
                        🔄 Refresh Page
                    </button>
                    <button onclick="showDemoModal()" class="error-btn secondary">
                        🎬 Try Demo
                    </button>
                </div>
                
                <details class="error-details">
                    <summary>Technical Details</summary>
                    <pre>${JSON.stringify({
                        type: errorType,
                        status: statusCode,
                        message: error.message,
                        timestamp: new Date().toISOString()
                    }, null, 2)}</pre>
                </details>
            </div>
        `;
        
        const errorSection = document.getElementById('error');
        errorSection.innerHTML = errorHTML;
        errorSection.classList.remove('hidden');
        
        // Hide other sections
        document.getElementById('results').classList.add('hidden');
        document.getElementById('statusMessage').classList.add('hidden');
        
        // Log to console for debugging
        console.error('FoodSnap Error:', {
            type: errorType,
            error: error,
            statusCode: statusCode
        });
    }
}

// Loading State Manager
class LoadingStateManager {
    constructor() {
        this.stages = [
            { id: 'upload', message: '📤 Uploading image...', duration: 2000 },
            { id: 'caption', message: '📸 Analyzing visual features...', duration: 3000 },
            { id: 'llm', message: '🤖 Generating detailed analysis...', duration: 15000 },
            { id: 'format', message: '📊 Formatting results...', duration: 1000 }
        ];
        this.currentStage = 0;
        this.interval = null;
    }

    start() {
        this.currentStage = 0;
        this.showStage(0);
        
        let elapsed = 0;
        this.interval = setInterval(() => {
            elapsed += 100;
            
            // Update progress for current stage
            const stage = this.stages[this.currentStage];
            if (stage && elapsed >= stage.duration && this.currentStage < this.stages.length - 1) {
                this.currentStage++;
                this.showStage(this.currentStage);
                elapsed = 0;
            }
            
            // Update progress bar
            this.updateProgress(elapsed, stage?.duration || 1000);
        }, 100);
    }

    showStage(index) {
        const stage = this.stages[index];
        if (!stage) return;
        
        const statusDiv = document.getElementById('statusMessage');
        statusDiv.innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <span class="loading-message">${stage.message}</span>
                <div class="loading-progress">
                    <div class="loading-progress-bar" id="progressBar"></div>
                </div>
                <div class="loading-stage">Step ${index + 1} of ${this.stages.length}</div>
            </div>
        `;
        statusDiv.className = 'status-message info';
        statusDiv.classList.remove('hidden');
    }

    updateProgress(elapsed, total) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            const percentage = Math.min((elapsed / total) * 100, 100);
            progressBar.style.width = `${percentage}%`;
        }
    }

    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }
}

// Export for use in main app
window.ErrorHandler = ErrorHandler;
window.LoadingStateManager = LoadingStateManager;
