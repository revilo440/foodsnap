// Live Modal endpoint - using ASGI app pattern like Step 3
// Replace with your actual Modal endpoint after deployment
const API_ENDPOINT = 'https://YOUR_USERNAME--foodsnap-fastapi-app.modal.run/analyze';

// Initialize error handler and loading manager
let errorHandler = null;
let loadingManager = null;

if (typeof ErrorHandler !== 'undefined') {
    errorHandler = new ErrorHandler();
}
if (typeof LoadingStateManager !== 'undefined') {
    loadingManager = new LoadingStateManager();
}

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImage = document.getElementById('removeImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusMessage = document.getElementById('statusMessage');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');

// Cache elements
const cacheStatsBtn = document.getElementById('cacheStatsBtn');
const cacheInfo = document.getElementById('cacheInfo');

let selectedFile = null;

// File upload handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

removeImage.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.classList.add('hidden');
    uploadArea.classList.remove('hidden');
    analyzeBtn.disabled = true;
    hideResults();
});

function handleFileSelect(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    
    hideResults();
}

// Analysis
analyzeBtn.addEventListener('click', async function analyzeImage() {
    if (!selectedFile) return;
    
    // Disable button and show loading
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
    analyzeBtn.querySelector('.btn-loader').classList.remove('hidden');
    
    // Hide previous results
    results.classList.add('hidden');
    error.classList.add('hidden');
    
    // Start loading state manager if available
    if (loadingManager) {
        loadingManager.start();
    } else {
        showStatus('Uploading image for analysis...', 'info');
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        const elapsed = Date.now() - startTime;
        
        if (data.error) {
            showError(data.error);
        } else {
            // Stop loading manager
            if (loadingManager) {
                loadingManager.stop();
            }
            
            displayResults(data);
            showStatus('Analysis complete!', 'success');
        }
    } catch (err) {
        console.error('Analysis error:', err);
        
        // Stop loading manager
        if (loadingManager) {
            loadingManager.stop();
        }
        
        // Use enhanced error handler if available
        if (errorHandler) {
            errorHandler.displayError(err, err.status);
        } else {
            errorMessage.textContent = err.message || 'Failed to analyze image';
            error.classList.remove('hidden');
            showStatus('Analysis failed. Please try again.', 'error');
        }
    } finally {
        // Re-enable button
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Food';
        analyzeBtn.querySelector('.btn-loader').classList.add('hidden');
    }
});

function displayResults(data) {
    // Show results section with animation
    const resultsSection = document.getElementById('results');
    resultsSection.classList.remove('hidden');
    resultsSection.style.animation = 'fadeIn 0.5s ease-out';
    document.getElementById('error').classList.add('hidden');
    
    document.getElementById('dishName').textContent = data.dish_name || 'Unknown Dish';
    document.getElementById('description').textContent = data.description || '';
    document.getElementById('cuisine').textContent = `🍴 ${data.cuisine || 'Unknown'}`;
    // Caption removed for production - judges don't need to see BLIP output
    // document.getElementById('caption').textContent = data.caption || '';
    
    // Display confidence with animation
    const confidenceEl = document.getElementById('confidence');
    const confidence = data.confidence || 0;
    const confidencePercent = Math.round(confidence * 100);
    
    // Animated counter effect
    let currentPercent = 0;
    const increment = confidencePercent / 20;
    const counter = setInterval(() => {
        currentPercent += increment;
        if (currentPercent >= confidencePercent) {
            currentPercent = confidencePercent;
            clearInterval(counter);
        }
        confidenceEl.textContent = `${Math.round(currentPercent)}% confidence`;
    }, 50);
    
    // Set confidence color with animation
    if (confidence >= 0.7) {
        confidenceEl.className = 'confidence-badge high';
    } else if (confidence >= 0.4) {
        confidenceEl.className = 'confidence-badge medium';
    } else {
        confidenceEl.className = 'confidence-badge low';
    }
    
    // Cache badge
    const cached = document.getElementById('cached');
    if (data.cached) {
        cached.classList.remove('hidden');
    } else {
        cached.classList.add('hidden');
    }
    
    // Display timing with animation
    const timingEl = document.getElementById('timing');
    if (data.timings_ms && data.timings_ms.total_ms) {
        timingEl.textContent = `⚡ ${(data.timings_ms.total_ms / 1000).toFixed(1)}s`;
        timingEl.classList.remove('hidden');
        timingEl.style.animation = 'slideIn 0.5s ease-out 0.2s both';
    }
    
    // Recipe info
    if (data.recipe) {
        document.getElementById('difficulty').textContent = `📊 ${data.recipe.difficulty || 'medium'}`;
        document.getElementById('servings').textContent = `👥 ${data.recipe.servings || 1} servings`;
        document.getElementById('prepTime').textContent = `⏱️ Prep: ${data.recipe.prep_time || 'unknown'}`;
        document.getElementById('cookTime').textContent = `🔥 Cook: ${data.recipe.cook_time || 'unknown'}`;
        
        // Instructions
        const instructionsList = document.getElementById('instructions');
        instructionsList.innerHTML = '';
        if (data.recipe.instructions && Array.isArray(data.recipe.instructions)) {
            data.recipe.instructions.forEach(step => {
                const li = document.createElement('li');
                // Simply display the instruction as-is, CSS handles numbering
                li.textContent = step.trim();
                instructionsList.appendChild(li);
            });
        }
    }
    
    // Tags
    const tagsContainer = document.getElementById('tags');
    tagsContainer.innerHTML = '';
    if (data.tags && Array.isArray(data.tags)) {
        data.tags.forEach(tag => {
            const span = document.createElement('span');
            span.className = 'tag';
            span.textContent = tag;
            tagsContainer.appendChild(span);
        });
    }
    
    // Ingredients
    const ingredientsList = document.getElementById('ingredientsList');
    ingredientsList.innerHTML = '';
    if (data.ingredients && Array.isArray(data.ingredients)) {
        data.ingredients.forEach(ing => {
            const li = document.createElement('li');
            const name = ing.name || 'Unknown';
            const amount = ing.amount || '';
            const unit = ing.unit || '';
            const optional = ing.optional ? ' (optional)' : '';
            
            li.innerHTML = `
                <span class="ingredient-name">${name}</span>
                <span class="ingredient-amount">${amount} ${unit}${optional}</span>
            `;
            ingredientsList.appendChild(li);
        });
    }
    
    // Nutrition
    const nutritionGrid = document.getElementById('nutritionGrid');
    nutritionGrid.innerHTML = '';
    if (data.nutrition) {
        Object.entries(data.nutrition).forEach(([key, value], index) => {
            if (key !== 'allergens') {
                const item = document.createElement('div');
                item.className = 'nutrition-item';
                item.style.animation = `fadeIn 0.3s ease-out ${index * 0.1}s both`;
                
                // Add emoji icons for nutrition items
                const icons = {
                    calories: '🔥',
                    protein: '💪',
                    carbs: '🌾',
                    fat: '🥑',
                    fiber: '🌿',
                    sugar: '🍯',
                    sodium: '🧂'
                };
                
                item.innerHTML = `
                    <span class="nutrition-label">${icons[key] || ''} ${key.replace(/_/g, ' ').toUpperCase()}</span>
                    <span class="nutrition-value">${value}</span>
                `;
                nutritionGrid.appendChild(item);
            }
        });
    }
    
    // Allergens
    const allergensDiv = document.getElementById('allergens');
    allergensDiv.innerHTML = '';
    if (data.allergens && Array.isArray(data.allergens)) {
        if (data.allergens.length > 0) {
            data.allergens.forEach(allergen => {
                const span = document.createElement('span');
                span.className = 'allergen-badge';
                span.textContent = `⚠️ ${allergen}`;
                allergensDiv.appendChild(span);
            });
        } else {
            allergensDiv.innerHTML = '<span class="no-allergens">No common allergens detected</span>';
        }
    }
    

}

function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    statusDiv.classList.remove('hidden');
    
    // Add fade-in animation
    statusDiv.style.animation = 'fadeIn 0.3s ease-out';
    
    // Auto-hide success messages after 3 seconds
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.style.animation = 'fadeOut 0.3s ease-out';
            setTimeout(() => statusDiv.classList.add('hidden'), 300);
        }, 3000);
    }
}

function showError(message) {
    errorMessage.textContent = message;
    error.classList.remove('hidden');
}

function hideError() {
    error.classList.add('hidden');
}

function hideResults() {
    results.classList.add('hidden');
    statusMessage.classList.add('hidden');
}

// Tab functionality
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;
        
        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Show corresponding panel
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
    });
});

// Cache stats
cacheStatsBtn.addEventListener('click', async () => {
    try {
        const response = await fetch(API_ENDPOINT.replace('/analyze', '/cache/stats'));
        const stats = await response.json();
        
        const hitRate = stats.hits + stats.misses > 0 
            ? ((stats.hits / (stats.hits + stats.misses)) * 100).toFixed(1)
            : 0;
            
        cacheInfo.innerHTML = `
            Entries: ${stats.entries || 0} | 
            Hits: ${stats.hits || 0} | 
            Misses: ${stats.misses || 0} | 
            Hit Rate: ${hitRate}%
        `;
        cacheInfo.classList.remove('hidden');
        
        setTimeout(() => {
            cacheInfo.classList.add('hidden');
        }, 5000);
    } catch (err) {
        console.error('Failed to fetch cache stats:', err);
    }
});
