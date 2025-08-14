// Demo Mode for FoodSnap - Sample Images for Quick Testing

const DEMO_IMAGES = [
    {
        name: 'Caesar Salad',
        url: 'https://images.unsplash.com/photo-1550304943-4f24f54ddde9?w=800&auto=format&fit=crop',
        description: 'Fresh caesar salad with grilled chicken'
    },
    {
        name: 'Sushi Platter',
        url: 'https://images.unsplash.com/photo-1579584425555-c3ce17fd4351?w=800&auto=format&fit=crop',
        description: 'Assorted sushi rolls and sashimi'
    },
    {
        name: 'Pizza Margherita',
        url: 'https://images.unsplash.com/photo-1604068549290-dea0e4a305ca?w=800&auto=format&fit=crop',
        description: 'Classic Italian pizza with fresh basil'
    },
    {
        name: 'Burger & Fries',
        url: 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800&auto=format&fit=crop',
        description: 'Gourmet burger with crispy fries'
    },
    {
        name: 'Pasta Carbonara',
        url: 'https://images.unsplash.com/photo-1612874742237-6526221588e3?w=800&auto=format&fit=crop',
        description: 'Creamy carbonara with pancetta'
    },
    {
        name: 'Tacos',
        url: 'https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=800&auto=format&fit=crop',
        description: 'Mexican street tacos with lime'
    }
];

// Add demo mode UI
function initDemoMode() {
    // Create demo button in header
    const header = document.querySelector('header');
    const demoButton = document.createElement('button');
    demoButton.className = 'demo-btn';
    demoButton.innerHTML = '🎬 Try Demo Images';
    demoButton.onclick = showDemoModal;
    header.appendChild(demoButton);
    
    // Create demo modal
    const modal = document.createElement('div');
    modal.id = 'demoModal';
    modal.className = 'demo-modal hidden';
    modal.innerHTML = `
        <div class="demo-modal-content">
            <div class="demo-modal-header">
                <h2>🍽️ Sample Food Images</h2>
                <button class="demo-close" onclick="closeDemoModal()">×</button>
            </div>
            <div class="demo-grid">
                ${DEMO_IMAGES.map((img, index) => `
                    <div class="demo-item" onclick="loadDemoImage(${index})">
                        <img src="${img.url}" alt="${img.name}" loading="lazy">
                        <div class="demo-item-overlay">
                            <h3>${img.name}</h3>
                            <p>${img.description}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

function showDemoModal() {
    document.getElementById('demoModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeDemoModal() {
    document.getElementById('demoModal').classList.add('hidden');
    document.body.style.overflow = '';
}

async function loadDemoImage(index) {
    const demo = DEMO_IMAGES[index];
    closeDemoModal();
    
    // Show loading state
    showStatus(`Loading demo: ${demo.name}...`, 'info');
    
    try {
        // Fetch image as blob
        const response = await fetch(demo.url);
        const blob = await response.blob();
        
        // Create File object
        const file = new File([blob], `${demo.name.toLowerCase().replace(/\s+/g, '-')}.jpg`, {
            type: 'image/jpeg'
        });
        
        // Load into preview
        handleFileSelect(file);
        
        showStatus(`Demo image loaded: ${demo.name}`, 'success');
    } catch (err) {
        showStatus('Failed to load demo image', 'error');
        console.error('Demo load error:', err);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initDemoMode);
