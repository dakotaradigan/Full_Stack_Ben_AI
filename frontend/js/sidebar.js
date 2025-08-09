// Sidebar Manager
class SidebarManager {
    constructor(app) {
        this.app = app;
        this.sidebar = document.getElementById('sidebar');
        this.closeBtn = document.getElementById('closeSidebar');
        this.isOpen = false;
        
        this.init();
    }
    
    init() {
        // Close button
        this.closeBtn.addEventListener('click', () => this.close());
        
        // Click outside to close
        document.addEventListener('click', (e) => {
            if (this.isOpen && 
                !this.sidebar.contains(e.target) && 
                !this.app.toggleSidebarBtn.contains(e.target)) {
                this.close();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Escape to close
            if (e.key === 'Escape' && this.isOpen) {
                this.close();
            }
            
            // Cmd/Ctrl + K to toggle
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                this.toggle();
            }
        });
        
        // Swipe gestures for mobile
        this.setupSwipeGestures();
    }
    
    open() {
        this.sidebar.classList.add('open');
        this.isOpen = true;
        
        // Add overlay on mobile
        if (window.innerWidth <= 768) {
            this.addOverlay();
        }
    }
    
    close() {
        this.sidebar.classList.remove('open');
        this.isOpen = false;
        this.removeOverlay();
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    addOverlay() {
        if (!this.overlay) {
            this.overlay = document.createElement('div');
            this.overlay.className = 'sidebar-overlay';
            this.overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
                opacity: 0;
                transition: opacity 0.3s ease;
            `;
            
            this.overlay.addEventListener('click', () => this.close());
            document.body.appendChild(this.overlay);
            
            // Trigger animation
            setTimeout(() => {
                this.overlay.style.opacity = '1';
            }, 10);
        }
    }
    
    removeOverlay() {
        if (this.overlay) {
            this.overlay.style.opacity = '0';
            setTimeout(() => {
                if (this.overlay && this.overlay.parentNode) {
                    this.overlay.parentNode.removeChild(this.overlay);
                    this.overlay = null;
                }
            }, 300);
        }
    }
    
    setupSwipeGestures() {
        let touchStartX = 0;
        let touchEndX = 0;
        const threshold = 50;
        
        // Detect swipe start
        document.addEventListener('touchstart', (e) => {
            touchStartX = e.changedTouches[0].screenX;
        });
        
        // Detect swipe end
        document.addEventListener('touchend', (e) => {
            touchEndX = e.changedTouches[0].screenX;
            this.handleSwipe(touchStartX, touchEndX, threshold);
        });
    }
    
    handleSwipe(startX, endX, threshold) {
        const diff = endX - startX;
        
        // Swipe right to open (from left edge)
        if (diff > threshold && startX < 50 && !this.isOpen) {
            this.open();
        }
        
        // Swipe left to close
        if (diff < -threshold && this.isOpen) {
            this.close();
        }
    }
    
    highlightSuggestion(query) {
        // Find and highlight matching suggestion
        const suggestions = this.sidebar.querySelectorAll('.suggestion-item');
        suggestions.forEach(item => {
            if (item.textContent.toLowerCase().includes(query.toLowerCase())) {
                item.classList.add('highlighted');
                setTimeout(() => {
                    item.classList.remove('highlighted');
                }, 2000);
            }
        });
    }
}

// Add CSS for highlighted suggestion
const style = document.createElement('style');
style.textContent = `
    .suggestion-item.highlighted {
        background: var(--primary-light) !important;
        animation: pulse 0.5s ease;
    }
    
    .sidebar-overlay {
        backdrop-filter: blur(4px);
    }
    
    @media (min-width: 769px) {
        .sidebar {
            position: relative;
            transform: translateX(0);
            box-shadow: none;
        }
        
        .sidebar.open {
            transform: translateX(0);
        }
        
        .main-content {
            margin-left: 0;
        }
        
        .sidebar.open ~ .main-content {
            margin-left: 280px;
        }
    }
`;
document.head.appendChild(style);