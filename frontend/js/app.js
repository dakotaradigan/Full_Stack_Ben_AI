// Main Application Controller
class BenAIApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.isConnected = false;
        this.messageHistory = [];
        this.currentView = 'welcome'; // 'welcome' or 'chat'
        this.apiBaseUrl = 'http://localhost:8000';
        this.wsUrl = 'ws://localhost:8000';
        
        this.init();
    }
    
    init() {
        this.setupElements();
        this.setupEventListeners();
        this.checkBackendConnection();
        this.loadSuggestions();
        
        // Initialize other modules
        this.chat = new ChatManager(this);
        this.websocket = new WebSocketManager(this);
        this.sidebar = new SidebarManager(this);
    }
    
    setupElements() {
        // Views
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.chatInterface = document.getElementById('chatInterface');
        
        // Inputs
        this.welcomeInput = document.getElementById('welcomeInput');
        this.chatInput = document.getElementById('chatInput');
        
        // Buttons
        this.welcomeSendBtn = document.getElementById('welcomeSend');
        this.chatSendBtn = document.getElementById('chatSend');
        this.newChatBtn = document.getElementById('newChat');
        this.toggleSidebarBtn = document.getElementById('toggleSidebar');
        
        // Messages
        this.messagesContainer = document.getElementById('messagesContainer');
        
        // Toasts
        this.connectionToast = document.getElementById('connectionToast');
        this.errorToast = document.getElementById('errorToast');
        
        // Other
        this.charCount = document.getElementById('charCount');
        this.typingIndicator = document.getElementById('typingIndicator');
    }
    
    setupEventListeners() {
        // Welcome screen
        this.welcomeSendBtn.addEventListener('click', () => this.sendFromWelcome());
        this.welcomeInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendFromWelcome();
            }
        });
        
        // Chat interface
        this.chatSendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Character count
        this.chatInput.addEventListener('input', () => {
            const length = this.chatInput.value.length;
            this.charCount.textContent = `${length} / 5000`;
            
            // Auto-resize textarea
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        });
        
        // New chat button
        this.newChatBtn.addEventListener('click', () => this.startNewChat());
        
        // Sidebar toggle
        this.toggleSidebarBtn.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });
        
        // Tab key for suggestions
        this.welcomeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                document.getElementById('sidebar').classList.add('open');
            }
        });
    }
    
    async checkBackendConnection() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.isConnected = true;
                this.showConnectionStatus('Connected', 'success');
            } else {
                this.showConnectionStatus('Backend unavailable', 'error');
            }
        } catch (error) {
            console.error('Connection check failed:', error);
            this.showConnectionStatus('Connection failed', 'error');
        }
    }
    
    async loadSuggestions() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/suggestions`);
            const suggestions = await response.json();
            
            // Populate suggestion sections
            this.populateSuggestionSection('gettingStartedSuggestions', suggestions.getting_started);
            this.populateSuggestionSection('portfolioSuggestions', suggestions.use_cases);
            this.populateSuggestionSection('benchmarkSuggestions', suggestions.benchmark_search);
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    }
    
    populateSuggestionSection(elementId, suggestions) {
        const container = document.getElementById(elementId);
        if (!container) return;
        
        container.innerHTML = '';
        suggestions.forEach(suggestion => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.textContent = suggestion.title;
            item.addEventListener('click', () => {
                this.submitQuery(suggestion.query);
            });
            container.appendChild(item);
        });
    }
    
    submitQuery(query) {
        if (this.currentView === 'welcome') {
            this.welcomeInput.value = query;
            this.sendFromWelcome();
        } else {
            this.chatInput.value = query;
            this.sendMessage();
        }
        
        // Close sidebar on mobile
        if (window.innerWidth <= 768) {
            document.getElementById('sidebar').classList.remove('open');
        }
    }
    
    sendFromWelcome() {
        const message = this.welcomeInput.value.trim();
        if (!message) return;
        
        // Switch to chat view
        this.switchToChatView();
        
        // Send message
        this.chat.sendMessage(message);
        
        // Clear input
        this.welcomeInput.value = '';
    }
    
    sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;
        
        // Send message
        this.chat.sendMessage(message);
        
        // Clear input and reset height
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        this.charCount.textContent = '0 / 5000';
    }
    
    switchToChatView() {
        this.currentView = 'chat';
        this.welcomeScreen.style.display = 'none';
        this.chatInterface.style.display = 'flex';
        
        // Focus chat input
        setTimeout(() => this.chatInput.focus(), 100);
    }
    
    switchToWelcomeView() {
        this.currentView = 'welcome';
        this.welcomeScreen.style.display = 'flex';
        this.chatInterface.style.display = 'none';
        
        // Clear messages
        this.messagesContainer.innerHTML = '';
        this.messageHistory = [];
        
        // Focus welcome input
        setTimeout(() => this.welcomeInput.focus(), 100);
    }
    
    startNewChat() {
        // Generate new session ID
        this.sessionId = this.generateSessionId();
        
        // Reset WebSocket connection
        if (this.websocket) {
            this.websocket.reconnect();
        }
        
        // Switch to welcome view
        this.switchToWelcomeView();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    showConnectionStatus(message, type = 'info') {
        const toast = this.connectionToast;
        const text = document.getElementById('connectionText');
        
        text.textContent = message;
        toast.style.display = 'flex';
        
        if (type === 'success') {
            toast.style.background = '#d4edda';
            toast.style.color = '#155724';
        } else if (type === 'error') {
            toast.style.background = '#f8d7da';
            toast.style.color = '#721c24';
        }
        
        setTimeout(() => {
            toast.style.display = 'none';
        }, 3000);
    }
    
    showError(message) {
        const toast = this.errorToast;
        const text = document.getElementById('errorText');
        
        text.textContent = message;
        toast.style.display = 'flex';
        
        setTimeout(() => {
            toast.style.display = 'none';
        }, 5000);
    }
    
    formatTime(date) {
        return new Intl.DateTimeFormat('en-US', {
            hour: 'numeric',
            minute: 'numeric',
            hour12: true
        }).format(date);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.benAI = new BenAIApp();
});