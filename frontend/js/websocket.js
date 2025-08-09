// WebSocket Manager
class WebSocketManager {
    constructor(app) {
        this.app = app;
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        
        this.connect();
    }
    
    connect() {
        try {
            const wsUrl = `${this.app.wsUrl}/ws/${this.app.sessionId}`;
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => this.onOpen();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onClose();
            this.ws.onerror = (error) => this.onError(error);
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }
    
    onOpen() {
        console.log('WebSocket connected');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        
        // Show connection status
        this.app.showConnectionStatus('Real-time connection established', 'success');
        
        // Start heartbeat
        this.startHeartbeat();
    }
    
    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'message':
                    this.app.chat.receiveMessage(data);
                    break;
                    
                case 'typing':
                    if (data.status === 'start') {
                        this.app.chat.showTypingIndicator();
                    } else {
                        this.app.chat.hideTypingIndicator();
                    }
                    break;
                    
                case 'error':
                    this.app.showError(data.message || 'An error occurred');
                    this.app.chat.hideTypingIndicator();
                    this.app.chat.setProcessingState(false);
                    break;
                    
                case 'pong':
                    // Heartbeat response
                    break;
                    
                default:
                    console.log('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error processing message:', error);
        }
    }
    
    onClose() {
        console.log('WebSocket disconnected');
        this.isConnected = false;
        
        // Stop heartbeat
        this.stopHeartbeat();
        
        // Show disconnection status
        this.app.showConnectionStatus('Connection lost. Reconnecting...', 'error');
        
        // Attempt to reconnect
        this.scheduleReconnect();
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.isConnected = false;
    }
    
    sendMessage(message) {
        if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            // Fallback to API
            this.app.chat.sendViaAPI(message);
            return;
        }
        
        try {
            const data = {
                type: 'message',
                message: message,
                timestamp: new Date().toISOString()
            };
            
            this.ws.send(JSON.stringify(data));
        } catch (error) {
            console.error('Error sending WebSocket message:', error);
            // Fallback to API
            this.app.chat.sendViaAPI(message);
        }
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached');
            this.app.showConnectionStatus('Unable to establish real-time connection. Using standard mode.', 'error');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect();
        }, delay);
    }
    
    reconnect() {
        this.disconnect();
        this.reconnectAttempts = 0;
        this.connect();
    }
    
    disconnect() {
        this.stopHeartbeat();
        
        if (this.ws) {
            this.ws.onclose = null; // Prevent reconnection
            this.ws.close();
            this.ws = null;
        }
        
        this.isConnected = false;
    }
    
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Send ping every 30 seconds
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
}