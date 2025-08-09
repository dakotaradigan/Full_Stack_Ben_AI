// Chat Manager
class ChatManager {
    constructor(app) {
        this.app = app;
        this.isProcessing = false;
    }
    
    async sendMessage(message) {
        if (!message || this.isProcessing) return;
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Disable send button
        this.setProcessingState(true);
        
        try {
            // Try WebSocket first if connected
            if (this.app.websocket && this.app.websocket.isConnected) {
                this.app.websocket.sendMessage(message);
            } else {
                // Fallback to REST API
                await this.sendViaAPI(message);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.app.showError('Failed to send message. Please try again.');
            this.hideTypingIndicator();
            this.setProcessingState(false);
        }
    }
    
    async sendViaAPI(message) {
        try {
            const response = await fetch(`${this.app.apiBaseUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.app.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();
            
            // Add assistant response
            this.addMessage('assistant', data.response, data.function_calls);
            
            // Update session ID if needed
            if (data.session_id) {
                this.app.sessionId = data.session_id;
            }
            
        } catch (error) {
            console.error('API error:', error);
            throw error;
        } finally {
            this.setProcessingState(false);
        }
    }
    
    addMessage(role, content, functionCalls = null) {
        const messagesContainer = this.app.messagesContainer;
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        // Create avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = role === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<span>✨</span>';
        
        // Create content container
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Create message bubble
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        
        // Process content (handle markdown, code blocks, etc.)
        bubbleDiv.innerHTML = this.processMessageContent(content);
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = this.app.formatTime(new Date());
        
        // Add function call indicator if present
        if (functionCalls && functionCalls.length > 0) {
            const funcDiv = document.createElement('div');
            funcDiv.className = 'function-indicator';
            funcDiv.innerHTML = `<i class="fas fa-cog"></i> Used ${functionCalls[0].name}`;
            contentDiv.appendChild(funcDiv);
        }
        
        // Assemble message
        contentDiv.appendChild(bubbleDiv);
        contentDiv.appendChild(timeDiv);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        // Add to container
        messagesContainer.appendChild(messageDiv);
        
        // Add to history
        this.app.messageHistory.push({
            role: role,
            content: content,
            timestamp: new Date().toISOString()
        });
        
        // Force immediate scroll after DOM manipulation
        requestAnimationFrame(() => {
            this.scrollToBottom();
        });
    }
    
    processMessageContent(content) {
        // Escape HTML
        let processed = this.escapeHtml(content);
        
        // Convert markdown bold
        processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convert markdown italic
        processed = processed.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Convert code blocks
        processed = processed.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Convert inline code
        processed = processed.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Convert line breaks
        processed = processed.replace(/\n/g, '<br>');
        
        // Convert URLs to links
        processed = processed.replace(
            /(https?:\/\/[^\s]+)/g,
            '<a href="$1" target="_blank" rel="noopener">$1</a>'
        );
        
        // Highlight dollar amounts
        processed = processed.replace(
            /\$[\d,]+(?:\.\d{2})?/g,
            '<span class="amount">$&</span>'
        );
        
        return processed;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showTypingIndicator() {
        this.app.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.app.typingIndicator.style.display = 'none';
    }
    
    setProcessingState(isProcessing) {
        this.isProcessing = isProcessing;
        this.app.chatSendBtn.disabled = isProcessing;
        this.app.chatInput.disabled = isProcessing;
        
        if (isProcessing) {
            this.app.chatSendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
        } else {
            this.app.chatSendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }
    
    scrollToBottom() {
        const container = this.app.messagesContainer.parentElement; // chat-container
        if (!container) return;
        
        // Use requestAnimationFrame for better performance and multiple attempts
        requestAnimationFrame(() => {
            // Scroll immediately
            container.scrollTop = container.scrollHeight;
            
            // Then scroll again after a short delay to handle any layout shifts
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 100);
            
            // Final scroll to handle any async content loading
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 300);
        });
    }
    
    receiveMessage(data) {
        // Hide typing indicator
        this.hideTypingIndicator();
        
        // Add message
        if (data.type === 'message') {
            this.addMessage('assistant', data.data.response, data.data.function_calls);
            // Ensure scroll after WebSocket message
            this.scrollToBottom();
        }
        
        // Re-enable input
        this.setProcessingState(false);
    }
    
    clearChat() {
        this.app.messagesContainer.innerHTML = '';
        this.app.messageHistory = [];
    }
}