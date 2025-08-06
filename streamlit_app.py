import streamlit as st
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Import functions from chatbot_enhanced (same directory)
try:
    from chatbot_enhanced import (
        sanitize_input, 
        validate_response_security,
        call_function,
        FUNCTIONS,
        SYSTEM_PROMPT,
        DISCLAIMER_TEXT,
        DISCLAIMER_FREQUENCY,
        client,
        _with_retry_and_circuit_breaker,
        CHAT_MODEL,
        num_tokens_from_messages,
        trim_history,
        MAX_TOKENS_PER_REQUEST,
        get_minimum
    )
    # Backend connection status
    try:
        test_result = get_minimum("Russell 2000")
        st.sidebar.success("✅ Backend connected")
    except Exception as e:
        st.sidebar.error(f"❌ Backend error: {str(e)[:50]}...")
        
except ImportError as e:
    st.error(f"Could not import chatbot functions: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Ben - Benchmark Advisor",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for warm, Claude-inspired design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .stApp {
        background-color: #f7f5f3;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Welcome screen styling */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 50vh;
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .avatar {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #e27d60, #c94c33);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(226, 125, 96, 0.3);
        position: relative;
    }
    
    .avatar::before {
        content: "✨";
        font-size: 24px;
        color: white;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 400;
        color: #4a4a4a !important;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #6b6b6b;
        max-width: 600px;
        line-height: 1.6;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Input container */
    .input-container {
        width: 100%;
        max-width: 700px;
        position: relative;
        margin-bottom: 1rem;
    }
    
    /* Custom input styling */
    .stTextInput > div > div > input {
        border: 2px solid #e27d60 !important;
        border-radius: 25px !important;
        padding: 1rem 4rem 1rem 1.5rem !important;
        font-size: 1rem !important;
        background-color: white !important;
        color: #2d2d2d !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        line-height: 1.4 !important;
        min-height: 50px !important;
        height: auto !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #c94c33 !important;
        box-shadow: 0 4px 20px rgba(226, 125, 96, 0.2) !important;
        outline: none !important;
    }
    
    /* Integrated send button */
    .send-button {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        background: #e27d60;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        color: white;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 2px 6px rgba(226, 125, 96, 0.3);
        z-index: 10;
    }
    
    .send-button:hover {
        background: #c94c33;
        transform: translateY(-50%) scale(1.05);
        box-shadow: 0 4px 12px rgba(226, 125, 96, 0.4);
    }
    
    .send-button::before {
        content: "↑";
        font-size: 16px;
        font-weight: bold;
    }
    
    .input-hint {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Chat interface styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .message {
        margin-bottom: 1.5rem;
        display: flex;
        align-items: flex-end;
    }
    
    .message.user {
        justify-content: flex-end;
    }
    
    .message.assistant {
        justify-content: flex-start;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.25rem;
        border-radius: 20px;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        position: relative;
    }
    
    .message-bubble.user {
        background: linear-gradient(135deg, #e27d60, #c94c33);
        color: white;
        border-bottom-right-radius: 5px;
    }
    
    .message-bubble.assistant {
        background: white;
        color: #2d2d2d !important;
        border: 1px solid #f0f0f0;
        border-bottom-left-radius: 5px;
    }
    
    /* Force consistent text colors within assistant messages */
    .message-bubble.assistant * {
        color: #2d2d2d !important;
    }
    
    .message-bubble.assistant strong {
        color: #2d2d2d !important;
        font-weight: 600;
    }
    
    .message-bubble.assistant b {
        color: #2d2d2d !important;
        font-weight: 600;
    }
    
    /* Fix bullet point formatting */
    .message-bubble.assistant ul {
        margin: 0.5rem 0 !important;
        padding-left: 1.2rem !important;
    }
    
    .message-bubble.assistant li {
        margin-bottom: 0.5rem !important;
        line-height: 1.5 !important;
        color: #2d2d2d !important;
    }
    
    .message-bubble.assistant p {
        margin-bottom: 0.8rem !important;
        color: #2d2d2d !important;
    }
    
    /* Override any Streamlit markdown defaults */
    .message-bubble.assistant .stMarkdown {
        color: #2d2d2d !important;
    }
    
    .message-bubble.assistant .stMarkdown * {
        color: #2d2d2d !important;
    }
    
    /* Ensure no red or other colored text in assistant messages */
    .message-bubble.assistant span {
        color: #2d2d2d !important;
    }
    
    .message-bubble.assistant div {
        color: #2d2d2d !important;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        text-align: right;
    }
    
    .message-time.assistant {
        color: #999;
        text-align: left;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.5rem;
    }
    
    .loading-bubble {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 20px;
        border-bottom-left-radius: 5px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .loading-dots {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #e27d60;
        animation: pulse 1.4s ease-in-out infinite both;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    .loading-dot:nth-child(3) { animation-delay: 0; }
    
    @keyframes pulse {
        0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Hide streamlit input labels and containers */
    .stTextInput > label {
        display: none;
    }
    
    .stTextInput > div {
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .stTextInput > div > div {
        border: none !important;
        background: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Chat header styling */
    .chat-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .chat-header h1 {
        font-size: 1.8rem;
        font-weight: 500;
        color: #3d3d3d;
        margin: 0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .chat-header .subtitle {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: #e27d60;
        color: white;
        border: none;
        border-radius: 50px;
        width: 40px;
        height: 40px;
        padding: 0;
        font-weight: bold;
        transition: all 0.2s ease;
        font-family: 'Segoe UI', sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(226, 125, 96, 0.3);
    }
    
    .stButton > button:hover {
        background: #c94c33;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(226, 125, 96, 0.3);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatWrapper:
    """Wrapper class to integrate chatbot_enhanced functionality with Streamlit"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  # Simple display history
        if "conversation_messages" not in st.session_state:
            st.session_state.conversation_messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # Clean API messages
        if "is_welcome_screen" not in st.session_state:
            st.session_state.is_welcome_screen = True
        if "response_count" not in st.session_state:
            st.session_state.response_count = 0
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
    
    def add_display_message(self, role: str, content: str):
        """Add message to display history"""
        timestamp_str = datetime.now().strftime("%I:%M %p")
        st.session_state.chat_history.append({
            "role": role, 
            "content": content, 
            "timestamp": timestamp_str
        })
    
    def add_conversation_message(self, role: str, content: str):
        """Add message to API conversation"""
        st.session_state.conversation_messages.append({
            "role": role,
            "content": content
        })
    
    def get_chat_response(self, user_input: str) -> str:
        """Get response from the chatbot backend - simplified approach"""
        try:
            # Sanitize input
            sanitized_input = sanitize_input(user_input)
            if not sanitized_input:
                return "I didn't receive any valid input. Please try again."
            
            # Add user message to conversation
            self.add_conversation_message("user", sanitized_input)
            
            # Use the conversation messages (guaranteed clean)
            response = _with_retry_and_circuit_breaker(
                client.chat.completions.create,
                model=CHAT_MODEL,
                messages=st.session_state.conversation_messages,
                tools=[{"type": "function", "function": func} for func in FUNCTIONS],
                tool_choice="auto",
                max_tokens=2000,
                temperature=0.1,
            )
            
            msg = response.choices[0].message
            
            # Handle tool calls
            if msg.tool_calls:
                # Add assistant message with tool calls
                st.session_state.conversation_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": msg.tool_calls
                })
                
                # Process each tool call
                for tool_call in msg.tool_calls:
                    try:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments or "{}")
                        result = call_function(func_name, args)
                        
                        # Add tool result
                        st.session_state.conversation_messages.append({
                            "role": "tool",
                            "content": json.dumps(result),
                            "tool_call_id": tool_call.id
                        })
                    except Exception as e:
                        # Add error result
                        st.session_state.conversation_messages.append({
                            "role": "tool", 
                            "content": json.dumps({"error": "Tool temporarily unavailable"}),
                            "tool_call_id": tool_call.id
                        })
                
                # Get final response
                follow_response = _with_retry_and_circuit_breaker(
                    client.chat.completions.create,
                    model=CHAT_MODEL,
                    messages=st.session_state.conversation_messages,
                    max_tokens=1500,
                    temperature=0.1,
                )
                final_content = follow_response.choices[0].message.content or ""
            else:
                final_content = msg.content or ""
            
            # Add assistant response to conversation
            self.add_conversation_message("assistant", final_content)
            
            # Validate response security
            final_content = validate_response_security(final_content)
            
            # Add disclaimer periodically (on 1st, 3rd, 5th, 7th... responses)
            st.session_state.response_count += 1
            if st.session_state.response_count % 2 == 1:  # Show on odd response numbers
                final_content = f"{final_content}\n\n\n<div style='font-size: 0.8em; font-style: italic; margin-top: 1em; padding: 0.5em; border-top: 1px solid #f0f0f0; color: #666;'>{DISCLAIMER_TEXT}</div>"
            
            return final_content
            
        except Exception as e:
            return f"I apologize, but I'm having trouble connecting right now. Please try again in a moment. Error: {str(e)}"
    
    def render_welcome_screen(self):
        """Render the welcome screen"""
        st.markdown("""
        <div class="welcome-container">
            <div class="avatar"></div>
            <h1 class="welcome-title">Hi there! I'm Ben AI your benchmark advisor</h1>
            <p class="welcome-subtitle">
                I'm here to help you find the perfect benchmark that fits your 
                criteria. Just tell me what you're looking for, and let's get started!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input container with integrated send button
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            
            # Create two-column layout for input and button
            input_col, button_col = st.columns([10, 1])
            
            with input_col:
                user_input = st.text_input(
                    "input",
                    placeholder="What kind of benchmark are you looking for today?",
                    key="welcome_input",
                    label_visibility="collapsed"
                )
            
            with button_col:
                send_clicked = st.button("↑", key="welcome_send", help="Send your message")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <p class="input-hint">Press Enter to send, or click the ↑ button</p>
            """, unsafe_allow_html=True)
            
            # Handle input submission
            if (user_input and st.session_state.get("last_welcome_input") != user_input) or send_clicked:
                if user_input:
                    st.session_state.last_welcome_input = user_input
                    self.add_display_message("user", user_input)
                    st.session_state.is_welcome_screen = False
                    st.session_state.is_processing = True
                    st.rerun()
    
    def render_loading_animation(self):
        """Render the loading animation"""
        st.markdown("""
        <div class="loading-container">
            <div class="loading-bubble">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_message(self, message: dict, is_user: bool = False):
        """Render a single message bubble"""
        role = "user" if is_user else "assistant"
        time_str = message.get("timestamp", datetime.now().strftime("%I:%M %p"))
        
        # Process content to fix bullet point formatting
        content = message["content"]
        
        if not is_user:  # Only process assistant messages
            # Convert bullet points to proper HTML list format
            import re
            
            # Check if content has bullet points (• or -)
            if re.search(r'^[•\-]\s', content, re.MULTILINE):
                # Split into lines and process bullets
                lines = content.split('\n')
                processed_lines = []
                in_list = False
                
                for line in lines:
                    line = line.strip()
                    if re.match(r'^[•\-]\s', line):
                        if not in_list:
                            if processed_lines:  # Add closing tag for previous content
                                processed_lines.append('')
                            processed_lines.append('<ul>')
                            in_list = True
                        # Convert bullet to list item
                        bullet_content = re.sub(r'^[•\-]\s', '', line)
                        processed_lines.append(f'<li>{bullet_content}</li>')
                    else:
                        if in_list and line:  # Close list if we have content after bullets
                            processed_lines.append('</ul>')
                            processed_lines.append('')
                            in_list = False
                        if line:  # Only add non-empty lines
                            processed_lines.append(line)
                
                if in_list:  # Close list if still open
                    processed_lines.append('</ul>')
                
                content = '\n'.join(processed_lines)
        
        st.markdown(f"""
        <div class="message {role}">
            <div class="message-bubble {role}">
                {content}
                <div class="message-time {role}">{time_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_chat_interface(self):
        """Render the chat interface"""
        # Add chat header
        st.markdown("""
        <div class="chat-header">
            <h1>Ben AI</h1>
            <div class="subtitle">Your Benchmark Advisor</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                self.render_message(message, is_user=True)
            elif message["role"] == "assistant":
                self.render_message(message, is_user=False)
        
        # Show loading animation if processing
        if st.session_state.is_processing:
            self.render_loading_animation()
            
            # Get the last user message from chat history
            user_messages = [m for m in st.session_state.chat_history if m["role"] == "user"]
            assistant_messages = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
            
            if user_messages and len(assistant_messages) < len(user_messages):
                last_user_input = user_messages[-1]["content"]
                
                # Process the response
                try:
                    response = self.get_chat_response(last_user_input)
                    self.add_display_message("assistant", response)
                    st.session_state.is_processing = False
                    st.rerun()
                except Exception as e:
                    error_msg = f"I apologize, but I'm having trouble connecting right now. Please try again in a moment."
                    self.add_display_message("assistant", error_msg)
                    st.session_state.is_processing = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input for new messages
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            
            # Create two-column layout for input and button
            input_col, button_col = st.columns([10, 1])
            
            with input_col:
                # Use a unique key that changes after each message to force clearing
                input_key = f"chat_input_{len(st.session_state.chat_history)}"
                new_input = st.text_input(
                    "message",
                    placeholder="Type your message...",
                    key=input_key,
                    label_visibility="collapsed",
                    max_chars=500
                )
            
            with button_col:
                send_clicked = st.button("↑", key=f"chat_send_{len(st.session_state.chat_history)}", help="Send your message")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Handle input submission
            if (new_input and st.session_state.get(f"last_chat_input_{len(st.session_state.chat_history)}") != new_input) or send_clicked:
                if new_input:
                    st.session_state[f"last_chat_input_{len(st.session_state.chat_history)}"] = new_input
                    self.add_display_message("user", new_input)
                    st.session_state.is_processing = True
                    st.rerun()
    
    def run(self):
        """Main app runner"""
        if st.session_state.is_welcome_screen:
            self.render_welcome_screen()
        else:
            self.render_chat_interface()

def main():
    """Main function to run the Streamlit app"""
    chat_app = StreamlitChatWrapper()
    chat_app.run()

if __name__ == "__main__":
    main()