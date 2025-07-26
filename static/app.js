document.getElementById('startBtn').addEventListener('click', () => {
    document.getElementById('landing').classList.add('hidden');
    document.getElementById('chat-container').classList.remove('hidden');
});

function appendMessage(text, cls){
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = `message ${cls}`;
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function sendMessage(msg){
    if(!msg.trim()) return;
    appendMessage(msg, 'user');
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage(data.response, 'assistant');
    });
}

document.getElementById('sendBtn').addEventListener('click', () => {
    const input = document.getElementById('userInput');
    sendMessage(input.value);
    input.value = '';
});

document.getElementById('userInput').addEventListener('keypress', (e) => {
    if(e.key === 'Enter'){
        e.preventDefault();
        document.getElementById('sendBtn').click();
    }
});

document.getElementById('applyFilters').addEventListener('click', () => {
    const selected = {};
    document.querySelectorAll('.sidebar input[type=checkbox]:checked').forEach(cb => {
        if(!selected[cb.name]) selected[cb.name] = [];
        selected[cb.name].push(cb.value);
    });
    const msg = 'Show benchmarks matching ' + JSON.stringify(selected);
    sendMessage(msg);
});
