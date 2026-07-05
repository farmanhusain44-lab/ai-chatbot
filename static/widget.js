(function() {
    // AI Chatbot Widget Embed Script
    // Usage: <script src="https://your-domain.com/widget.js" data-agent="AI Agent" data-welcome="Hello!"></script>

    function initWidget() {
    const scripts = document.getElementsByTagName('script');
    const currentScript = scripts[scripts.length - 1];
    const baseUrl = currentScript.src.replace('/widget.js', '');
    const agentName = currentScript.getAttribute('data-agent') || 'AI Agent';
    const welcomeText = currentScript.getAttribute('data-welcome') || 'Hello! How can I help you today?';
    const position = currentScript.getAttribute('data-position') || 'bottom-right';
    const color = currentScript.getAttribute('data-color') || '#667eea';

    const widgetId = 'ai-chatbot-widget-' + Date.now();

    // Styles for the floating button and iframe container
    const styles = document.createElement('style');
    styles.textContent = `
        #${widgetId}-container {
            position: fixed;
            ${position.includes('right') ? 'right: 20px;' : 'left: 20px;'}
            ${position.includes('bottom') ? 'bottom: 20px;' : 'top: 20px;'}
            z-index: 999999;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        #${widgetId}-button {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: ${color};
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s, box-shadow 0.2s;
            position: relative;
        }
        #${widgetId}-button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 25px rgba(0,0,0,0.4);
        }
        #${widgetId}-panel {
            position: fixed;
            ${position.includes('right') ? 'right: 20px;' : 'left: 20px;'}
            ${position.includes('bottom') ? 'bottom: 90px;' : 'top: 90px;'}
            width: 360px;
            height: 500px;
            max-width: calc(100vw - 40px);
            max-height: calc(100vh - 110px);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            border: none;
            display: none;
            background: white;
        }
        @media (max-width: 480px) {
            #${widgetId}-panel {
                width: calc(100vw - 40px);
                height: calc(100vh - 110px);
                ${position.includes('right') ? 'right: 10px;' : 'left: 10px;'}
                ${position.includes('bottom') ? 'bottom: 80px;' : 'top: 80px;'}
            }
        }
        #${widgetId}-panel.open {
            display: block;
        }
        .${widgetId}-badge {
            position: absolute;
            top: -2px;
            right: -2px;
            width: 14px;
            height: 14px;
            background: #2ecc71;
            border-radius: 50%;
            border: 2px solid white;
        }
    `;
    document.head.appendChild(styles);

    // Container
    const container = document.createElement('div');
    container.id = widgetId + '-container';

    // Floating button
    const button = document.createElement('button');
    button.id = widgetId + '-button';
    button.setAttribute('aria-label', 'Open chat');
    button.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <span class="${widgetId}-badge"></span>
    `;

    // Chat panel iframe
    const panel = document.createElement('iframe');
    panel.id = widgetId + '-panel';
    panel.src = `${baseUrl}/widget.html?name=${encodeURIComponent(agentName)}&welcome=${encodeURIComponent(welcomeText)}`;
    panel.title = agentName;
    panel.setAttribute('allow', 'microphone');

    container.appendChild(panel);
    container.appendChild(button);
    document.body.appendChild(container);

    // Toggle panel
    let isOpen = false;
    button.addEventListener('click', function() {
        isOpen = !isOpen;
        if (isOpen) {
            panel.classList.add('open');
            // Focus the input inside iframe
            try {
                panel.contentWindow.focus();
            } catch (e) {}
        } else {
            panel.classList.remove('open');
        }
    });

    // Close when clicking outside
    document.addEventListener('click', function(e) {
        if (isOpen && !container.contains(e.target)) {
            isOpen = false;
            panel.classList.remove('open');
        }
    });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWidget);
    } else {
        initWidget();
    }
})();
