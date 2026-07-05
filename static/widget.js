(function() {
    // AI Chatbot Widget Embed Script
    // Usage: <script src="https://your-domain.com/widget.js" data-agent="AI Agent" data-welcome="Hello!"></script>

    function initWidget() {
    const scripts = document.getElementsByTagName('script');
    const currentScript = scripts[scripts.length - 1];
    const baseUrl = currentScript.src.replace('/widget.js', '');
    const agentName = currentScript.getAttribute('data-agent') || 'Chat with us';
    const welcomeText = currentScript.getAttribute('data-welcome') || 'Welcome';
    const position = currentScript.getAttribute('data-position') || 'bottom-right';
    const color = currentScript.getAttribute('data-color') || '#25D366';

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
        #${widgetId}-tooltip {
            position: absolute;
            ${position.includes('right') ? 'right: 48px;' : 'left: 48px;'}
            ${position.includes('bottom') ? 'bottom: 8px;' : 'top: 8px;'}
            background: ${color};
            color: white;
            padding: 0;
            border-radius: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            font-size: 13px;
            font-weight: 500;
            white-space: nowrap;
            max-width: 0;
            overflow: hidden;
            opacity: 0;
            height: 44px;
            line-height: 44px;
            transition: max-width 0.7s ease, opacity 0.4s ease, padding 0.4s ease;
            pointer-events: none;
            z-index: 999998;
            border: none;
            display: flex;
            align-items: center;
            transform-origin: ${position.includes('right') ? 'right center' : 'left center'};
        }
        #${widgetId}-tooltip.show {
            opacity: 1;
            max-width: 240px;
            padding: 0 18px;
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

    // Cycling tooltip above button
    const tooltip = document.createElement('div');
    tooltip.id = widgetId + '-tooltip';
    tooltip.textContent = 'Welcome';
    tooltip.style.direction = 'ltr';

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
    container.appendChild(tooltip);
    document.body.appendChild(container);

    // Cycle popup messages: English first, then Arabic - repeat every 15s
    const tooltipMessages = [
        { text: 'Welcome', lang: 'en' },
        { text: 'Chat with us', lang: 'en' },
        { text: 'مرحباً', lang: 'ar' },
        { text: 'تحدث معنا', lang: 'ar' }
    ];
    let tooltipIndex = 0;
    let tooltipInterval = null;

    function startTooltipCycle() {
        tooltip.classList.add('show');
        tooltipIndex = 0;
        tooltip.textContent = tooltipMessages[0].text;
        tooltip.style.direction = tooltipMessages[0].lang === 'ar' ? 'rtl' : 'ltr';
        if (tooltipInterval) clearInterval(tooltipInterval);
        tooltipInterval = setInterval(function() {
            tooltipIndex = (tooltipIndex + 1) % tooltipMessages.length;
            const item = tooltipMessages[tooltipIndex];
            tooltip.classList.remove('show');
            setTimeout(function() {
                tooltip.textContent = item.text;
                tooltip.style.direction = item.lang === 'ar' ? 'rtl' : 'ltr';
                tooltip.classList.add('show');
            }, 400);
        }, 15000);
    }

    function stopTooltipCycle() {
        if (tooltipInterval) {
            clearInterval(tooltipInterval);
            tooltipInterval = null;
        }
        tooltip.classList.remove('show');
    }

    startTooltipCycle();

    // Toggle panel
    let isOpen = false;
    button.addEventListener('click', function() {
        isOpen = !isOpen;
        if (isOpen) {
            panel.classList.add('open');
            stopTooltipCycle();
            // Focus the input inside iframe
            try {
                panel.contentWindow.focus();
            } catch (e) {}
        } else {
            panel.classList.remove('open');
            startTooltipCycle();
        }
    });

    // Close when clicking outside
    document.addEventListener('click', function(e) {
        if (isOpen && !container.contains(e.target)) {
            isOpen = false;
            panel.classList.remove('open');
            startTooltipCycle();
        }
    });

    // Close widget from iframe close button
    window.addEventListener('message', function(event) {
        if (event.data && event.data.action === 'close-widget') {
            isOpen = false;
            panel.classList.remove('open');
            startTooltipCycle();
        }
    });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initWidget);
    } else {
        initWidget();
    }
})();
