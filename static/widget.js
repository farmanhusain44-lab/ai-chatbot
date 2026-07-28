(function() {
    // AI Chatbot Widget Embed Script
    // Usage: <script src="https://your-domain.com/widget.js" data-agent="AI Agent" data-welcome="Hello!"></script>

    // Guard: if someone accidentally embeds this script twice (e.g., basic + customized), only render one widget
    // Uses a global flag on window so a second load is a no-op instead of stacking two chat bubbles on the page.
    if (window.__botifyaiWidgetLoaded) {
        try { console.warn('[BotifyAI] Widget script loaded more than once — ignoring the second copy. Keep only one <script src=".../widget.js"> tag on your page.'); } catch(e) {}
        return;
    }
    window.__botifyaiWidgetLoaded = true;

    function initWidget() {
    const scripts = document.getElementsByTagName('script');
    const currentScript = scripts[scripts.length - 1];
    const baseUrl = currentScript.src.replace('/widget.js', '');
    const agentName = currentScript.getAttribute('data-agent') || 'Chat with us';
    const welcomeText = currentScript.getAttribute('data-welcome') || 'Welcome';
    const accessCode = currentScript.getAttribute('data-access') || '';
    const position = currentScript.getAttribute('data-position') || 'bottom-right';
    const color = currentScript.getAttribute('data-color') || '#25D366';
    const lang = currentScript.getAttribute('data-lang') || (navigator.language || 'en').split('-')[0].toLowerCase();
    const region = currentScript.getAttribute('data-region') || '';
    // Client-facing customization
    const iconStyle = (currentScript.getAttribute('data-icon-style') || 'logo').toLowerCase(); // logo|chat|robot|phone|sparkle|headset|custom
    const iconUrl = currentScript.getAttribute('data-icon') || '';                              // custom image URL, wins over iconStyle
    const shape = (currentScript.getAttribute('data-shape') || 'round').toLowerCase();          // round|square|pill
    const size = (currentScript.getAttribute('data-size') || 'medium').toLowerCase();           // small|medium|large
    const chatColor = currentScript.getAttribute('data-chat-color') || '';                      // chat window header color (defaults to button color)

    const SIZE_MAP = { small: 48, medium: 60, large: 72 };
    const buttonSize = SIZE_MAP[size] || 60;
    const iconPx = Math.round(buttonSize * 0.7);
    const borderRadius = shape === 'round' ? '50%' : (shape === 'pill' ? Math.round(buttonSize/2) + 'px' : '14px');

    // Preset icons — SVG-based so they render sharp at any size and pick up the color
    const ICON_PRESETS = {
        logo:    `<img src="https://i.imgur.com/FfNwGyA.png" width="${iconPx}" height="${iconPx}" style="border-radius:${shape==='round'?'50%':'8px'};object-fit:cover;" alt="BotifyAI"/>`,
        chat:    `<svg width="${iconPx}" height="${iconPx}" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
        robot:   `<span style="font-size:${Math.round(iconPx*0.85)}px;line-height:1;">🤖</span>`,
        phone:   `<svg width="${iconPx}" height="${iconPx}" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.37 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.33 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/></svg>`,
        sparkle: `<span style="font-size:${Math.round(iconPx*0.85)}px;line-height:1;">✨</span>`,
        headset: `<svg width="${iconPx}" height="${iconPx}" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/></svg>`,
        whatsapp:`<svg width="${iconPx}" height="${iconPx}" viewBox="0 0 24 24" fill="white"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.626.712.226 1.36.194 1.872.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>`,
    };
    let iconMarkup;
    if (iconUrl) {
        iconMarkup = `<img src="${iconUrl}" width="${iconPx}" height="${iconPx}" style="border-radius:${shape==='round'?'50%':'8px'};object-fit:cover;" alt="${agentName}"/>`;
    } else if (iconStyle.length <= 2) {
        // A single emoji character passed as iconStyle — render as text
        iconMarkup = `<span style="font-size:${Math.round(iconPx*0.85)}px;line-height:1;">${iconStyle}</span>`;
    } else {
        iconMarkup = ICON_PRESETS[iconStyle] || ICON_PRESETS.logo;
    }

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
            width: ${buttonSize}px;
            height: ${buttonSize}px;
            border-radius: ${borderRadius};
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
            padding: 0;
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

    const tooltip = document.createElement('div');
    tooltip.id = widgetId + '-tooltip';
    tooltip.style.display = 'none';

    // Floating button
    const button = document.createElement('button');
    button.id = widgetId + '-button';
    button.setAttribute('aria-label', 'Open chat');
    button.innerHTML = iconMarkup + `<span class="${widgetId}-badge"></span>`;

    // Chat panel iframe
    const panel = document.createElement('iframe');
    panel.id = widgetId + '-panel';
    // Pass icon + colors to the iframe so the chat panel header can match the button's look
    const effectiveChatColor = chatColor || color;
    panel.src = `${baseUrl}/widget.html?name=${encodeURIComponent(agentName)}&welcome=${encodeURIComponent(welcomeText)}&access=${encodeURIComponent(accessCode)}&lang=${encodeURIComponent(lang)}&region=${encodeURIComponent(region)}&icon-style=${encodeURIComponent(iconStyle)}&icon=${encodeURIComponent(iconUrl)}&color=${encodeURIComponent(effectiveChatColor)}`;
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

    // Tooltip disabled — clean button only

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
