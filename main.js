class FaceRegistrationSystem {
    constructor() {
        this.apiBaseUrl = 'https://alma-unvirulent-lanita.ngrok-free.dev';
    }

    generateJWT(userId) {
        return btoa(JSON.stringify({ userId, exp: Date.now() + 86400000 }));
    }

    verifyJWT(token) {
        try { return JSON.parse(atob(token)).exp > Date.now(); } catch { return false; }
    }

    showNotification(message, type = 'info') {
        const div = document.createElement('div');
        div.className = 'notification';
        div.style.backgroundColor = type === 'success' ? '#00ff88' : type === 'error' ? '#ff4444' : '#333';
        div.style.color = type === 'success' ? 'black' : 'white';
        div.textContent = message;
        document.body.appendChild(div);
        setTimeout(() => {
            div.style.transform = 'translateX(150%)';
            setTimeout(() => div.remove(), 300);
        }, 3000);
        requestAnimationFrame(() => div.style.transform = 'translateX(0)');
    }
}

const originalFetch = window.fetch;
window.fetch = function (url, options = {}) {
    if (url.includes('ngrok')) {
        if (!options.headers) {
            options.headers = {};
        }
        if (options.headers instanceof Headers) {
            options.headers.append('ngrok-skip-browser-warning', 'true');
        } else {
            options.headers['ngrok-skip-browser-warning'] = 'true';
        }
    }
    return originalFetch(url, options);
};

const faceSystem = new FaceRegistrationSystem();