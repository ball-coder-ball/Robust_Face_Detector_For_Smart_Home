class FaceRegistrationSystem {
    constructor() {
        // ⚠️⚠️ อย่าลืมแก้ URL ตรงนี้เป็นของ ngrok ล่าสุดจาก Colab ⚠️⚠️
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

// --- OVERRIDE FETCH เพื่อแก้ปัญหา Ngrok Warning ---
const originalFetch = window.fetch;
window.fetch = function(url, options = {}) {
    // ถ้า URL เป็นของ ngrok ให้เพิ่ม Header พิเศษ
    if (url.includes('ngrok')) {
        if (!options.headers) {
            options.headers = {};
        }
        // วิธีที่ 1: เพิ่ม Header (สำหรับ ngrok บางเวอร์ชัน)
        // options.headers['ngrok-skip-browser-warning'] = 'true'; 
        
        // วิธีที่ 2: หรือถ้าเป็น object Header แบบนี้
        if (options.headers instanceof Headers) {
             options.headers.append('ngrok-skip-browser-warning', 'true');
        } else {
             options.headers['ngrok-skip-browser-warning'] = 'true';
        }
    }
    return originalFetch(url, options);
};

const faceSystem = new FaceRegistrationSystem();