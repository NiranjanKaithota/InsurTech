// ================= GAME STATE =================
const state = {
    speed: 0, // km/h
    throttle: 0,
    brake: 0,
    risk: 'SAFE',
    frame: 0,
    playerX: 0,    // -1 (Left) to 1 (Right)
    obstacles: [], // Array of {x, z, type}
    gameOver: false
};

// ================= CONTROLS =================
const controls = {
    up: false,
    down: false,
    left: false,
    right: false
};

// Initial state send
let lastControlState = JSON.stringify(controls);

document.addEventListener('keydown', (e) => {
    if (e.repeat) return;
    if (e.key === 'ArrowUp' || e.key === 'w') controls.up = true;
    if (e.key === 'ArrowDown' || e.key === 's') controls.down = true;
    if (e.key === 'ArrowLeft' || e.key === 'a') controls.left = true;
    if (e.key === 'ArrowRight' || e.key === 'd') controls.right = true;
    sendControls();
});

document.addEventListener('keyup', (e) => {
    if (e.key === 'ArrowUp' || e.key === 'w') controls.up = false;
    if (e.key === 'ArrowDown' || e.key === 's') controls.down = false;
    if (e.key === 'ArrowLeft' || e.key === 'a') controls.left = false;
    if (e.key === 'ArrowRight' || e.key === 'd') controls.right = false;
    sendControls();
});

async function sendControls() {
    // Only send Up/Down to backend (physics)
    // Left/Right is handled locally for lane changes
    const backendControls = { up: controls.up, down: controls.down };

    const currentControlState = JSON.stringify(backendControls);
    if (currentControlState === lastControlState) return;
    lastControlState = currentControlState;

    try {
        await fetch("http://localhost:5001/api/control", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: currentControlState
        });
    } catch (e) {
        // console.error("Control error:", e);
    }
}

// ================= DATA FETCHING =================
async function fetchLive() {
    if (state.gameOver) return;

    try {
        const res = await fetch("http://localhost:5001/api/live");
        if (!res.ok) throw new Error("Backend offline");
        const data = await res.json();

        // Update Local State
        state.speed = data.speed;
        state.throttle = data.throttle;
        state.brake = data.brake;
        state.risk = data.risk;

        // Update DOM
        updateText('speed', Math.round(data.speed));
        updateText('limit', data.speed_limit);
        updateText('throttle-val', Math.round(data.throttle) + "%");
        updateText('brake-val', Math.round(data.brake) + "%");

        updateGauge(data.speed);

        document.getElementById('throttle').style.width = data.throttle + "%";
        document.getElementById('brake').style.width = data.brake + "%";

        const riskEl = document.getElementById("risk");
        riskEl.innerText = data.risk;
        riskEl.className = `status-indicator ${data.risk.toLowerCase()}`;

        updateConnectionStatus(true);
    } catch (e) {
        console.error("Fetch error:", e);
        updateConnectionStatus(false);
    }
}

function updateText(id, value) {
    const el = document.getElementById(id);
    if (el && el.innerText !== String(value)) {
        el.innerText = value;
    }
}

function updateConnectionStatus(connected) {
    const el = document.querySelector('.connection-status');
    if (el) {
        el.innerText = connected ? "CONNECTED" : "OFFLINE";
        el.style.color = connected ? "var(--neon-green)" : "var(--neon-red)";
        el.style.textShadow = connected ? "0 0 5px var(--neon-green)" : "0 0 5px var(--neon-red)";
    }
}

function updateGauge(speed) {
    const maxSpeed = 160;
    const radius = 90;
    const circumference = 2 * Math.PI * radius;
    const progress = document.querySelector('.gauge-progress');

    const speedRatio = Math.min(speed, maxSpeed) / maxSpeed;
    const offset = circumference - (speedRatio * circumference);

    if (progress) progress.style.strokeDashoffset = offset;
}

// ================= GAME LOGIC =================
function updateGameLogic() {
    if (state.gameOver) return;

    // 1. Move Player
    const steerSpeed = 0.05;
    if (controls.left) state.playerX = Math.max(-1.5, state.playerX - steerSpeed);
    if (controls.right) state.playerX = Math.min(1.5, state.playerX + steerSpeed);

    // 2. Spawn Obstacles
    // Spawn chance increases with speed
    const spawnChance = 0.01 + (state.speed * 0.0005);
    if (Math.random() < spawnChance) {
        state.obstacles.push({
            x: (Math.random() * 3) - 1.5, // Random lane
            z: 1000, // Start far away
            color: Math.random() > 0.5 ? '#ff003c' : '#ffaa00'
        });
    }

    // 3. Move Obstacles
    // Move speed depends on player speed relative to "traffic speed"
    const moveSpeed = 5 + (state.speed * 0.5);

    state.obstacles.forEach(obs => {
        obs.z -= moveSpeed;
    });

    // 4. Remove passed obstacles
    state.obstacles = state.obstacles.filter(obs => obs.z > -100);

    // 5. Collision Detection
    // Simple Box Collision
    const playerZ = 0;
    const hitBoxWidth = 0.5;
    const hitBoxDepth = 50;

    state.obstacles.forEach(obs => {
        if (Math.abs(obs.z - playerZ) < hitBoxDepth && Math.abs(obs.x - state.playerX) < hitBoxWidth) {
            triggerGameOver();
        }
    });
}

function triggerGameOver() {
    state.gameOver = true;
    document.getElementById('risk').innerText = "CRASHED!";
    document.getElementById('risk').className = "status-indicator dangerous";
    document.querySelector('header h1').innerText = "GAME OVER";
}

// ================= VISUALS (CANVAS) =================
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');

let width, height;

function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function project(x, y, z) {
    const scale = 300 / (z + 300); // Perspective scale
    const x2d = (x * width) / 2 * scale + width / 2;
    const y2d = (y * height) / 2 * scale + height / 2;
    return { x: x2d, y: y2d, scale: scale };
}

function drawGame() {
    // Clear
    ctx.fillStyle = 'rgba(0,0,0,1)';
    if (state.gameOver) ctx.fillStyle = 'rgba(50,0,0,1)';
    ctx.fillRect(0, 0, width, height);

    // Grid Perspective
    const horizon = height * 0.55;

    // Draw Sky
    const skyGrad = ctx.createLinearGradient(0, 0, 0, horizon);
    skyGrad.addColorStop(0, "#050510");
    skyGrad.addColorStop(1, "#1a1a2e");
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, width, horizon);

    // Draw Sun
    ctx.beginPath();
    ctx.arc(width / 2, horizon - 50, 60, 0, Math.PI * 2);
    const sunGrad = ctx.createLinearGradient(width / 2, horizon - 110, width / 2, horizon + 10);
    sunGrad.addColorStop(0, "#ffff00");
    sunGrad.addColorStop(1, "#ff003c");
    ctx.fillStyle = sunGrad;
    ctx.fill();

    // Draw Ground/Grid
    ctx.lineWidth = 2;
    ctx.strokeStyle = `rgba(0, 243, 255, 0.4)`; // Neon Cyan
    if (state.risk === 'RISKY') ctx.strokeStyle = `rgba(255, 170, 0, 0.6)`;
    if (state.risk === 'DANGEROUS') ctx.strokeStyle = `rgba(255, 0, 60, 0.8)`;

    // Move grid lines based on speed
    if (!state.gameOver) state.frame += state.speed * 0.1;
    const offset = state.frame % 50;

    ctx.beginPath();

    // Road boundaries
    // We project 3D points to 2D
    // Z goes into screen

    // Vertical lines
    for (let i = -10; i <= 10; i++) {
        // Perspective math manual for grid lines
        // A bit simplified for visual effect
        const x_bottom = (width / 2) + (i * 100);
        const x_top = (width / 2); // Vanishing point

        ctx.moveTo(x_bottom, height);
        ctx.lineTo(x_top, horizon);
    }

    // Horizontal moving lines
    for (let i = 0; i < 20; i++) {
        // Use a power function for Z depth spacing
        const z = i * 50 - offset;
        const y = height - Math.pow(i, 1.5) * 5 - (offset / 10);

        if (y > horizon) {
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
        }
    }
    ctx.stroke();

    // --- 3D Projection for Objects ---
    // Helper to draw sprites

    // Draw Obstacles (sort by Z, furthest first)
    state.obstacles.sort((a, b) => b.z - a.z).forEach(obs => {
        const p = project(obs.x, 0.5, obs.z); // y=0.5 (ground level approx)
        if (p.scale > 0) {
            const size = 150 * p.scale;
            ctx.fillStyle = obs.color;
            ctx.shadowBlur = 20;
            ctx.shadowColor = obs.color;
            ctx.fillRect(p.x - size / 2, p.y + (height * 0.2), size, size * 0.6);
            ctx.shadowBlur = 0;

            // Tail lights
            ctx.fillStyle = '#ffaaaa';
            ctx.fillRect(p.x - size / 3, p.y + (height * 0.2) + size * 0.2, size / 4, size / 5);
            ctx.fillRect(p.x + size / 10, p.y + (height * 0.2) + size * 0.2, size / 4, size / 5);
        }
    });

    // Draw Player Car
    // Position is fixed in Z (close to camera), X depends on controls
    const playerSize = 120;
    const px = width / 2 + (state.playerX * width * 0.15); // Map -1..1 to screen width
    const py = height - 100;

    ctx.fillStyle = '#0aff0a'; // Neon Green
    ctx.shadowBlur = 30;
    ctx.shadowColor = '#0aff0a';

    // Retro Car Shape
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px - playerSize / 2, py + playerSize / 2); // Left bottom
    ctx.lineTo(px + playerSize / 2, py + playerSize / 2); // Right bottom
    ctx.closePath();
    ctx.fill();

    ctx.shadowBlur = 0;

    updateGameLogic();
    requestAnimationFrame(drawGame);
}

// Start loops
setInterval(fetchLive, 100); // Poll backend
drawGame(); // Start animation loop
