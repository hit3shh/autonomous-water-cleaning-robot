// dashboard.js — Real-time updates for Water Cleaning Robot Dashboard

// ═══════════════════════════════════════════════════
//  CHART SETUP
// ═══════════════════════════════════════════════════
const ctx = document.getElementById("rewardChart").getContext("2d");

const rewardChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Reward",
      data: [],
      borderColor: "#10b981",
      backgroundColor: "rgba(16,185,129,0.08)",
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 5,
      pointBackgroundColor: "#10b981",
      fill: true,
      tension: 0.4,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#1a2235",
        borderColor: "#2a3a5c",
        borderWidth: 1,
        titleColor: "#94a3b8",
        bodyColor: "#e2e8f0",
      }
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        grid: {
          color: "rgba(255,255,255,0.05)",
        },
        ticks: {
          color: "#64748b",
          font: { family: "JetBrains Mono", size: 10 }
        }
      }
    }
  }
});

// ═══════════════════════════════════════════════════
//  ACTION CONFIG
// ═══════════════════════════════════════════════════
const ACTION_ICONS = {
  "LEFT"     : "fa-arrow-left",
  "RIGHT"    : "fa-arrow-right",
  "FORWARD"  : "fa-arrow-up",
  "BACKWARD" : "fa-arrow-down",
  "COLLECT"  : "fa-hand-sparkles",
  "IDLE"     : "fa-circle-pause",
};

// Colors for detection dots
const CLASS_COLORS = [
  "#3b82f6", "#10b981", "#f59e0b",
  "#ef4444", "#8b5cf6", "#06b6d4",
  "#ec4899", "#84cc16"
];

// ═══════════════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════════════
function updateClock() {
  const now = new Date();
  document.getElementById("nav-time").textContent =
    now.toLocaleTimeString("en-US", {
      hour:   "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════════════════════════════════
//  CONTROL BUTTONS
// ═══════════════════════════════════════════════════
function startSystem() {
  fetch("/start")
    .then(r => r.json())
    .then(data => {
      console.log("Started:", data);
      // Hide camera overlay
      document.getElementById("cam-overlay").classList.add("hidden");
      // Update badge
      const badge = document.getElementById("cam-badge");
      badge.textContent = "LIVE";
      badge.classList.add("green-badge");
    });
}

function stopSystem() {
  fetch("/stop")
    .then(r => r.json())
    .then(data => {
      console.log("Stopped:", data);
      // Show camera overlay
      document.getElementById("cam-overlay").classList.remove("hidden");
      document.getElementById("cam-overlay").innerHTML =
        '<i class="fa-solid fa-video-slash"></i><p>System stopped</p>';
      // Update badge
      const badge = document.getElementById("cam-badge");
      badge.textContent = "STOPPED";
      badge.classList.remove("green-badge");
    });
}

function resetSystem() {
  fetch("/reset")
    .then(r => r.json())
    .then(data => {
      console.log("Reset:", data);
      // Clear chart
      rewardChart.data.labels = [];
      rewardChart.data.datasets[0].data = [];
      rewardChart.update();
      // Clear history table
      document.getElementById("history-body").innerHTML =
        '<tr><td colspan="5" class="no-data">History cleared</td></tr>';
      // Clear class list
      document.getElementById("class-list").innerHTML =
        '<div class="no-detect"><i class="fa-solid fa-circle-info"></i> No detections yet</div>';
    });
}

// ═══════════════════════════════════════════════════
//  UPDATE STATUS — polls /status every 500ms
// ═══════════════════════════════════════════════════
function updateStatus() {
  fetch("/status")
    .then(r => r.json())
    .then(data => {

      // ── Stat cards ──────────────────────────────
      document.getElementById("stat-steps").textContent =
        data.step.toLocaleString();
      document.getElementById("stat-trash").textContent =
        data.trash_count;
      document.getElementById("stat-reward").textContent =
        data.total_reward.toFixed(1);
      document.getElementById("stat-detected").textContent =
        data.detected_now;
      document.getElementById("fps-val").textContent =
        data.fps;

      // ── System status badge ──────────────────────
      const dot  = document.getElementById("status-dot");
      const text = document.getElementById("status-text");
      if (data.running) {
        dot.classList.add("active");
        text.textContent = "ONLINE";
        text.style.color = "#10b981";
      } else {
        dot.classList.remove("active");
        text.textContent = "OFFLINE";
        text.style.color = "#64748b";
      }

      // ── Action panel ─────────────────────────────
      const action     = data.action || "IDLE";
      const arrowEl    = document.getElementById("action-arrow");
      const iconEl     = document.getElementById("action-icon");
      const nameEl     = document.getElementById("action-name");

      // Remove old action classes
      arrowEl.className = "action-arrow " + action;

      // Set icon
      const iconClass = ACTION_ICONS[action] || "fa-circle-pause";
      iconEl.className = `fa-solid ${iconClass}`;

      // Set name with color
      nameEl.textContent = action;

      // Detail rows
      document.getElementById("detail-pos").textContent =
        data.state_label || "—";

      const rewardEl = document.getElementById("detail-reward");
      rewardEl.textContent = data.reward;
      rewardEl.className   = "detail-val " +
        (data.reward > 0 ? "reward-pos" : data.reward < 0 ? "reward-neg" : "");

      document.getElementById("detail-step").textContent =
        data.step.toLocaleString();

      // ── Detection list ───────────────────────────
      const detectList = document.getElementById("detect-list");
      if (data.detections && data.detections.length > 0) {
        detectList.innerHTML = data.detections.map((d, i) => {
          const color   = CLASS_COLORS[i % CLASS_COLORS.length];
          const confPct = Math.round(d.confidence * 100);
          return `
            <div class="detect-item">
              <div class="detect-dot" style="background:${color}"></div>
              <div class="detect-name">${d.class_name}</div>
              <div class="conf-bar-wrap">
                <div class="conf-bar-fill" style="width:${confPct}%"></div>
              </div>
              <div class="detect-conf">${confPct}%</div>
            </div>`;
        }).join("");
      } else {
        detectList.innerHTML =
          '<div class="no-detect">' +
          '<i class="fa-solid fa-circle-info"></i>' +
          ' No objects detected</div>';
      }

      // ── Class counts ─────────────────────────────
      const classList = document.getElementById("class-list");
      const counts    = data.class_counts || {};
      const active    = Object.entries(counts)
        .filter(([k, v]) => v > 0)
        .sort((a, b) => b[1] - a[1]);

      if (active.length > 0) {
        classList.innerHTML = active.map(([name, cnt], i) => {
          const color = CLASS_COLORS[i % CLASS_COLORS.length];
          return `
            <div class="class-row">
              <div class="class-name">
                <div class="class-dot" style="background:${color}"></div>
                ${name}
              </div>
              <div class="class-count">${cnt}</div>
            </div>`;
        }).join("");
      } else {
        classList.innerHTML =
          '<div class="no-detect">' +
          '<i class="fa-solid fa-circle-info"></i>' +
          ' No detections yet</div>';
      }

      // ── Reward trend chart ───────────────────────
      if (data.reward_trend && data.reward_trend.length > 0) {
        rewardChart.data.labels =
          data.reward_trend.map((_, i) => i + 1);
        rewardChart.data.datasets[0].data = data.reward_trend;

        // Color points based on value
        rewardChart.data.datasets[0].pointBackgroundColor =
          data.reward_trend.map(v =>
            v > 0 ? "#10b981" : v < 0 ? "#ef4444" : "#64748b"
          );
        rewardChart.update("none");
      }

    })
    .catch(err => console.warn("Status fetch error:", err));
}

// ═══════════════════════════════════════════════════
//  UPDATE HISTORY — polls /history every 1s
// ═══════════════════════════════════════════════════
let lastHistoryLen = 0;

function updateHistory() {
  fetch("/history")
    .then(r => r.json())
    .then(data => {
      const tbody = document.getElementById("history-body");

      if (!data || data.length === 0) {
        tbody.innerHTML =
          '<tr><td colspan="5" class="no-data">' +
          'No history yet — press START</td></tr>';
        lastHistoryLen = 0;
        return;
      }

      // Only redraw if data changed
      if (data.length === lastHistoryLen &&
          data[0]?.time === tbody.querySelector("td")?.textContent) {
        return;
      }

      lastHistoryLen = data.length;
      const isNew    = data.length > lastHistoryLen;

      tbody.innerHTML = data.map((entry, i) => {
        const rewardClass = entry.reward > 0
          ? "reward-pos"
          : entry.reward < 0
          ? "reward-neg" : "";

        const rewardSign  = entry.reward > 0 ? "+" : "";
        const newClass    = i === 0 ? "new-row" : "";

        return `
          <tr class="${newClass}">
            <td>${entry.time}</td>
            <td>${entry.object}</td>
            <td>${entry.confidence}</td>
            <td>
              <span class="action-pill pill-${entry.action}">
                ${entry.action}
              </span>
            </td>
            <td class="${rewardClass}">
              ${rewardSign}${entry.reward}
            </td>
          </tr>`;
      }).join("");

    })
    .catch(err => console.warn("History fetch error:", err));
}

// ═══════════════════════════════════════════════════
//  START POLLING
// ═══════════════════════════════════════════════════
// Poll status every 500ms
setInterval(updateStatus, 500);

// Poll history every 1 second
setInterval(updateHistory, 1000);

// Run once immediately
updateStatus();
updateHistory();

console.log("Dashboard JS loaded ✅");
