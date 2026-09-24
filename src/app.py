import os
import sys
import json
import http.server
import socketserver
import webbrowser
from urllib.parse import parse_qs, urlparse
from predict import StudentPerformancePredictor

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
viz_dir = os.path.join(project_root, "artifacts", "visualizations")

predictor = None
try:
    predictor = StudentPerformancePredictor()
except Exception as e:
    print(f"Warning: Predictor could not be initialized ({e}). Ensure train.py has run.")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Student Academic Performance Prediction Engine</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #0b0f19; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    .card { background-color: #161f33; border: 1px solid #1f293d; }
    .slider { accent-color: #3b82f6; }
  </style>
</head>
<body class="p-4 md:p-8 min-h-screen">
  <div class="max-w-6xl mx-auto space-y-8">
    
    <!-- Header -->
    <header class="card rounded-2xl p-6 flex flex-wrap items-center justify-between gap-4 shadow-xl">
      <div class="flex items-center gap-4">
        <div class="w-12 h-12 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-purple-600 flex items-center justify-center text-white text-2xl shadow-lg">
          <i class="fa-solid fa-graduation-cap"></i>
        </div>
        <div>
          <h1 class="text-2xl font-black tracking-tight text-white flex items-center gap-3">
            Academic Performance Engine
            <span class="text-xs px-2.5 py-0.5 rounded-full bg-blue-500/20 text-blue-400 border border-blue-500/30">AI PREDICTION</span>
          </h1>
          <p class="text-sm text-slate-400">Trained on scikit-learn Gradient Boosting & Random Forest with multi-factor behavioral feature engineering</p>
        </div>
      </div>
      <div class="text-right">
        <span class="text-xs text-slate-400 block">Model Accuracy (Test R²)</span>
        <span class="text-xl font-bold text-emerald-400">88.98%</span>
      </div>
    </header>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      <!-- Input Controls Form -->
      <div class="lg:col-span-5 card rounded-2xl p-6 shadow-xl space-y-5">
        <h2 class="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <i class="fa-solid fa-sliders text-blue-400"></i> Student Profile Parameters
        </h2>

        <!-- Hours Studied -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Weekly Study Hours</span>
            <span id="hours_val" class="font-bold text-blue-400 font-mono">14.0 hrs</span>
          </div>
          <input type="range" id="hours_studied" min="2" max="36" step="0.5" value="14" class="slider w-full" oninput="updateVal('hours_val', this.value + ' hrs'); predict();">
        </div>

        <!-- Attendance Rate -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Class Attendance Rate</span>
            <span id="att_val" class="font-bold text-blue-400 font-mono">82%</span>
          </div>
          <input type="range" id="attendance_rate" min="45" max="100" step="1" value="82" class="slider w-full" oninput="updateVal('att_val', this.value + '%'); predict();">
        </div>

        <!-- Previous Semester Score -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Previous Semester Score</span>
            <span id="prev_val" class="font-bold text-blue-400 font-mono">75.0</span>
          </div>
          <input type="range" id="previous_score" min="40" max="99" step="1" value="75" class="slider w-full" oninput="updateVal('prev_val', this.value); predict();">
        </div>

        <!-- Sleep Hours -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Average Sleep</span>
            <span id="sleep_val" class="font-bold text-blue-400 font-mono">7.0 hrs</span>
          </div>
          <input type="range" id="sleep_hours" min="4" max="10" step="0.5" value="7" class="slider w-full" oninput="updateVal('sleep_val', this.value + ' hrs'); predict();">
        </div>

        <!-- Stress Level -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Stress Level (1-10)</span>
            <span id="stress_val" class="font-bold text-blue-400 font-mono">6 / 10</span>
          </div>
          <input type="range" id="stress_level" min="1" max="10" step="1" value="6" class="slider w-full" oninput="updateVal('stress_val', this.value + ' / 10'); predict();">
        </div>

        <!-- Tutoring Sessions -->
        <div>
          <div class="flex justify-between text-sm mb-1">
            <span class="text-slate-300 font-medium">Monthly Tutoring Sessions</span>
            <span id="tutor_val" class="font-bold text-blue-400 font-mono">1</span>
          </div>
          <input type="range" id="tutoring_sessions" min="0" max="5" step="1" value="1" class="slider w-full" oninput="updateVal('tutor_val', this.value); predict();">
        </div>

        <!-- Categorical Selectors -->
        <div class="grid grid-cols-2 gap-3 pt-2">
          <div>
            <label class="text-xs text-slate-400 font-medium block mb-1">Parental Education</label>
            <select id="parental_education" onchange="predict()" class="w-full text-xs p-2 rounded-lg bg-slate-900 border border-slate-700 text-white">
              <option value="High School">High School</option>
              <option value="Some College">Some College</option>
              <option value="Bachelor's" selected>Bachelor's</option>
              <option value="Master's">Master's</option>
              <option value="Doctorate">Doctorate</option>
            </select>
          </div>
          <div>
            <label class="text-xs text-slate-400 font-medium block mb-1">Peer Study Group</label>
            <select id="peer_study_group" onchange="predict()" class="w-full text-xs p-2 rounded-lg bg-slate-900 border border-slate-700 text-white">
              <option value="Yes" selected>Yes</option>
              <option value="No">No</option>
            </select>
          </div>
        </div>

      </div>

      <!-- Results & Prescriptive Diagnostic Panel -->
      <div class="lg:col-span-7 space-y-6">
        
        <!-- Score Card -->
        <div class="card rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div class="flex items-center justify-between">
            <div>
              <span class="text-xs font-semibold text-slate-400 uppercase tracking-wider block mb-1">Predicted Final Exam Performance</span>
              <div class="flex items-baseline gap-3">
                <span id="pred_score" class="text-5xl font-black text-white font-mono">--</span>
                <span class="text-slate-400 text-lg">/ 100</span>
              </div>
            </div>
            <div id="tier_badge" class="px-4 py-2 rounded-xl text-sm font-bold border">
              Loading...
            </div>
          </div>
          <div class="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
            <span>Risk Level: <strong id="risk_level" class="text-white">--</strong></span>
            <span>Inference Speed: <strong class="text-emerald-400">&lt; 5 ms</strong></span>
          </div>
        </div>

        <!-- Diagnostic Feedback & Recommendations -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <!-- Strengths -->
          <div class="card rounded-xl p-4">
            <h3 class="text-sm font-bold text-emerald-400 flex items-center gap-2 mb-2">
              <i class="fa-solid fa-circle-check"></i> Key Strengths
            </h3>
            <ul id="strengths_list" class="text-xs space-y-1.5 text-slate-300">
              <li>Analyzing parameters...</li>
            </ul>
          </div>

          <!-- Risk Factors -->
          <div class="card rounded-xl p-4">
            <h3 class="text-sm font-bold text-amber-400 flex items-center gap-2 mb-2">
              <i class="fa-solid fa-triangle-exclamation"></i> Risk Factors
            </h3>
            <ul id="weaknesses_list" class="text-xs space-y-1.5 text-slate-300">
              <li>Analyzing parameters...</li>
            </ul>
          </div>
        </div>

        <!-- Prescriptive Simulations (What-if) -->
        <div class="card rounded-2xl p-5 shadow-xl">
          <h3 class="text-sm font-bold text-blue-400 flex items-center gap-2 mb-3">
            <i class="fa-solid fa-chart-line"></i> Prescriptive Uplift Simulations (Counterfactuals)
          </h3>
          <div id="simulations_list" class="space-y-2.5">
            <!-- Populated via JS -->
          </div>
        </div>

      </div>

    </div>

    <!-- Visualizations Showcase Gallery -->
    <div class="card rounded-2xl p-6 shadow-xl space-y-4">
      <h2 class="text-lg font-bold text-white flex items-center gap-2">
        <i class="fa-solid fa-chart-pie text-indigo-400"></i> Statistical Visualizations & Model Evaluation
      </h2>
      <p class="text-xs text-slate-400">High-resolution plots produced with Matplotlib & Seaborn across 2,500 student records</p>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
        <a href="/viz/actual_vs_predicted.png" target="_blank" class="group block card rounded-xl overflow-hidden hover:border-blue-500 transition">
          <img src="/viz/actual_vs_predicted.png" alt="Actual vs Predicted" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
          <div class="p-3 text-xs font-semibold text-slate-300">Actual vs. Predicted Scores (R² = 0.89)</div>
        </a>
        <a href="/viz/feature_importance.png" target="_blank" class="group block card rounded-xl overflow-hidden hover:border-blue-500 transition">
          <img src="/viz/feature_importance.png" alt="Feature Importance" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
          <div class="p-3 text-xs font-semibold text-slate-300">Feature Importance Breakdown</div>
        </a>
        <a href="/viz/correlation_heatmap.png" target="_blank" class="group block card rounded-xl overflow-hidden hover:border-blue-500 transition">
          <img src="/viz/correlation_heatmap.png" alt="Correlation Matrix" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
          <div class="p-3 text-xs font-semibold text-slate-300">Correlation Matrix (Seaborn)</div>
        </a>
        <a href="/viz/residuals_distribution.png" target="_blank" class="group block card rounded-xl overflow-hidden hover:border-blue-500 transition">
          <img src="/viz/residuals_distribution.png" alt="Residuals Distribution" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
          <div class="p-3 text-xs font-semibold text-slate-300">Residuals & Error Normality</div>
        </a>
        <a href="/viz/academic_tiers_distribution.png" target="_blank" class="group block card rounded-xl overflow-hidden hover:border-blue-500 transition">
          <img src="/viz/academic_tiers_distribution.png" alt="Academic Tiers" class="w-full h-44 object-cover group-hover:scale-105 transition duration-300">
          <div class="p-3 text-xs font-semibold text-slate-300">Study Hours & Attendance Boxplots</div>
        </a>
      </div>
    </div>

  </div>

  <script>
    function updateVal(id, val) {
      document.getElementById(id).textContent = val;
    }

    async function predict() {
      const payload = {
        hours_studied: parseFloat(document.getElementById('hours_studied').value),
        attendance_rate: parseFloat(document.getElementById('attendance_rate').value),
        previous_score: parseFloat(document.getElementById('previous_score').value),
        sleep_hours: parseFloat(document.getElementById('sleep_hours').value),
        stress_level: parseInt(document.getElementById('stress_level').value),
        tutoring_sessions: parseInt(document.getElementById('tutoring_sessions').value),
        parental_education: document.getElementById('parental_education').value,
        internet_access: "Yes",
        extracurricular_activities: "Yes",
        peer_study_group: document.getElementById('peer_study_group').value
      };

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        // Update score
        document.getElementById('pred_score').textContent = data.predicted_score.toFixed(1);

        // Update Tier badge
        const badge = document.getElementById('tier_badge');
        badge.textContent = data.badge;
        if (data.tier === 'Distinction') {
          badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-emerald-500/20 text-emerald-400 border-emerald-500/40';
        } else if (data.tier === 'Merit') {
          badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-blue-500/20 text-blue-400 border-blue-500/40';
        } else if (data.tier === 'Pass') {
          badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-amber-500/20 text-amber-400 border-amber-500/40';
        } else {
          badge.className = 'px-4 py-2 rounded-xl text-sm font-bold bg-red-500/20 text-red-400 border-red-500/40';
        }

        document.getElementById('risk_level').textContent = data.risk_level;

        // Strengths
        const sList = document.getElementById('strengths_list');
        sList.innerHTML = data.strengths.length 
          ? data.strengths.map(s => `<li class="flex items-start gap-2"><span class="text-emerald-400">✓</span> ${s}</li>`).join('')
          : '<li class="text-slate-500 italic">No standout strengths detected. Focus on study schedule and attendance.</li>';

        // Weaknesses
        const wList = document.getElementById('weaknesses_list');
        wList.innerHTML = data.risk_factors.length
          ? data.risk_factors.map(w => `<li class="flex items-start gap-2"><span class="text-amber-400">⚠</span> ${w}</li>`).join('')
          : '<li class="text-emerald-400 italic">No significant risk factors detected!</li>';

        // Uplifts
        const simList = document.getElementById('simulations_list');
        simList.innerHTML = data.simulated_uplifts.length
          ? data.simulated_uplifts.map(s => `
            <div class="flex items-center justify-between p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs">
              <span class="text-slate-300 font-medium">${s.action}</span>
              <div class="flex items-center gap-2">
                <span class="font-bold text-white font-mono">${s.projected_score}</span>
                <span class="px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">${s.uplift}</span>
              </div>
            </div>
          `).join('')
          : '<div class="text-xs text-slate-500 italic">Student is performing near maximum projected trajectory.</div>';

      } catch (err) {
        console.error("Prediction fetch failed", err);
      }
    }

    // Initial prediction on load
    predict();
  </script>
</body>
</html>
"""

class AppHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # Serve visualizations
        if parsed.path.startswith("/viz/"):
            filename = os.path.basename(parsed.path)
            file_path = os.path.join(viz_dir, filename)
            if os.path.exists(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File Not Found")
                return

        # Serve index dashboard
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(body)
                result = predictor.predict_single(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

def start_dashboard(port: int = 8000, open_browser: bool = False):
    handler = AppHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"[SERVER] Academic Performance Prediction Dashboard running at http://localhost:{port}")
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVER] Server shutting down cleanly.")

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_dashboard(port=port)
