import React from "react";
import ReactDOM from "react-dom/client";
import {
  AlertTriangle,
  BarChart3,
  Cloud,
  Database,
  FileSearch,
  Globe2,
  ShieldCheck,
  Upload,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const demoText =
  "Federal health officials released a detailed report today after a multi-agency review. The report cites named researchers, public datasets, and a timeline for independent verification.";

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CloudPanel({ cloudInfo }) {
  return (
    <section className="cloud-panel">
      <div className="dashboard-header">
        <Cloud size={22} />
        <h2>Free AWS-Compatible Layer</h2>
      </div>
      <div className="cloud-grid">
        <Metric label="Provider" value={cloudInfo?.provider || "LocalStack S3"} />
        <Metric label="Bucket" value={cloudInfo?.bucket || "fake-news-model-artifacts"} />
        <Metric label="Status" value={cloudInfo?.enabled ? "Connected" : "Local ready"} />
        <Metric label="Artifacts" value={cloudInfo?.objects?.length || 0} />
      </div>
    </section>
  );
}

function ResultPanel({ result }) {
  if (!result) {
    return (
      <section className="empty-state">
        <FileSearch size={38} />
        <h2>Ready to analyze</h2>
        <p>Paste an article, extract one from a URL, or upload a text file to get a prediction.</p>
      </section>
    );
  }

  const probabilityData = [
    { name: "Fake", value: Math.round(result.fake_probability * 100), color: "#d94b4b" },
    { name: "Real", value: Math.round(result.real_probability * 100), color: "#287c63" },
  ];

  return (
    <section className={`result-panel ${result.label.includes("Fake") ? "fake" : "real"}`}>
      <div className="result-heading">
        <div>
          <span className="eyebrow">Prediction</span>
          <h2>{result.label}</h2>
        </div>
        <div className="confidence">
          <span>{Math.round(result.confidence * 100)}%</span>
          <small>confidence</small>
        </div>
      </div>

      <div className="chart-box">
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={probabilityData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {probabilityData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="metrics-grid">
        <Metric label="Words" value={result.stats.words} />
        <Metric label="Sentences" value={result.stats.sentences} />
        <Metric label="Unique Words" value={result.stats.unique_words} />
        <Metric label="Diversity" value={result.stats.lexical_diversity} />
      </div>

      <div className="explain-box">
        <h3>Signal Terms</h3>
        {result.explanation.length ? (
          <div className="chips">
            {result.explanation.map((item) => (
              <span className="chip" key={item.term}>
                {item.term} <b>{item.weight}</b>
              </span>
            ))}
          </div>
        ) : (
          <p>No unusually strong terms found by the lightweight explainer.</p>
        )}
      </div>
    </section>
  );
}

function App() {
  const [text, setText] = React.useState(demoText);
  const [url, setUrl] = React.useState("");
  const [result, setResult] = React.useState(null);
  const [modelInfo, setModelInfo] = React.useState(null);
  const [cloudInfo, setCloudInfo] = React.useState(null);
  const [history, setHistory] = React.useState([]);
  const [serverHistory, setServerHistory] = React.useState([]);
  const [explanationMethod, setExplanationMethod] = React.useState("linear");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  const refreshHistory = React.useCallback(() => {
    fetch(`${API_URL}/prediction-history?limit=6`)
      .then((response) => response.json())
      .then((data) => setServerHistory(data.results || []))
      .catch(() => setServerHistory([]));
  }, []);

  React.useEffect(() => {
    fetch(`${API_URL}/model-info`)
      .then((response) => response.json())
      .then(setModelInfo)
      .catch(() => setModelInfo({ model: "offline", metrics: {} }));
    fetch(`${API_URL}/cloud/status`)
      .then((response) => response.json())
      .then(setCloudInfo)
      .catch(() => setCloudInfo({ enabled: false, provider: "LocalStack S3" }));
    refreshHistory();
  }, [refreshHistory]);

  async function analyze() {
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, explanation_method: explanationMethod }),
      });

      if (!response.ok) {
        throw new Error("Prediction failed. Check that the FastAPI server is running.");
      }

      const data = await response.json();
      setResult(data);
      setHistory((items) => [{ label: data.label, confidence: data.confidence, text }, ...items].slice(0, 6));
      refreshHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function extractFromUrl() {
    if (!url.trim()) return;
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/extract-url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error("Could not extract article text from that URL.");
      }

      const data = await response.json();
      setText(`${data.title}\n\n${data.text}`.trim());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function uploadFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setText(String(reader.result || ""));
    reader.readAsText(file);
  }

  const metrics = modelInfo?.metrics || {};

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <span className="eyebrow">ML + Cloud Portfolio Project</span>
          <h1>Fake News Intelligence Lab</h1>
          <p>
            A production-style fake news detector with a React interface, FastAPI backend,
            model training pipeline, explainability signals, and cloud-ready packaging.
          </p>
        </div>
        <div className="status-strip">
          <ShieldCheck size={20} />
          <span>{modelInfo ? `Model: ${modelInfo.model}` : "Checking API..."}</span>
        </div>
      </section>

      <section className="feature-strip">
        <div><Database size={18} /><span>PostgreSQL logging</span></div>
        <div><Cloud size={18} /><span>LocalStack S3 artifacts</span></div>
        <div><BarChart3 size={18} /><span>MLflow + DVC workflow</span></div>
      </section>

      <section className="workspace">
        <div className="input-panel">
          <div className="panel-header">
            <h2>Article Input</h2>
            <label className="icon-button" title="Upload text file">
              <Upload size={18} />
              <input type="file" accept=".txt,.csv" onChange={uploadFile} />
            </label>
          </div>

          <div className="url-row">
            <Globe2 size={18} />
            <input
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="Paste article URL"
            />
            <button onClick={extractFromUrl} disabled={loading}>Extract</button>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste a news article here..."
          />

          <div className="segmented-control" aria-label="Explanation method">
            <button
              className={explanationMethod === "linear" ? "active" : ""}
              onClick={() => setExplanationMethod("linear")}
              type="button"
            >
              Linear
            </button>
            <button
              className={explanationMethod === "lime" ? "active" : ""}
              onClick={() => setExplanationMethod("lime")}
              type="button"
            >
              LIME
            </button>
          </div>

          {error && (
            <div className="error">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}

          <button className="primary-button" onClick={analyze} disabled={loading || text.length < 20}>
            {loading ? "Analyzing..." : "Analyze News"}
          </button>
        </div>

        <ResultPanel result={result} />
      </section>

      <section className="dashboard">
        <div className="dashboard-header">
          <BarChart3 size={22} />
          <h2>Model Dashboard</h2>
        </div>
        <div className="metrics-grid">
          <Metric label="Accuracy" value={metrics.accuracy ? metrics.accuracy.toFixed(3) : "Train model"} />
          <Metric label="F1 Score" value={metrics.f1 ? metrics.f1.toFixed(3) : "Train model"} />
          <Metric label="ROC AUC" value={metrics.roc_auc ? metrics.roc_auc.toFixed(3) : "Train model"} />
          <Metric label="Samples" value={metrics.dataset_samples || "Pending"} />
        </div>
      </section>

      <CloudPanel cloudInfo={cloudInfo} />

      <section className="history">
        <h2>Recent Checks</h2>
        {serverHistory.length ? (
          <div className="history-list">
            {serverHistory.map((item) => (
              <article key={item.id}>
                <strong>{item.label}</strong>
                <span>{Math.round(item.confidence * 100)}%</span>
                <p>{item.text_preview.slice(0, 150)}...</p>
              </article>
            ))}
          </div>
        ) : history.length ? (
          <div className="history-list">
            {history.map((item, index) => (
              <article key={`${item.label}-${index}`}>
                <strong>{item.label}</strong>
                <span>{Math.round(item.confidence * 100)}%</span>
                <p>{item.text.slice(0, 150)}...</p>
              </article>
            ))}
          </div>
        ) : (
          <p>No predictions yet.</p>
        )}
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
