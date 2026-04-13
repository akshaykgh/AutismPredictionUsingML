import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000/api";

const categoricalOptions = {
  age_group: ["toddler", "child", "adolescent", "adult"],
  sex: ["male", "female"],
  ethnicity: ["asian", "black", "hispanic", "middle eastern", "white", "other"],
  jaundice_history: ["yes", "no"],
  family_asd: ["yes", "no"],
  used_screening_app_before: ["yes", "no"],
  speech_delay: ["yes", "no"],
  anxiety_flag: ["yes", "no"],
  who_completed_test: ["self", "parent", "caregiver", "clinician"]
};

const numericDefaults = {
  age_years: 8,
  family_asd_score: 0.35,
  communication_delay_score: 0.45,
  sensory_sensitivity_score: 0.4,
  social_responsiveness_score: 0.48,
  ...Object.fromEntries(Array.from({ length: 10 }, (_, index) => [`a${index + 1}_score`, 0]))
};

const defaultForm = {
  age_group: "child",
  sex: "male",
  ethnicity: "white",
  jaundice_history: "no",
  family_asd: "no",
  used_screening_app_before: "no",
  speech_delay: "no",
  anxiety_flag: "no",
  who_completed_test: "parent",
  ...numericDefaults
};

function App() {
  const [metadata, setMetadata] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadMetadata() {
      try {
        const response = await fetch(`${API_BASE}/metadata`);
        const payload = await response.json();
        setMetadata(payload);
      } catch (loadError) {
        setError("Unable to load model metadata. Start the Flask API first.");
      } finally {
        setLoading(false);
      }
    }

    loadMetadata();
  }, []);

  function updateField(field, value) {
    setForm((current) => ({
      ...current,
      [field]: value
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(
          Object.fromEntries(
            Object.entries(form).map(([key, value]) => [
              key,
              numericDefaults[key] !== undefined ? Number(value) : value
            ])
          )
        )
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Prediction failed");
      }

      setResult(payload);
      setHistory((current) => [
        {
          timestamp: new Date().toLocaleString(),
          ageGroup: form.age_group,
          probability: payload.probability,
          label: payload.label
        },
        ...current
      ].slice(0, 6));
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return <div className="status-screen">Loading dashboard...</div>;
  }

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Autism Screening Intelligence</p>
          <h1>Interactive ASD risk screening across four age groups.</h1>
          <p className="hero-copy">
            Trainable ML stack with threshold optimization, model comparison, and
            explainability surfaced in a clinician-friendly dashboard.
          </p>
        </div>
        <div className="hero-panel">
          <div className="metric-card emphasis">
            <span>Selected model</span>
            <strong>{metadata?.selected_model || "Unavailable"}</strong>
          </div>
          <div className="metric-grid">
            <MetricCard label="Accuracy" value={metadata?.summary_metrics?.accuracy} />
            <MetricCard label="ROC-AUC" value={metadata?.summary_metrics?.roc_auc} />
            <MetricCard
              label="FN Reduction"
              value={`${metadata?.summary_metrics?.false_negative_reduction_pct ?? 0}%`}
            />
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel form-panel">
          <div className="panel-header">
            <h2>Risk Assessment Form</h2>
            <p>Submit screening inputs for a real-time ASD risk prediction.</p>
          </div>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {Object.entries(categoricalOptions).map(([field, options]) => (
                <label key={field}>
                  <span>{toLabel(field)}</span>
                  <select
                    value={form[field]}
                    onChange={(event) => updateField(field, event.target.value)}
                  >
                    {options.map((option) => (
                      <option key={option} value={option}>
                        {toLabel(option)}
                      </option>
                    ))}
                  </select>
                </label>
              ))}

              {Object.keys(numericDefaults).map((field) => (
                <label key={field}>
                  <span>{toLabel(field)}</span>
                  <input
                    type="number"
                    step={field.includes("_score") && !field.startsWith("a") ? "0.01" : "1"}
                    min={field === "age_years" ? "1" : "0"}
                    max={field.startsWith("a") ? "1" : field.includes("_score") ? "1" : "45"}
                    value={form[field]}
                    onChange={(event) => updateField(field, event.target.value)}
                  />
                </label>
              ))}
            </div>
            <button className="primary-button" type="submit" disabled={submitting}>
              {submitting ? "Scoring..." : "Run Prediction"}
            </button>
          </form>
          {error ? <p className="error-banner">{error}</p> : null}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Prediction Output</h2>
            <p>Threshold-aware classification tuned to reduce missed risk cases.</p>
          </div>
          {result ? (
            <div className="result-stack">
              <div className={`result-badge ${result.prediction ? "high-risk" : "low-risk"}`}>
                {result.label}
              </div>
              <div className="gauge">
                <div
                  className="gauge-fill"
                  style={{ width: `${Math.round(result.probability * 100)}%` }}
                />
              </div>
              <div className="result-grid">
                <MetricCard label="Probability" value={result.probability} />
                <MetricCard label="Threshold" value={result.threshold} />
                <MetricCard label="Class" value={result.prediction} />
              </div>
            </div>
          ) : (
            <div className="empty-state">Prediction results will appear here after submission.</div>
          )}

          <div className="panel-header compact">
            <h2>Age Group Performance</h2>
          </div>
          <div className="age-group-list">
            {(metadata?.age_group_metrics || []).map((group) => (
              <div className="age-group-row" key={group.age_group}>
                <div>
                  <strong>{toLabel(group.age_group)}</strong>
                  <span>{group.samples} samples</span>
                </div>
                <div>
                  <strong>{group.roc_auc}</strong>
                  <span>ROC-AUC</span>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Top Predictors</h2>
            <p>Feature importance from the selected production model.</p>
          </div>
          <div className="importance-chart">
            {(metadata?.feature_importance || []).map((feature) => (
              <div className="importance-row" key={feature.feature}>
                <span>{feature.feature.replace(/^numeric__|^categorical__/, "")}</span>
                <div className="importance-bar">
                  <div
                    className="importance-fill"
                    style={{ width: `${Math.round(feature.importance * 100)}%` }}
                  />
                </div>
                <strong>{feature.importance}</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Recent Screenings</h2>
            <p>Local session history for demo and triage workflows.</p>
          </div>
          {history.length ? (
            <div className="history-list">
              {history.map((item, index) => (
                <div className="history-card" key={`${item.timestamp}-${index}`}>
                  <strong>{item.label}</strong>
                  <span>{toLabel(item.ageGroup)}</span>
                  <span>{item.probability}</span>
                  <time>{item.timestamp}</time>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">No recent screenings yet.</div>
          )}
        </section>
      </main>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value ?? "--"}</strong>
    </div>
  );
}

function toLabel(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default App;
