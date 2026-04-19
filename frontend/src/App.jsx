import { useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:5000/api";

/** Shown in order; XGBoost appears only after a successful train (requires libomp on macOS). */
const CLASSIFIER_MODEL_IDS = ["xgboost", "random_forest", "decision_tree"];

const categoricalOptions = {
  sex: ["male", "female"],
  ethnicity: ["asian", "black", "hispanic", "middle eastern", "white", "other"],
  jaundice_history: ["yes", "no"],
  family_asd: ["yes", "no"],
  used_screening_app_before: ["yes", "no"],
  speech_delay: ["yes", "no"],
  anxiety_flag: ["yes", "no"],
  who_completed_test: ["self", "parent", "caregiver", "clinician"]
};

const fieldLabels = {
  age_years: "Age in years",
  family_asd_score: "Family ASD risk score",
  communication_delay_score: "Communication delay score",
  sensory_sensitivity_score: "Sensory sensitivity score",
  social_responsiveness_score: "Social responsiveness score",
  sex: "Sex",
  ethnicity: "Ethnicity",
  jaundice_history: "Jaundice history",
  family_asd: "Family ASD history",
  used_screening_app_before: "Used screening app before",
  speech_delay: "Speech delay",
  anxiety_flag: "Anxiety flag",
  who_completed_test: "Who completed test"
};

/** Self-report adult: classic AQ-10 Adult (I). */
const aq10AdultItems = [
  {
    field: "a1_score",
    prompt: "I often notice small sounds when others do not.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a2_score",
    prompt: "I usually concentrate more on the whole picture than the small details.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a3_score",
    prompt: "I find it easy to do more than one thing at once.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a4_score",
    prompt: "If there is an interruption, I can switch back to what I was doing very quickly.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a5_score",
    prompt: "I find it easy to read between the lines when someone is talking to me.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a6_score",
    prompt: "I know how to tell if someone listening to me is getting bored.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a7_score",
    prompt: "When I am reading a story, I find it difficult to work out the characters' intentions.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a8_score",
    prompt: "I like to collect information about categories of things.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a9_score",
    prompt: "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a10_score",
    prompt: "I find it difficult to work out people's intentions.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  }
];

/** Parent/caregiver-report adolescent wording (S/he); scoring maps each item to binary features a1–a10. */
const aq10AdolescentItems = [
  {
    field: "a1_score",
    prompt: "S/he notices patterns in things all the time.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a2_score",
    prompt: "S/he usually concentrates more on the whole picture, rather than the small details.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a3_score",
    prompt:
      "In a social group, s/he can easily keep track of several different people's conversations.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a4_score",
    prompt: "If there is an interruption, s/he can switch back to what s/he was doing very quickly.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a5_score",
    prompt: "S/he frequently finds that s/he doesn't know how to keep a conversation going.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a6_score",
    prompt: "S/he is good at social chit-chat.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a7_score",
    prompt:
      "When s/he was younger, s/he used to enjoy playing games involving pretending with other children.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a8_score",
    prompt: "S/he finds it difficult to imagine what it would be like to be someone else.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a9_score",
    prompt: "S/he finds social situations easy.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a10_score",
    prompt: "S/he finds it hard to make new friends.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  }
];

/** Parent-report child wording (AQ-10 Child style); scoring per item aligned to the same binary slots as adult. */
const aq10ChildItems = [
  {
    field: "a1_score",
    prompt: "S/he often notices small sounds when others do not.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a2_score",
    prompt: "S/he usually concentrates more on the whole picture, rather than the small details.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a3_score",
    prompt:
      "In a social group, s/he can easily keep track of several different people's conversations.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a4_score",
    prompt: "S/he finds it easy to go back and forth between different activities.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a5_score",
    prompt: "S/he doesn't know how to keep a conversation going with his/her peers.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a6_score",
    prompt: "S/he is good at social chit-chat.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a7_score",
    prompt:
      "When s/he is read a story, s/he finds it difficult to work out the character's intentions or feelings.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  },
  {
    field: "a8_score",
    prompt:
      "When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a9_score",
    prompt: "S/he finds it easy to work out what someone is thinking or feeling just by looking at their face.",
    scoring: { "definitely agree": 0, "slightly agree": 0, "slightly disagree": 1, "definitely disagree": 1 }
  },
  {
    field: "a10_score",
    prompt: "S/he finds it hard to make new friends.",
    scoring: { "definitely agree": 1, "slightly agree": 1, "slightly disagree": 0, "definitely disagree": 0 }
  }
];

const aq10QuestionSets = {
  child: aq10ChildItems,
  adolescent: aq10AdolescentItems,
  adult: aq10AdultItems
};

const aq10AnswerOptions = [
  "definitely agree",
  "slightly agree",
  "slightly disagree",
  "definitely disagree"
];

const clinicalIndicatorFields = [
  "family_asd_score",
  "communication_delay_score",
  "sensory_sensitivity_score",
  "social_responsiveness_score"
];

const defaultForm = {
  age_group: "adult",
  age_years: 24,
  sex: "male",
  ethnicity: "white",
  jaundice_history: "no",
  family_asd: "no",
  used_screening_app_before: "no",
  speech_delay: "no",
  anxiety_flag: "no",
  who_completed_test: "self",
  family_asd_score: 0.35,
  communication_delay_score: 0.45,
  sensory_sensitivity_score: 0.4,
  social_responsiveness_score: 0.48,
  aq1_response: "",
  aq2_response: "",
  aq3_response: "",
  aq4_response: "",
  aq5_response: "",
  aq6_response: "",
  aq7_response: "",
  aq8_response: "",
  aq9_response: "",
  aq10_response: ""
};

function getActiveModelMetrics(metadata) {
  if (!metadata) return null;
  const name = metadata.active_model ?? metadata.selected_model;
  const row = metadata.model_comparison?.find((m) => m.model_name === name);
  if (row) {
    return {
      accuracy: Number(row.accuracy.toFixed(4)),
      roc_auc: Number(row.roc_auc.toFixed(4)),
      false_negative_reduction_pct: row.false_negative_reduction_pct ?? 0
    };
  }
  return metadata.summary_metrics;
}

function getActiveFeatureImportance(metadata) {
  if (!metadata) return [];
  const name = metadata.active_model ?? metadata.selected_model;
  const row = metadata.model_comparison?.find((m) => m.model_name === name);
  return row?.feature_importance ?? metadata.feature_importance ?? [];
}

function App() {
  const [metadata, setMetadata] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadMetadata() {
      try {
        const response = await fetch(`${API_BASE}/metadata`);
        const payload = await response.json();
        setMetadata(payload);
      } catch {
        setError("Unable to load model metadata. Start the Flask API first.");
      } finally {
        setLoading(false);
      }
    }

    loadMetadata();
  }, []);

  const activeMetrics = getActiveModelMetrics(metadata);
  const activeFeatureImportance = getActiveFeatureImportance(metadata);

  async function handleModelChange(modelName) {
    if (!modelName || modelName === (metadata?.active_model ?? metadata?.selected_model)) {
      return;
    }
    setModelSwitching(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/select-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName })
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Could not switch model");
      }
      if (payload.metadata) {
        setMetadata(payload.metadata);
      }
    } catch (err) {
      setError(err.message || "Failed to switch model");
    } finally {
      setModelSwitching(false);
    }
  }

  function updateField(field, value) {
    setForm((current) => {
      const next = { ...current, [field]: value };
      if (field === "age_group") {
        for (let i = 1; i <= 10; i += 1) {
          next[`aq${i}_response`] = "";
        }
      }
      return next;
    });
  }

  const aq10Questions = aq10QuestionSets[form.age_group] || aq10QuestionSets.adult;

  function buildPayload() {
    const unanswered = aq10Questions.find((question, index) => !form[`aq${index + 1}_response`]);
    if (unanswered) {
      throw new Error("Please answer all 10 AQ-10 questions before submitting.");
    }

    const aqScores = Object.fromEntries(
      aq10Questions.map((question, index) => {
        const answer = form[`aq${index + 1}_response`];
        return [question.field, question.scoring[answer]];
      })
    );

    return {
      age_group: form.age_group,
      age_years: Number(form.age_years),
      sex: form.sex,
      ethnicity: form.ethnicity,
      jaundice_history: form.jaundice_history,
      family_asd: form.family_asd,
      used_screening_app_before: form.used_screening_app_before,
      speech_delay: form.speech_delay,
      anxiety_flag: form.anxiety_flag,
      who_completed_test: form.who_completed_test,
      family_asd_score: Number(form.family_asd_score),
      communication_delay_score: Number(form.communication_delay_score),
      sensory_sensitivity_score: Number(form.sensory_sensitivity_score),
      social_responsiveness_score: Number(form.social_responsiveness_score),
      ...aqScores
    };
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
        body: JSON.stringify(buildPayload())
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
          <p className="eyebrow">AQ-10 Screening</p>
          <h1>Autism screening with age-appropriate AQ-10 items.</h1>
          <p className="hero-copy">
            Child and adolescent use parent/caregiver report (S/he); adult uses self-report (I).
            Responses map to the same binary features for the model before calling the Flask API.
          </p>
        </div>
        <div className="hero-panel">
          <div className="metric-card emphasis">
            <span>Selected model</span>
            <select
              className="model-select"
              aria-label="Classifier model"
              value={metadata?.active_model ?? metadata?.selected_model ?? ""}
              disabled={!metadata?.available_models?.length || modelSwitching}
              onChange={(event) => handleModelChange(event.target.value)}
            >
              {CLASSIFIER_MODEL_IDS.map((name) => {
                const trained = Boolean(metadata?.available_models?.includes(name));
                return (
                  <option key={name} value={name} disabled={!trained}>
                    {trained ? toLabel(name) : `${toLabel(name)} — not available`}
                  </option>
                );
              })}
            </select>
            {metadata && !metadata.available_models?.length ? (
              <p className="model-select-hint">
                Retrain once to enable switching (saves all trained classifiers).
              </p>
            ) : null}
          </div>
          <div className="metric-grid">
            <MetricCard label="Accuracy" value={activeMetrics?.accuracy} />
            <MetricCard label="ROC-AUC" value={activeMetrics?.roc_auc} />
            <MetricCard
              label="FN Reduction"
              value={`${activeMetrics?.false_negative_reduction_pct ?? 0}%`}
            />
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel form-panel">
          <div className="panel-header">
            <h2>AQ-10 Questionnaire</h2>
            <p>
              {form.age_group === "child" || form.age_group === "adolescent"
                ? "Parent or caregiver answers the AQ-10 items about the young person (S/he)."
                : "The respondent answers ten self-report AQ-10 items (I)."}
              {" "}
              Switching age group clears questionnaire answers.
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="subsection">
              <h3>Demographics</h3>
              <div className="form-grid">
                <label>
                  <span>Age group (questionnaire)</span>
                  <select
                    value={form.age_group}
                    onChange={(event) => updateField("age_group", event.target.value)}
                  >
                    <option value="child">Child</option>
                    <option value="adolescent">Adolescent</option>
                    <option value="adult">Adult</option>
                  </select>
                </label>
                <label>
                  <span>{fieldLabels.age_years}</span>
                  <input
                    type="number"
                    min={form.age_group === "child" ? 2 : form.age_group === "adolescent" ? 10 : 18}
                    max={form.age_group === "child" ? 12 : form.age_group === "adolescent" ? 17 : 100}
                    step="1"
                    value={form.age_years}
                    onChange={(event) => updateField("age_years", event.target.value)}
                  />
                </label>

                {Object.entries(categoricalOptions).map(([field, options]) => (
                  <label key={field}>
                    <span>{fieldLabels[field]}</span>
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
              </div>
            </div>

            <div className="subsection">
              <div className="subsection-heading">
                <h3>
                  {form.age_group === "child"
                    ? "AQ-10 Child Items"
                    : form.age_group === "adolescent"
                      ? "AQ-10 Adolescent Items"
                      : "AQ-10 Adult Items"}
                </h3>
                <p>Select the response that best matches the person being screened.</p>
              </div>
              <div className="question-list">
                {aq10Questions.map((question, index) => {
                  const responseField = `aq${index + 1}_response`;
                  return (
                    <div className="question-card" key={`${form.age_group}-${question.field}`}>
                      <div className="question-meta">Question {index + 1}</div>
                      <strong>{question.prompt}</strong>
                      <div className="answer-row">
                        {aq10AnswerOptions.map((option) => (
                          <label className="choice-chip" key={option}>
                            <input
                              type="radio"
                              name={responseField}
                              value={option}
                              checked={form[responseField] === option}
                              onChange={(event) => updateField(responseField, event.target.value)}
                            />
                            <span>{toLabel(option)}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="subsection">
              <div className="subsection-heading">
                <h3>Clinical Context</h3>
                <p>
                  These supporting indicators remain numeric because the current backend model
                  expects normalized values between 0 and 1.
                </p>
              </div>
              <div className="slider-stack">
                {clinicalIndicatorFields.map((field) => (
                  <label className="slider-field" key={field}>
                    <div className="slider-labels">
                      <span>{fieldLabels[field]}</span>
                      <strong>{Number(form[field]).toFixed(2)}</strong>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={form[field]}
                      onChange={(event) => updateField(field, event.target.value)}
                    />
                  </label>
                ))}
              </div>
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
            {(activeFeatureImportance || []).map((feature) => (
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
