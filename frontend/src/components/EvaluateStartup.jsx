import { useEffect, useMemo, useState } from "react";
import api from "../api/client";

const fieldConfig = [
  {
    name: "one_liner",
    label: "One-liner",
    type: "text",
    placeholder: "We build X for Y to do Z",
  },
  { name: "problem", label: "Problem", type: "textarea", rows: 3 },
  { name: "market_size", label: "Market Size", type: "text" },
  { name: "traction", label: "Traction", type: "textarea", rows: 2 },
  { name: "team_size", label: "Team Size", type: "number" },
  { name: "background", label: "Background", type: "textarea", rows: 2 },
  {
    name: "working_how_long",
    label: "Working How Long",
    type: "text",
  },
  { name: "why_now", label: "Why Now", type: "textarea", rows: 2 },
  { name: "biggest_risk", label: "Biggest Risk", type: "textarea", rows: 2 },
  {
    name: "yc_batch",
    label: "YC Batch",
    type: "text",
    placeholder: "e.g. W25",
  },
];

const dimensionLabels = [
  "Problem Clarity",
  "Market Size",
  "Traction",
  "Team Strength",
  "Timing",
];

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
}

export default function EvaluateStartup({ startupProfile, setStartupProfile }) {
  const [form, setForm] = useState({
    one_liner: "",
    problem: "",
    market_size: "",
    traction: "",
    team_size: "",
    background: "",
    working_how_long: "",
    why_now: "",
    biggest_risk: "",
    yc_batch: "",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (startupProfile && Object.keys(startupProfile).length > 0) {
      setForm((prev) => ({ ...prev, ...startupProfile }));
    }
  }, [startupProfile]);

  const scores = useMemo(() => {
    const dimensionScores =
      result?.dimension_scores || result?.dimensions || result?.scores || {};

    return dimensionLabels.map((label) => {
      const key = label.toLowerCase().replace(/\s+/g, "_");
      const value =
        dimensionScores[key] ||
        dimensionScores[label] ||
        result?.[key] ||
        0;
      return { label, value: Number(value) || 0 };
    });
  }, [result]);

  const overallScore =
    result?.yc_fit_score || result?.fit_score || result?.score || 0;

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const { data } = await api.post("/api/evaluate", form);
      setResult(data);
    } catch (error) {
      setResult({ error: "Evaluation failed. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const handleSave = () => {
    setStartupProfile(form);
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const response = await api.post("/api/evaluate", form, {
        responseType: "blob",
      });
      downloadBlob(response.data, "yc-evaluation.pdf");
    } catch (error) {
      setResult({ error: "Download failed. Please try again." });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <section className="py-12">
      <div className="grid gap-10 lg:grid-cols-[3fr_2fr]">
        <form
          onSubmit={handleSubmit}
          className="space-y-6 border border-[#eee] bg-white p-8"
        >
          {fieldConfig.map((field) => (
            <label key={field.name} className="block">
              <div className="mb-2 text-[12px] uppercase tracking-[0.1em] text-[#555]">
                {field.label}
              </div>
              {field.type === "textarea" ? (
                <textarea
                  name={field.name}
                  rows={field.rows}
                  value={form[field.name]}
                  onChange={handleChange}
                  className="w-full rounded-none border border-gray-200 px-4 py-3 text-[14px]"
                />
              ) : (
                <input
                  name={field.name}
                  type={field.type}
                  value={form[field.name]}
                  onChange={handleChange}
                  placeholder={field.placeholder}
                  className="w-full rounded-none border border-gray-200 px-4 py-3 text-[14px]"
                />
              )}
            </label>
          ))}
          <button type="submit" className="btn-yc w-full">
            Evaluate My Startup →
          </button>
        </form>

        <div className="space-y-6">
          {!result && !loading && (
            <div className="border border-[#eee] bg-white p-8 text-[14px] text-[#666]">
              Submit the form to see your YC fit score and breakdown.
            </div>
          )}
          {loading && (
            <div className="space-y-6 border border-[#eee] bg-white p-8 animate-pulse">
              <div className="h-12 w-32 bg-[#f2f2f2]" />
              <div className="h-6 w-48 bg-[#f2f2f2]" />
              <div className="h-36 w-full bg-[#f2f2f2]" />
            </div>
          )}
          {result && !loading && (
            <div className="space-y-6 border border-[#eee] bg-white p-8">
              {result.error ? (
                <div className="text-[14px] text-[#cc0000]">
                  {result.error}
                </div>
              ) : (
                <>
                  <div>
                    <div className="font-serif text-[96px] font-bold text-[#FF6600]">
                      {Math.round(Number(overallScore) || 0)}
                    </div>
                    <div className="text-[12px] uppercase tracking-[0.2em] text-[#999]">
                      YC Fit Score
                    </div>
                  </div>
                  <div className="space-y-4">
                    {scores.map((score) => (
                      <div key={score.label}>
                        <div className="mb-2 flex items-center justify-between text-[12px] uppercase tracking-[0.1em] text-[#666]">
                          <span>{score.label}</span>
                          <span>{score.value}/10</span>
                        </div>
                        <div className="h-2 w-full bg-[#f1f1f1]">
                          <div
                            className="h-2 bg-[#FF6600]"
                            style={{ width: `${Math.min(score.value * 10, 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <button onClick={handleSave} className="btn-outline">
                      Save to Memory
                    </button>
                    <button
                      onClick={handleDownload}
                      className="btn-yc"
                      disabled={downloading}
                    >
                      {downloading ? "Preparing..." : "Download PDF"}
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
