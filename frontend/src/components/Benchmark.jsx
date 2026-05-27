import { useEffect, useMemo, useState } from "react";
import api from "../api/client";

function scoreColor(score) {
  if (score >= 0.85) return "text-green-600";
  if (score >= 0.6) return "text-yellow-600";
  return "text-red-600";
}

function statusLabel(score) {
  if (score >= 0.85) return "Strong";
  if (score >= 0.6) return "Borderline";
  return "Weak";
}

export default function Benchmark() {
  const [data, setData] = useState(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const metrics = useMemo(() => {
    if (!data) return [];
    return [
      { label: "Relevance Score", value: data.relevance_score },
      { label: "Source Grounding", value: data.source_grounding },
      { label: "Hallucination Rate", value: data.hallucination_rate },
      { label: "Avg Latency", value: data.avg_latency },
    ];
  }, [data]);

  const results = data?.results || data?.questions || [];

  const runBenchmark = async () => {
    setRunning(true);
    setProgress(0);
    const timer = setInterval(() => {
      setProgress((prev) => (prev < 95 ? prev + 3 : prev));
    }, 900);

    try {
      const response = await api.post("/api/benchmark");
      setData(response.data);
      setProgress(100);
    } catch (error) {
      try {
        const response = await api.get("/api/benchmark");
        setData(response.data);
        setProgress(100);
      } catch (secondError) {
        setData(null);
      }
    } finally {
      clearInterval(timer);
      setRunning(false);
    }
  };

  useEffect(() => {
    if (!running && progress === 100) {
      const timer = setTimeout(() => setProgress(0), 2000);
      return () => clearTimeout(timer);
    }
  }, [running, progress]);

  return (
    <section className="py-12">
      <div className="flex items-center justify-between rounded-none bg-[#FF6600] px-8 py-6 text-white">
        <div>
          <div className="text-[14px] uppercase tracking-[0.2em]">
            RAG Benchmark
          </div>
          <div className="mt-2 text-[24px] font-semibold">
            RAG Benchmark
          </div>
        </div>
        <div className="text-[48px] font-semibold">
          {data?.overall_score ?? "0.00"}
        </div>
      </div>

      <div className="mt-8 grid gap-6 md:grid-cols-2">
        {metrics.map((metric) => (
          <div
            key={metric.label}
            className="border border-[#eee] bg-white p-6"
          >
            <div className="text-[36px] font-semibold text-[#FF6600]">
              {metric.value ?? "-"}
            </div>
            <div className="mt-1 text-[12px] uppercase tracking-[0.2em] text-[#666]">
              {metric.label}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 flex items-center justify-between">
        <button className="btn-yc" onClick={runBenchmark} disabled={running}>
          {running ? "Running..." : "Run Benchmark"}
        </button>
        {running && (
          <div className="w-48">
            <div className="h-2 w-full bg-[#f1f1f1]">
              <div
                className="h-2 bg-[#FF6600]"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="mt-8 overflow-x-auto border border-[#eee]">
        <table className="w-full text-left text-[13px]">
          <thead className="bg-[#fafafa] text-[11px] uppercase tracking-[0.2em] text-[#666]">
            <tr>
              <th className="px-4 py-3">#</th>
              <th className="px-4 py-3">Question</th>
              <th className="px-4 py-3">Score</th>
              <th className="px-4 py-3">Category</th>
              <th className="px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {results.map((item, index) => (
              <tr key={`row-${index}`} className="table-row-alt">
                <td className="px-4 py-3 text-[#666]">{index + 1}</td>
                <td className="px-4 py-3 text-[#111]">
                  {(item.question || "").slice(0, 60)}
                </td>
                <td className={`px-4 py-3 ${scoreColor(item.score || 0)}`}>
                  {item.score ?? "-"}
                </td>
                <td className="px-4 py-3 text-[#666]">
                  {item.category || "General"}
                </td>
                <td className="px-4 py-3">
                  <span className="rounded-full bg-[#f2f2f2] px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-[#666]">
                    {statusLabel(item.score || 0)}
                  </span>
                </td>
              </tr>
            ))}
            {results.length === 0 && (
              <tr>
                <td
                  className="px-4 py-6 text-center text-[#666]"
                  colSpan={5}
                >
                  Run the benchmark to see results.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
