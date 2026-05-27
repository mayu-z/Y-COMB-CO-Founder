import { useEffect, useRef, useState } from "react";
import api from "../api/client";

const defaultPrompt =
  "Ask about fundraising, GTM, product-market fit...";

function buildStartupContext(profile) {
  if (!profile || Object.keys(profile).length === 0) return "";
  return Object.entries(profile)
    .filter(([, value]) => value)
    .map(([key, value]) => `${key}: ${value}`)
    .join(" | ");
}

function normalizeSources(data) {
  if (!data) return [];
  if (Array.isArray(data.sources)) return data.sources;
  if (Array.isArray(data.contexts)) {
    return data.contexts.map((text, index) => ({
      source_type: "context",
      snippet: text,
      chunk_id: index,
    }));
  }
  return [];
}

export default function AskYC({ startupProfile }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (endRef.current) {
      endRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const { data } = await api.post("/api/ask", {
        question,
        startup_context: buildStartupContext(startupProfile),
      });

      const assistantText =
        data?.answer ||
        data?.response ||
        data?.result ||
        data?.message ||
        "No response yet.";

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: assistantText },
      ]);
      setSources(normalizeSources(data));
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Something went wrong. Please try again.",
        },
      ]);
      setSources([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="py-12">
      <div className="grid gap-8 lg:grid-cols-[3fr_2fr]">
        <div className="flex flex-col rounded-none border border-[#eee] bg-white">
          <div className="flex-1 space-y-6 p-6">
            {messages.length === 0 && !loading && (
              <div className="text-[14px] leading-relaxed text-[#666]">
                Ask YC anything about fundraising, founder-market fit, or GTM.
              </div>
            )}
            {messages.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "assistant" && (
                  <div className="mr-3 flex h-8 w-8 items-center justify-center rounded-full bg-[#FF6600] text-[11px] font-semibold text-white">
                    YC
                  </div>
                )}
                <div
                  className={`max-w-[80%] text-[14px] leading-relaxed ${
                    message.role === "user"
                      ? "rounded-none bg-[#f5f5f5] px-4 py-3 text-right"
                      : "border-l-2 border-[#FF6600] bg-white px-4 py-3 shadow-sm"
                  }`}
                >
                  {message.text}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex items-center gap-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#FF6600] text-[11px] font-semibold text-white">
                  YC
                </div>
                <div className="flex items-center gap-2 rounded-none border-l-2 border-[#FF6600] bg-white px-4 py-3 shadow-sm">
                  <span className="dot-1 h-2 w-2 rounded-full bg-[#FF6600]" />
                  <span className="dot-2 h-2 w-2 rounded-full bg-[#FF6600]" />
                  <span className="dot-3 h-2 w-2 rounded-full bg-[#FF6600]" />
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>
          <form
            onSubmit={handleSubmit}
            className="sticky bottom-0 border-t border-[#eee] bg-white p-4"
          >
            <div className="flex gap-3">
              <input
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder={defaultPrompt}
                className="flex-1 rounded-none border border-[#ddd] px-4 py-3 text-[14px]"
              />
              <button type="submit" className="btn-yc">
                Ask
              </button>
            </div>
          </form>
        </div>

        <aside className="space-y-4">
          <div className="text-[14px] font-semibold text-[#111]">Sources</div>
          {loading && (
            <div className="rounded-none border border-[#eee] bg-white p-4 text-[13px] text-[#777]">
              Fetching sources...
            </div>
          )}
          {!loading && sources.length === 0 && (
            <div className="rounded-none border border-[#eee] bg-white p-4 text-[13px] text-[#777]">
              Sources will appear here after the first answer.
            </div>
          )}
          {sources.map((source, index) => (
            <div
              key={`${source.chunk_id}-${index}`}
              className="rounded-none border border-[#eee] bg-white p-4"
            >
              <div className="mb-2 inline-flex items-center rounded-full bg-[#FF6600]/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-[#FF6600]">
                {source.source_type || "source"}
              </div>
              <p className="text-[13px] leading-relaxed text-[#666]">
                {source.snippet || source.text || "Source snippet unavailable."}
              </p>
              <div className="mt-2 text-[10px] uppercase tracking-[0.2em] text-[#bbb]">
                {source.chunk_id ?? "chunk"}
              </div>
            </div>
          ))}
        </aside>
      </div>
    </section>
  );
}
