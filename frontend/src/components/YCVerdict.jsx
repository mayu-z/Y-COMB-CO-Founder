import { useEffect, useMemo, useState } from "react";
import api from "../api/client";

const meterLabels = ["Conviction", "Clarity", "Urgency", "Traction"];

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
}

export default function YCVerdict({ startupProfile, setStartupProfile }) {
  const [profile, setProfile] = useState(startupProfile || {});
  const [verdictData, setVerdictData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    setProfile(startupProfile || {});
  }, [startupProfile]);

  const meters = useMemo(() => {
    const raw = verdictData?.meters || verdictData?.scores || {};
    return meterLabels.map((label) => {
      const key = label.toLowerCase();
      const value = raw[key] || raw[label] || verdictData?.[key] || 0;
      return { label, value: Number(value) || 0 };
    });
  }, [verdictData]);

  const verdictLabel =
    verdictData?.label ||
    verdictData?.verdict ||
    verdictData?.outcome ||
    "Pending";

  const tagline = verdictData?.tagline || verdictData?.summary || "";
  const interviewChance =
    verdictData?.interview_chance || verdictData?.interviewChance || 0;
  const founderFit =
    verdictData?.founder_market_fit || verdictData?.founderFit || 0;

  const strongest = verdictData?.strongest || "";
  const weakest = verdictData?.weakest || "";
  const dnaMatch = verdictData?.dna_match || verdictData?.dnaMatch || "";
  const dnaReason = verdictData?.dna_reason || verdictData?.dnaReason || "";

  const partnerNotes =
    verdictData?.partner_notes || verdictData?.notes || [];
  const improvements =
    verdictData?.improvements || verdictData?.top_improvements || [];

  const handleChange = (event) => {
    const { name, value } = event.target;
    setProfile((prev) => ({ ...prev, [name]: value }));
  };

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const { data } = await api.post("/api/verdict", profile);
      setVerdictData(data);
      setStartupProfile(profile);
    } catch (error) {
      setVerdictData({ error: "Verdict failed. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const response = await api.post("/api/verdict", profile, {
        responseType: "blob",
      });
      downloadBlob(response.data, "yc-verdict.pdf");
    } catch (error) {
      setVerdictData({ error: "Download failed. Please try again." });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <section className="py-12">
      <div className="space-y-6">
        <div className="border border-[#eee] bg-white p-6">
          <label className="block text-[12px] uppercase tracking-[0.1em] text-[#555]">
            One-liner
          </label>
          <input
            name="one_liner"
            value={profile.one_liner || ""}
            onChange={handleChange}
            placeholder="We build X for Y to do Z"
            className="mt-3 w-full rounded-none border border-gray-200 px-4 py-3 text-[14px]"
          />
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <button onClick={handleGenerate} className="btn-yc">
              Generate Verdict →
            </button>
            <button
              onClick={handleDownload}
              className="btn-outline"
              disabled={downloading}
            >
              {downloading ? "Preparing..." : "Download Verdict PDF →"}
            </button>
          </div>
        </div>

        {loading && (
          <div className="border border-[#eee] bg-white p-8 animate-pulse">
            <div className="h-16 w-64 bg-[#f2f2f2]" />
            <div className="mt-4 h-6 w-48 bg-[#f2f2f2]" />
            <div className="mt-8 h-40 w-full bg-[#f2f2f2]" />
          </div>
        )}

        {verdictData && !loading && (
          <div className="space-y-10">
            {verdictData.error && (
              <div className="border border-[#eee] bg-white p-6 text-[14px] text-[#cc0000]">
                {verdictData.error}
              </div>
            )}
            {!verdictData.error && (
              <>
                <div className="border border-[#eee] bg-white p-10 shadow-sm">
                  <div className="text-[10px] uppercase tracking-[0.4em] text-[#FF6600]">
                    YC Partner Verdict
                  </div>
                  <div className="mt-6 font-serif text-[48px] font-bold text-black md:text-[72px]">
                    {verdictLabel}
                  </div>
                  <div className="mt-3 text-[18px] italic text-[#666]">
                    {tagline}
                  </div>

                  <div className="mt-10 grid gap-6 md:grid-cols-2">
                    <div>
                      <div className="font-serif text-[64px] font-bold text-[#FF6600]">
                        {interviewChance}%
                      </div>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-[#999]">
                        Interview Chance
                      </div>
                    </div>
                    <div>
                      <div className="font-serif text-[64px] font-bold text-[#FF6600]">
                        {founderFit}
                      </div>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-[#999]">
                        Founder Market Fit
                      </div>
                    </div>
                  </div>

                  <div className="mt-10 grid gap-6 md:grid-cols-3">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-green-600">
                        Strongest
                      </div>
                      <div className="mt-2 text-[14px] font-semibold text-black">
                        {strongest}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-red-600">
                        Weakest
                      </div>
                      <div className="mt-2 text-[14px] font-semibold text-black">
                        {weakest}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gray-500">
                        DNA Match
                      </div>
                      <div className="mt-2 text-[14px] font-semibold text-black">
                        {dnaMatch}
                      </div>
                      <div className="text-[12px] italic text-[#777]">
                        {dnaReason}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <div className="text-[12px] uppercase tracking-[0.2em] text-[#555]">
                      Partner Notes
                    </div>
                    <div className="mt-4 grid gap-4 md:grid-cols-3">
                      {[0, 1, 2].map((index) => (
                        <div
                          key={`note-${index}`}
                          className="flex gap-3 bg-[#1a1a1a] p-4 text-white"
                        >
                          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#FF6600] text-[11px] font-semibold">
                            YC
                          </div>
                          <div className="text-[14px] leading-relaxed">
                            {partnerNotes[index] ||
                              "A YC partner will share concise feedback here."}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="text-[12px] uppercase tracking-[0.2em] text-[#555]">
                      Before You Reapply
                    </div>
                    <div className="mt-4 space-y-4">
                      {[0, 1, 2].map((index) => (
                        <div
                          key={`improve-${index}`}
                          className="flex items-start gap-4 border border-[#eee] bg-white p-4"
                        >
                          <div className="font-serif text-[32px] font-bold text-[#FF6600]">
                            {index + 1}
                          </div>
                          <div className="text-[14px] text-[#555]">
                            {improvements[index] ||
                              "Tighten your thesis and show sharper traction milestones."}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="grid gap-6 md:grid-cols-4">
                    {meters.map((meter) => (
                      <div
                        key={meter.label}
                        className="border border-[#eee] bg-white p-4"
                      >
                        <div className="text-[12px] uppercase tracking-[0.2em] text-[#666]">
                          {meter.label}
                        </div>
                        <div className="mt-2 text-[32px] font-semibold text-[#111]">
                          {meter.value}
                          <span className="text-[12px] text-[#999]">/100</span>
                        </div>
                        <div className="mt-3 h-2 w-full bg-[#f1f1f1]">
                          <div
                            className="h-2 bg-[#FF6600]"
                            style={{ width: `${Math.min(meter.value, 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>

                  <button className="btn-yc w-full" onClick={handleDownload}>
                    Download Verdict PDF →
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
