const stats = [
  { value: "2,808", label: "Chunks" },
  { value: "1,494", label: "Companies" },
  { value: "100Q", label: "Benchmark" },
];

const ticker = [
  "Airbnb",
  "Stripe",
  "Dropbox",
  "DoorDash",
  "Brex",
  "Coinbase",
  "GitLab",
  "Gusto",
  "Scale AI",
  "Instacart",
  "OpenAI",
  "Reddit",
  "Twitch",
  "Figma",
  "Cruise",
];

export default function Hero({ onPrimary, onSecondary }) {
  return (
    <section className="border-b border-[#f2f2f2]">
      <div className="mx-auto w-full max-w-6xl px-6 py-16">
        <h1 className="font-serif text-[40px] leading-tight text-black md:text-[64px]">
          YC Co-Founder turns builders
          <br />
          into formidable founders.
        </h1>
        <p className="mt-6 max-w-2xl text-[18px] leading-relaxed text-[#555]">
          RAG-powered startup advisor trained on 2,808 chunks of real YC
          knowledge — Paul Graham essays, 1,494 funded companies, Startup School,
          and more.
        </p>
        <div className="mt-8 flex flex-col gap-4 sm:flex-row">
          <button className="btn-yc" onClick={onPrimary}>
            Get Your Verdict →
          </button>
          <button className="btn-outline" onClick={onSecondary}>
            Ask a Question
          </button>
        </div>

        <div className="mt-12 grid gap-8 border-y border-[#f2f2f2] py-8 sm:grid-cols-3">
          {stats.map((stat) => (
            <div key={stat.label}>
              <div className="font-serif text-[32px] font-bold text-[#FF6600]">
                {stat.value}
              </div>
              <div className="text-[12px] uppercase tracking-[0.2em] text-[#999]">
                {stat.label}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 marquee text-[13px] uppercase tracking-[0.2em] text-[#999]">
          <div className="marquee-track">
            {ticker.map((name, index) => (
              <span key={`${name}-${index}`}>
                {name}
                <span className="px-3">•</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
