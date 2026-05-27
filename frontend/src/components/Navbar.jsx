import { useState } from "react";

const tabs = [
  { id: "ask", label: "Ask YC" },
  { id: "evaluate", label: "Evaluate" },
  { id: "verdict", label: "Verdict" },
  { id: "companies", label: "Companies" },
  { id: "benchmark", label: "Benchmark" },
];

export default function Navbar({ activeTab, setActiveTab }) {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-[#e5e5e5] bg-white/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center bg-[#FF6600]">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M6 4L12 12L18 4H21L13.5 14.5V20H10.5V14.5L3 4H6Z"
                fill="white"
              />
            </svg>
          </div>
          <span className="text-[15px] font-semibold text-black">
            Co-Founder
          </span>
        </div>

        <nav className="hidden items-center gap-6 md:flex">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative pb-2 text-[14px] font-medium text-[#333] transition-colors hover:text-[#FF6600] ${
                activeTab === tab.id ? "text-[#FF6600]" : ""
              }`}
            >
              {tab.label}
              {activeTab === tab.id && (
                <span className="absolute bottom-0 left-0 h-[2px] w-full bg-[#FF6600]" />
              )}
            </button>
          ))}
        </nav>

        <button
          onClick={() => setOpen((prev) => !prev)}
          className="md:hidden"
          aria-label="Toggle navigation"
        >
          <div className="flex h-8 w-8 flex-col items-center justify-center gap-1">
            <span className="h-[2px] w-5 bg-black" />
            <span className="h-[2px] w-5 bg-black" />
            <span className="h-[2px] w-5 bg-black" />
          </div>
        </button>
      </div>

      {open && (
        <div className="border-t border-[#e5e5e5] bg-white px-6 py-4 md:hidden">
          <div className="flex flex-col gap-4">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setOpen(false);
                }}
                className={`text-left text-[14px] font-medium ${
                  activeTab === tab.id ? "text-[#FF6600]" : "text-[#333]"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
