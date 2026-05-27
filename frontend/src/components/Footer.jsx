export default function Footer() {
  return (
    <footer className="bg-[#111] text-white">
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        <div className="grid gap-10 md:grid-cols-[2fr_1fr_1fr_1fr]">
          <div>
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
              <div className="text-[14px] font-semibold">YC Co-Founder</div>
            </div>
            <div className="mt-4 text-[14px] text-[#ccc]">
              Make something people want.
            </div>
          </div>
          <div>
            <div className="text-[12px] uppercase tracking-[0.2em] text-[#999]">
              Features
            </div>
            <ul className="mt-4 space-y-2 text-[14px] text-[#ccc]">
              <li>Ask YC</li>
              <li>Evaluate</li>
              <li>Verdict</li>
            </ul>
          </div>
          <div>
            <div className="text-[12px] uppercase tracking-[0.2em] text-[#999]">
              Resources
            </div>
            <ul className="mt-4 space-y-2 text-[14px] text-[#ccc]">
              <li>YC Library</li>
              <li>Startup School</li>
              <li>Benchmark</li>
            </ul>
          </div>
          <div>
            <div className="text-[12px] uppercase tracking-[0.2em] text-[#999]">
              Project
            </div>
            <ul className="mt-4 space-y-2 text-[14px] text-[#ccc]">
              <li>
                <a
                  className="hover:text-white"
                  href="https://github.com/mayu-z/Y-COMB-CO-Founder"
                  target="_blank"
                  rel="noreferrer"
                >
                  GitHub
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-10 border-t border-[#222] pt-6 text-[12px] text-[#777]">
          © 2025 YC Co-Founder — Built by Mayuresh Singh | IFIM Bangalore
        </div>
      </div>
    </footer>
  );
}
