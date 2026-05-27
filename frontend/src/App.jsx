import { useEffect, useState } from "react";
import Navbar from "./components/Navbar.jsx";
import Hero from "./components/Hero.jsx";
import AskYC from "./components/AskYC.jsx";
import EvaluateStartup from "./components/EvaluateStartup.jsx";
import YCVerdict from "./components/YCVerdict.jsx";
import BrowseCompanies from "./components/BrowseCompanies.jsx";
import Benchmark from "./components/Benchmark.jsx";
import Footer from "./components/Footer.jsx";

export default function App() {
  const [activeTab, setActiveTab] = useState("ask");
  const [startupProfile, setStartupProfile] = useState(
    JSON.parse(localStorage.getItem("startup_profile") || "{}")
  );

  useEffect(() => {
    localStorage.setItem("startup_profile", JSON.stringify(startupProfile));
  }, [startupProfile]);

  return (
    <div className="min-h-screen bg-white text-neutral-900">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      <Hero
        onPrimary={() => setActiveTab("verdict")}
        onSecondary={() => setActiveTab("ask")}
      />
      <main className="mx-auto w-full max-w-6xl px-6 pb-20">
        {activeTab === "ask" && (
          <AskYC startupProfile={startupProfile} />
        )}
        {activeTab === "evaluate" && (
          <EvaluateStartup
            startupProfile={startupProfile}
            setStartupProfile={setStartupProfile}
          />
        )}
        {activeTab === "verdict" && (
          <YCVerdict
            startupProfile={startupProfile}
            setStartupProfile={setStartupProfile}
          />
        )}
        {activeTab === "companies" && <BrowseCompanies />}
        {activeTab === "benchmark" && <Benchmark />}
      </main>
      <Footer />
    </div>
  );
}
