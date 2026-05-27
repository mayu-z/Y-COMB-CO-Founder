import { useEffect, useMemo, useState } from "react";
import api from "../api/client";

const batches = [
  "All",
  "W24",
  "S24",
  "W23",
  "S23",
  "W22",
  "S22",
  "W21",
];

const statusOptions = ["All", "Active", "Acquired", "Public"];

export default function BrowseCompanies() {
  const [search, setSearch] = useState("");
  const [batch, setBatch] = useState("All");
  const [status, setStatus] = useState("All");
  const [limit, setLimit] = useState(50);
  const [companies, setCompanies] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handle = setTimeout(() => {
      fetchCompanies();
    }, 300);
    return () => clearTimeout(handle);
  }, [search, batch, limit]);

  const fetchCompanies = async () => {
    setLoading(true);
    try {
      const { data } = await api.get("/api/companies", {
        params: {
          search,
          batch: batch === "All" ? "" : batch,
          limit,
        },
      });
      setCompanies(data.companies || data || []);
    } catch (error) {
      setCompanies([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredCompanies = useMemo(() => {
    if (status === "All") return companies;
    return companies.filter(
      (company) =>
        (company.status || "").toLowerCase() === status.toLowerCase()
    );
  }, [companies, status]);

  return (
    <section className="py-12">
      <div className="border-b border-[#eee] pb-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center">
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search companies"
            className="flex-1 border-none border-b border-[#ddd] px-2 py-2 text-[14px] focus:border-[#FF6600]"
          />
          <select
            value={batch}
            onChange={(event) => setBatch(event.target.value)}
            className="border-none border-b border-[#ddd] px-2 py-2 text-[14px]"
          >
            {batches.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <select
            value={status}
            onChange={(event) => setStatus(event.target.value)}
            className="border-none border-b border-[#ddd] px-2 py-2 text-[14px]"
          >
            {statusOptions.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {loading && (
          <div className="col-span-full text-[14px] text-[#666]">
            Loading companies...
          </div>
        )}
        {!loading && filteredCompanies.length === 0 && (
          <div className="col-span-full text-[14px] text-[#666]">
            No companies found.
          </div>
        )}
        {filteredCompanies.map((company, index) => (
          <div
            key={`${company.name}-${index}`}
            className="cursor-pointer border border-[#eee] bg-white p-5 transition-shadow hover:shadow-md"
          >
            <div className="flex items-center justify-between">
              <div className="text-[16px] font-semibold text-black">
                {company.name || "Company"}
              </div>
              <span className="rounded-full bg-[#f2f2f2] px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-[#666]">
                {company.batch || "Batch"}
              </span>
            </div>
            <div className="mt-2">
              <span
                className={`rounded-full px-2 py-1 text-[10px] uppercase tracking-[0.2em] ${
                  company.status === "Public"
                    ? "bg-blue-100 text-blue-600"
                    : company.status === "Acquired"
                      ? "bg-orange-100 text-orange-600"
                      : "bg-green-100 text-green-600"
                }`}
              >
                {company.status || "Active"}
              </span>
            </div>
            <p className="mt-3 line-clamp-2 text-[13px] text-[#666]">
              {company.description || "Description coming soon."}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {(company.tags || company.industries || ["SaaS"]).map(
                (tag, tagIndex) => (
                  <span
                    key={`${tag}-${tagIndex}`}
                    className="rounded-full bg-[#f2f2f2] px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-[#666]"
                  >
                    {tag}
                  </span>
                )
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10">
        <button
          className="btn-outline"
          onClick={() => setLimit((prev) => prev + 50)}
        >
          Load More
        </button>
      </div>
    </section>
  );
}
