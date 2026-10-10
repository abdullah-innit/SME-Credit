"use client";

import { useEffect, useState } from "react";

interface HealthData {
  status: string;
  version: string;
  offline_mode: boolean;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: HealthData = await res.json();
        setHealth(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Connection failed");
        setHealth(null);
      } finally {
        setLoading(false);
      }
    };
    checkHealth();
  }, []);

  return (
    <main className="flex-1 flex flex-col items-center justify-center px-4 py-16">
      {/* ── Hero ──────────────────────────────────────────────── */}
      <div className="text-center max-w-3xl mx-auto mb-16">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-medium mb-8">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          AICON&apos;26 — Build With AI · Financial Operations
        </div>

        <h1 className="text-5xl sm:text-6xl font-bold tracking-tight mb-6">
          <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent">
            SME Credit
          </span>
        </h1>

        <p className="text-xl text-gray-400 leading-relaxed max-w-2xl mx-auto">
          AI-powered credit scrutiny for Pakistani SMEs.{" "}
          <span className="text-gray-300">
            Score businesses on real cash-flow behaviour
          </span>
          , not just collateral and bureau history.
        </p>

        <p className="mt-4 text-sm text-amber-400/80 font-medium">
          ⚠️ All data shown is synthetic — no real customers or bank
          partnerships.
        </p>
      </div>

      {/* ── Backend Status Card ───────────────────────────────── */}
      <div className="w-full max-w-md">
        <div className="rounded-2xl border border-gray-800 bg-gray-900/50 backdrop-blur-sm p-6 shadow-xl shadow-black/20">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Backend Status
          </h2>

          {loading && (
            <div className="flex items-center gap-3 text-gray-500">
              <svg
                className="animate-spin h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Connecting to backend…
            </div>
          )}

          {error && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-red-400">
                <span className="h-3 w-3 rounded-full bg-red-500" />
                <span className="font-medium">Disconnected</span>
              </div>
              <p className="text-sm text-gray-500">{error}</p>
              <p className="text-xs text-gray-600">
                Make sure the backend is running on{" "}
                <code className="text-gray-500">localhost:8000</code>
              </p>
            </div>
          )}

          {health && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-emerald-500 shadow-lg shadow-emerald-500/30" />
                  <span className="font-medium text-emerald-400">
                    Connected
                  </span>
                </div>
                <span className="text-xs text-gray-600 font-mono">
                  v{health.version}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="rounded-lg bg-gray-800/50 p-3">
                  <p className="text-xs text-gray-500 mb-1">Status</p>
                  <p className="text-sm font-medium text-gray-200 capitalize">
                    {health.status}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-800/50 p-3">
                  <p className="text-xs text-gray-500 mb-1">Mode</p>
                  <p className="text-sm font-medium text-gray-200">
                    {health.offline_mode ? "🔒 Offline" : "🌐 Online"}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── SDG Badges ────────────────────────────────────────── */}
      <div className="mt-12 flex items-center gap-4 text-xs text-gray-600">
        <span className="px-3 py-1 rounded-full border border-gray-800">
          SDG 8 — Decent Work &amp; Growth
        </span>
        <span className="px-3 py-1 rounded-full border border-gray-800">
          SDG 9 — Industry &amp; Innovation
        </span>
      </div>
    </main>
  );
}
