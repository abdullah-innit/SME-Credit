"use client";
import { useEffect, useState } from "react";

const DECISION_TEXT: Record<string, string> = { approve: "Approve", refer: "Refer to committee", decline: "Decline" };
const COLOR: Record<string, string> = { approve: "var(--approve)", refer: "var(--refer)", decline: "var(--decline)" };
const pkr = (n: number) => `Rs. ${Math.round(n).toLocaleString("en-PK")}`;

function pointOnArc(score: number, r: number) {
  const f = (score - 300) / 550;
  return [110 - r * Math.cos(Math.PI * f), 110 - r * Math.sin(Math.PI * f)];
}

function Dial({ score, decision }: { score: number; decision: string }) {
  const L = Math.PI * 90;
  const frac = (score - 300) / 550;
  const ticks = [685, 767.5].map((s) => [pointOnArc(s, 80), pointOnArc(s, 100)]);
  return (
    <svg className="dial" viewBox="0 0 220 140" role="img" aria-label={`Credit score ${score} out of 850`}>
      <path className="track" d="M 20 110 A 90 90 0 0 1 200 110" fill="none" strokeWidth="14" strokeLinecap="round" />
      <path className="fill" d="M 20 110 A 90 90 0 0 1 200 110" fill="none" strokeWidth="14" strokeLinecap="round"
        stroke={COLOR[decision]} strokeDasharray={L} strokeDashoffset={L * (1 - frac)} />
      {ticks.map(([a, b], i) => <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#14232b" strokeWidth="1.5" />)}
      <text className="score" x="110" y="98" textAnchor="middle">{score}</text>
      <text className="small" x="110" y="118" textAnchor="middle">score out of 850</text>
      <text className="small" x="20" y="130" textAnchor="middle">300</text>
      <text className="small" x="200" y="130" textAnchor="middle">850</text>
    </svg>
  );
}

export default function Page() {
  const [impact, setImpact] = useState<any>(null);
  const [list, setList] = useState<any[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [applied, setApplied] = useState(0);
  const [committee, setCommittee] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/impact").then((r) => r.json()).then(setImpact).catch(() => setError("Cannot reach the backend on port 8000."));
    fetch("/api/applicants").then((r) => r.json()).then((l) => { setList(l); setSel(l[0]?.id ?? null); }).catch(() => {});
  }, []);

  useEffect(() => {
    if (sel == null) return;
    setDetail(null); setCommittee(null); setApplied(0);
    fetch(`/api/applicants/${sel}`).then((r) => r.json()).then(setDetail);
  }, [sel]);

  async function convene() {
    setBusy(true);
    try {
      const r = await fetch("/api/committee", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: sel }) });
      setCommittee(await r.json());
    } finally { setBusy(false); }
  }

  const a = detail?.assessment;
  const steps = detail?.path?.steps ?? [];
  const shown = a && (applied === 0 ? a : { ...a, pd: steps[applied - 1].pd_after, score: steps[applied - 1].score_after,
    decision: applied === steps.length ? detail.path.final.decision : "refer" });
  const maxImpact = a ? Math.max(...a.factors.map((f: any) => Math.abs(f.impact_pts)), 1) : 1;
  const shownDecision = shown && (shown.pd <= 0.15 ? "approve" : shown.pd > 0.3 ? "decline" : "refer");

  return (
    <>
      <div className="impact">
        {error && <span className="err">{error}</span>}
        {impact && (<>
          On {impact.sample_size} held-out small businesses, alternative-data scoring approves <b>{impact.ai_approved}</b> versus <b>{impact.traditional_approved}</b> under a credit-history rule,
          including <b>{impact.thin_file_approved_by_ai}</b> with no credit file, at a {impact.ai_default_rate}% default rate versus {impact.traditional_default_rate}%.
          That is about <b>{impact.estimated_jobs_supported}</b> more jobs financed, with decisions in minutes instead of {impact.decision_time.traditional_days} days.
          <small>{impact.assumptions}</small>
        </>)}
      </div>
      <div className="shell">
        <nav className="rail" aria-label="Applicants">
          <h2>Applications</h2>
          {list.map((x) => (
            <button key={x.id} className="row" aria-current={sel === x.id} onClick={() => setSel(x.id)}>
              <div className="name"><span className={`dot ${x.decision}`} />{x.name}</div>
              <div className="sub"><span>{x.city}{x.thin_file ? ", no credit file" : ""}</span><span>{pkr(x.loan_amount)}</span></div>
            </button>
          ))}
        </nav>
        <main className="main">
          {!detail && <p>Loading application…</p>}
          {detail && a && shown && (<>
            <div className="head">
              <h1>{detail.record.name}</h1>
              <p>{detail.record.sector} in {detail.record.city}. Requesting {pkr(applied && steps.length ? (applied === steps.length ? detail.path.final.loan_amount : detail.record.loan_amount) : detail.record.loan_amount)}
                {!detail.record.has_credit_history && <span className="tag">No credit file</span>}</p>
            </div>
            <div className="top">
              <Dial score={shown.score} decision={shownDecision} />
              <div>
                <div className={`verdict ${shownDecision}`}>{DECISION_TEXT[shownDecision]}</div>
                <div className="risk">Estimated default risk {(shown.pd * 100).toFixed(1)}%. Approval needs 15% or less.</div>
              </div>
            </div>

            <section className="block">
              <h2>What is driving this score</h2>
              <p className="hint">Each bar shows how many points of default risk a factor adds or removes compared with an average applicant.</p>
              {a.factors.map((f: any) => (
                <div className="factor" key={f.key}>
                  <div className="lbl">{f.label}<span>{f.value}</span></div>
                  <div className="bar"><i className={f.impact_pts < 0 ? "good" : "bad"} style={{ width: `${(Math.abs(f.impact_pts) / maxImpact) * 50}%` }} /></div>
                  <div className="pts">{f.impact_pts > 0 ? "+" : ""}{f.impact_pts}</div>
                </div>
              ))}
            </section>

            {detail.path && (
              <section className="block">
                <h2>Path to approval</h2>
                {detail.path.reachable ? (<>
                  <p className="hint">These changes, applied in order, bring default risk under the approval line. Drag to see the score move.</p>
                  <div className="slider-lbl"><span>Changes applied</span><b>{applied} of {steps.length}</b></div>
                  <input className="slider" type="range" min={0} max={steps.length} value={applied} onChange={(e) => setApplied(+e.target.value)} aria-label="Changes applied" />
                  <ol className="steps">
                    {steps.map((s: any, i: number) => (
                      <li key={s.key} className={`step ${i < applied ? "on" : ""}`}>
                        <span className="n">{i + 1}</span>
                        <div><div>{s.action}</div><div className="chg">{s.from} to {s.to}</div></div>
                        <div className="after">risk {(s.pd_after * 100).toFixed(1)}%</div>
                      </li>
                    ))}
                  </ol>
                </>) : <p className="hint">No realistic combination of changes reaches approval for this request. A much smaller loan or a longer track record would be needed.</p>}
              </section>
            )}

            <section className="block">
              <h2>Credit committee</h2>
              <p className="hint">Three AI reviewers, one for the loan, one against, one for data quality, then a verdict with conditions.</p>
              <button className="btn" onClick={convene} disabled={busy}>{busy ? "Committee is reviewing…" : "Convene committee"}</button>
              {committee && (<>
                <div className="committee">
                  {(["underwriter", "skeptic", "compliance"] as const).map((k) => (
                    <div className="member" key={k}><h3>{k[0].toUpperCase() + k.slice(1)}</h3>{committee[k]}</div>
                  ))}
                </div>
                <div className="final">
                  <b>{DECISION_TEXT[committee.verdict.decision]}.</b> {committee.verdict.summary}
                  {committee.verdict.conditions?.length > 0 && <ul>{committee.verdict.conditions.map((c: string) => <li key={c}>{c}</li>)}</ul>}
                </div>
              </>)}
            </section>
          </>)}
        </main>
      </div>
    </>
  );
}
