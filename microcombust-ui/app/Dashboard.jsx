'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, AlertTriangle, ArrowDown, ArrowUp, ChevronDown, ChevronUp, Database,
  Download, ExternalLink, FileText, Flame, Loader2, Play, Send, ShieldCheck,
  Sparkles, User, Wind, X,
} from 'lucide-react';

/* ------------------------------------------------------------------ */
/* Config                                                              */
/* ------------------------------------------------------------------ */
/* The FastAPI backend binds the first free port (8000 is usually taken by a
   Windows service), so probe a short list instead of hard-coding one. */
const API_CANDIDATES = (
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000,http://localhost:8080'
)
  .split(',')
  .map((base) => base.trim().replace(/\/+$/, ''))
  .filter(Boolean);

// Port that answered on the last successful call, so we skip dead ones after.
let activeApi = null;

const isJson = (res) => String(res.headers.get('content-type') || '').includes('application/json');

/** POST JSON to `path`, trying each configured base until one answers. */
async function apiFetch(path, body) {
  const bases = activeApi
    ? [activeApi, ...API_CANDIDATES.filter((base) => base !== activeApi)]
    : [...API_CANDIDATES];

  let lastError = null;
  let lastResponse = null;

  for (const base of bases) {
    try {
      const res = await fetch(`${base}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (res.ok && isJson(res)) {
        activeApi = base;
        return res;
      }
      lastResponse = res;
    } catch (err) {
      lastError = err;
    }
  }

  // A real backend error is more useful than "nothing answered".
  return lastResponse ?? Promise.reject(lastError ?? new Error('No backend reachable'));
}

const GRAVITY_MODES = [
  { value: '0g', label: '0g', sub: 'Microgravity' },
  { value: '0.16g', label: '0.16g', sub: 'Lunar gravity' },
  { value: '1g', label: '1g', sub: 'Earth gravity' },
];

const FUELS = ['PMMA (Acrylic)', 'Nomex Fabric', 'Silicone Rubber', 'Kevlar Blend'];

// Short material names used by the matrix and table
const MATERIALS = [
  { id: 'PMMA', label: 'PMMA', base: 0.8, fuel: 'PMMA (Acrylic)' },
  { id: 'Silicone', label: 'Silicone', base: 0.75, fuel: 'Silicone Rubber' },
  { id: 'Cotton', label: 'Cotton', base: 0.55, fuel: null },
  { id: 'Nomex', label: 'Nomex', base: 0.45, fuel: 'Nomex Fabric' },
  { id: 'Kevlar', label: 'Kevlar', base: 0.25, fuel: 'Kevlar Blend' },
];

const TIERS = {
  High: {
    title: 'High risk',
    hint: 'Sustained spread likely',
    column: 'border-rose-500/25 bg-linear-to-b from-rose-500/20 to-rose-500/5',
    heading: 'text-rose-200',
    chip: 'border-rose-400/40 bg-rose-500/15 text-rose-200',
    bar: 'bg-linear-to-r from-rose-300 to-rose-500',
    dot: 'bg-rose-400',
  },
  Medium: {
    title: 'Medium risk',
    hint: 'Conditional spread',
    column: 'border-amber-500/25 bg-linear-to-b from-amber-500/20 to-amber-500/5',
    heading: 'text-amber-200',
    chip: 'border-amber-400/40 bg-amber-500/15 text-amber-200',
    bar: 'bg-linear-to-r from-amber-300 to-amber-500',
    dot: 'bg-amber-400',
  },
  Low: {
    title: 'Low risk',
    hint: 'Self-extinguishing in most cases',
    column: 'border-emerald-500/25 bg-linear-to-b from-emerald-500/20 to-emerald-500/5',
    heading: 'text-emerald-200',
    chip: 'border-emerald-400/40 bg-emerald-500/15 text-emerald-200',
    bar: 'bg-linear-to-r from-emerald-300 to-emerald-500',
    dot: 'bg-emerald-400',
  },
};

/* ------------------------------------------------------------------ */
/* Demo data. Replace with metrics extracted into ChromaDB.            */
/* Values below are ILLUSTRATIVE placeholders, not verified NASA data. */
/* ------------------------------------------------------------------ */
const SOFIE = 'https://science.nasa.gov/biological-physical/investigations/sofie/';
const SOFIE_GLENN =
  'https://www.nasa.gov/glenn/glenn-expertise-space-exploration/physical-sciences-program/combustion-science/solid-fuel-ignition-and-extinction-sofie/';
const PSI = 'https://science.nasa.gov/physical-sciences-informatics-psi/';
const NTRS = 'https://ntrs.nasa.gov/';

const SEED_RUNS = [
  { id: 1, fuel: 'PMMA', gravity: '0g', o2: 21, airflow: 5, spread: 0.9, risk: 'High', url: SOFIE },
  { id: 2, fuel: 'PMMA', gravity: '0g', o2: 30, airflow: 10, spread: 2.1, risk: 'High', url: SOFIE_GLENN },
  { id: 3, fuel: 'PMMA', gravity: '0.16g', o2: 21, airflow: 10, spread: 1.4, risk: 'High', url: PSI },
  { id: 4, fuel: 'PMMA', gravity: '1g', o2: 21, airflow: 10, spread: 1.1, risk: 'High', url: NTRS },
  { id: 5, fuel: 'Silicone', gravity: '0g', o2: 21, airflow: 15, spread: 1.2, risk: 'High', url: SOFIE },
  { id: 6, fuel: 'Silicone', gravity: '0.16g', o2: 25, airflow: 20, spread: 1.8, risk: 'High', url: PSI },
  { id: 7, fuel: 'Cotton', gravity: '0g', o2: 21, airflow: 10, spread: 0.6, risk: 'Medium', url: SOFIE_GLENN },
  { id: 8, fuel: 'Cotton', gravity: '1g', o2: 21, airflow: 10, spread: 0.8, risk: 'Medium', url: NTRS },
  { id: 9, fuel: 'Nomex', gravity: '0g', o2: 21, airflow: 20, spread: 0.4, risk: 'Medium', url: SOFIE },
  { id: 10, fuel: 'Nomex', gravity: '0.16g', o2: 30, airflow: 15, spread: 0.7, risk: 'Medium', url: PSI },
  { id: 11, fuel: 'Kevlar', gravity: '0g', o2: 21, airflow: 15, spread: 0.0, risk: 'Low', url: SOFIE },
  { id: 12, fuel: 'Kevlar', gravity: '1g', o2: 30, airflow: 20, spread: 0.2, risk: 'Low', url: NTRS },
];

const WELCOME = {
  id: 'welcome',
  role: 'assistant',
  text:
    'I search NASA Physical Sciences Informatics and NTRS records on microgravity combustion, including BASS, SAFFIRE, FLEX and SoFIE. Ask about flame spread, extinction limits or material flammability. Every answer lists its sources and a confidence score.',
  citations: [],
  confidence: null,
};

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
const tierOf = (s) => (s >= 0.6 ? 'High' : s >= 0.35 ? 'Medium' : 'Low');
const fuelToId = (fuel) => MATERIALS.find((m) => m.fuel === fuel)?.id;

// Offline estimate so the matrix stays usable when the backend is unreachable.
// A rough heuristic for demo purposes only, not a physics model.
function localRank({ oxygen, pressure, airflow, gravity }) {
  const g = gravity === '0g' ? 1.1 : gravity === '0.16g' ? 1.0 : 0.9;
  const fO2 = 0.55 + ((oxygen - 15) / 20) * 0.9;
  const fP = clamp(0.7 + 0.3 * (pressure / 101.3), 0.7, 1.3);
  const fFlow = 0.9 + (airflow / 50) * 0.3;
  return MATERIALS.map((m) => {
    const score = clamp(m.base * fO2 * fP * fFlow * g, 0, 1);
    return { material: m.id, score, tier: tierOf(score) };
  });
}

// Accepts a few reasonable response shapes from /rank-materials
function normalizeRanking(data) {
  const list = Array.isArray(data) ? data : data?.ranking || data?.results || [];
  return list
    .map((r) => {
      const name = String(r.material || r.name || '');
      const id = MATERIALS.find((m) => name.toLowerCase().includes(m.id.toLowerCase()))?.id || name;
      const score = clamp(Number(r.score ?? r.risk_score ?? 0), 0, 1);
      const tier = ['High', 'Medium', 'Low'].find(
        (t) => t.toLowerCase() === String(r.tier || r.risk || r.risk_tier || '').toLowerCase()
      ) || tierOf(score);
      return { material: id, score, tier };
    })
    .filter((r) => r.material);
}

// Accepts { answer, citations: [{title,url,confidence}] } or flat fields
function normalizeAnswer(data) {
  const citations =
    data?.citations ||
    data?.sources ||
    (data?.citation_url ? [{ title: data.citation_title || 'NASA source', url: data.citation_url }] : []);
  const list = citations.map((c) => ({
    title: c.title || c.name || 'NASA source',
    url: c.url || c.link,
    confidence: c.confidence ?? c.score ?? null,
  }));
  const confidence =
    data?.confidence ??
    (list.length && list.every((c) => c.confidence != null)
      ? list.reduce((a, c) => a + c.confidence, 0) / list.length
      : null);
  return {
    text: data?.answer || data?.response || 'No answer returned.',
    citations: list,
    confidence,
    question: data?.question || null,
    // Full ChromaDB passages, so inline citation pills can show the chunk
    // they were drawn from instead of only linking to the document.
    sources: Array.isArray(data?.retrieved_sources) ? data.retrieved_sources : [],
  };
}

const pct = (c) => (c == null ? null : Math.round((c <= 1 ? c * 100 : c)));

/* ------------------------------------------------------------------ */
/* Small UI pieces                                                     */
/* ------------------------------------------------------------------ */
function Panel({ title, icon: Icon, right, children, className = '', delay }) {
  return (
    <section
      style={delay ? { animationDelay: `${delay}ms` } : undefined}
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.04] shadow-[inset_0_1px_0_0_#ffffff0f,0_24px_60px_-30px_#000000f2] backdrop-blur-xl transition duration-300 hover:border-white/[0.16] ${className}`}
    >
      {/* Hairline highlight across the top edge catches the light */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px bg-linear-to-r from-transparent via-cyan-400/70 to-transparent"
      />
      <header className="flex items-center justify-between gap-3 border-b border-white/[0.07] bg-white/[0.03] px-5 py-3.5">
        <h2 className="flex items-center gap-2.5 text-sm font-semibold tracking-tight text-white/90">
          {Icon && (
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/10 text-cyan-300">
              <Icon className="h-4 w-4" aria-hidden />
            </span>
          )}
          {title}
        </h2>
        {right}
      </header>
      {children}
    </section>
  );
}

function Slider({ id, label, unit, value, min, max, step = 1, onChange }) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <div>
      <div className="mb-2.5 flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-[13px] font-medium text-slate-300">
          {label}
        </label>
        <span className="rounded-md border border-white/10 bg-white/[0.06] px-2 py-0.5 font-mono text-sm tabular-nums text-cyan-300">
          {value}
          <span className="ml-1 text-[11px] text-slate-400">{unit}</span>
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ background: `linear-gradient(to right, #22d3ee 0%, #6366f1 ${fill}%, rgb(30 41 59 / 0.85) ${fill}%)` }}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 [&::-moz-range-thumb]:h-4 [&::-moz-range-thumb]:w-4 [&::-moz-range-thumb]:cursor-grab [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-white [&::-moz-range-thumb]:shadow-[0_0_0_3px_#22d3ee59] [&::-webkit-slider-thumb]:mt-[-5px] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:cursor-grab [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-[#05070d] [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_0_3px_#22d3ee59,0_0_14px_#22d3eee6]"
      />
      <div className="mt-1.5 flex justify-between font-mono text-[10px] text-slate-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

function RiskChip({ tier }) {
  const t = TIERS[tier] || TIERS.Low;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider ${t.chip}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${t.dot}`} aria-hidden />
      {tier}
    </span>
  );
}

function Citation({ message }) {
  const [open, setOpen] = useState(false);
  const c = pct(message.confidence);
  const count = message.citations.length;
  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200 transition hover:border-cyan-300/50 hover:bg-cyan-400/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60"
      >
        <ShieldCheck className="h-3.5 w-3.5" aria-hidden />
        {count} {count === 1 ? 'source' : 'sources'}
        {c != null && <span className="font-mono text-cyan-100">{c}% confidence</span>}
        {open ? <ChevronUp className="h-3.5 w-3.5" aria-hidden /> : <ChevronDown className="h-3.5 w-3.5" aria-hidden />}
      </button>
      {open && (
        <ul className="mt-2 space-y-1.5 rounded-xl border border-white/10 bg-white/[0.04] p-2 backdrop-blur-md">
          {count === 0 && <li className="px-1 text-xs text-slate-400">No citations were returned for this answer.</li>}
          {message.citations.map((s, i) => (
            <li key={i}>
              <a
                href={s.url}
                target="_blank"
                rel="noreferrer"
                className="flex items-start justify-between gap-3 rounded-lg px-2.5 py-2 text-xs text-slate-300 transition hover:bg-white/[0.06] hover:text-cyan-200"
              >
                <span className="min-w-0">
                  <span className="block truncate">{s.title}</span>
                  <span className="block truncate font-mono text-[11px] text-slate-500">{s.url}</span>
                </span>
                <span className="flex shrink-0 items-center gap-1.5">
                  {s.confidence != null && <span className="font-mono text-slate-400">{pct(s.confidence)}%</span>}
                  <ExternalLink className="h-3.5 w-3.5" aria-hidden />
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Inline citation pills + source viewer                               */
/* ------------------------------------------------------------------ */
// Matches the markers the prompt asks Gemini to emit, e.g. [19930010989.pdf p.202]
const CITATION_RE = /\[([^\[\]]+?\.pdf)\s+(?:p|page)\.?\s*(\d+)\]/gi;

const fileNameOf = (value) =>
  String(value || '')
    .replace(/\\/g, '/')
    .split('/')
    .pop()
    .toLowerCase();

// Resolve a marker against the passages ChromaDB returned for this answer.
function findSource(sources, filename, page) {
  if (!sources?.length) return null;
  const name = String(filename || '').trim().toLowerCase();
  const matches = sources.filter(
    (s) => fileNameOf(s.filename || s.source) === name
  );
  if (!matches.length) return null;
  return matches.find((s) => String(s.page) === String(page)) || matches[0];
}

// Splits an answer into plain text and clickable citation pills.
function AnswerText({ text, sources, onOpen }) {
  const value = String(text || '');
  const nodes = [];
  let cursor = 0;
  let key = 0;

  for (const match of value.matchAll(CITATION_RE)) {
    const [whole, filename, page] = match;
    const at = match.index ?? 0;
    if (at > cursor) nodes.push(<span key={`t${key++}`}>{value.slice(cursor, at)}</span>);

    const source = findSource(sources, filename, page);
    nodes.push(
      <button
        key={`c${key++}`}
        type="button"
        onClick={() => onOpen(source || { filename, page, snippet: null })}
        title={
          source
            ? 'Show the passage this claim was drawn from'
            : 'No stored passage matched this citation'
        }
        className="mx-0.5 inline-flex max-w-[200px] items-center gap-1 whitespace-nowrap rounded-md border border-cyan-400/35 bg-cyan-400/15 px-1.5 py-0.5 align-baseline font-mono text-[11px] font-medium text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-400/25 hover:text-cyan-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60"
      >
        <FileText className="h-3 w-3 shrink-0" aria-hidden />
        <span className="truncate">{whole.slice(1, -1)}</span>
      </button>
    );
    cursor = at + whole.length;
  }

  if (cursor < value.length) nodes.push(<span key={`t${key++}`}>{value.slice(cursor)}</span>);
  return <>{nodes}</>;
}

function SourceModal({ source, onClose }) {
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  if (!source) return null;

  const filename = source.filename || fileNameOf(source.source) || 'NASA source';
  const snippet = source.snippet || '';
  const confidence = pct(source.confidence);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-label={`Source passage from ${filename}`}
    >
      <div
        onClick={onClose}
        aria-hidden
        className="absolute inset-0 bg-[#05070d]/85 backdrop-blur-sm"
      />
      <div className="relative flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0b1220] shadow-[0_40px_100px_-30px_#000000]">
        <header className="flex items-start justify-between gap-3 border-b border-white/10 bg-white/[0.04] px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-cyan-400/25 bg-cyan-400/10 text-cyan-300">
              <FileText className="h-4 w-4" aria-hidden />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-white">{filename}</p>
              <p className="truncate text-xs text-slate-400">
                Page {source.page ?? 'n/a'}
                {source.source ? `  ·  ${source.source}` : ''}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close source viewer"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.05] text-slate-400 transition hover:bg-white/[0.1] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Source text chunk · ChromaDB
          </p>
          {snippet ? (
            <p className="whitespace-pre-wrap rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm leading-relaxed text-slate-200">
              {snippet}
            </p>
          ) : (
            <p className="rounded-xl border border-amber-400/25 bg-amber-400/10 p-4 text-sm text-amber-100">
              This citation has no stored passage attached, so there is nothing to
              show here. The citations listed under the answer still link to the
              source document.
            </p>
          )}
        </div>

        <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 bg-white/[0.03] px-5 py-3">
          <span className="text-xs text-slate-400">
            Retrieved from the NASA paper index
            {confidence != null && (
              <span className="ml-2 font-mono text-slate-300">{confidence}% confidence</span>
            )}
          </span>
          {source.url && (
            <a
              href={source.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1.5 text-xs font-medium text-cyan-200 transition hover:bg-cyan-400/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60"
            >
              Open full document <ExternalLink className="h-3.5 w-3.5" aria-hidden />
            </a>
          )}
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Export summary report                                               */
/* ------------------------------------------------------------------ */
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Report uses its own palette: it is a printable document, so it is light.
const TIER_HEX = {
  High: { fg: '#be123c', bg: 'rgba(244,63,94,.12)', bar: '#f43f5e' },
  Medium: { fg: '#b45309', bg: 'rgba(245,158,11,.14)', bar: '#f59e0b' },
  Low: { fg: '#047857', bg: 'rgba(16,185,129,.14)', bar: '#10b981' },
};

const reportStamp = (d) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(
    d.getDate()
  ).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(
    d.getMinutes()
  ).padStart(2, '0')}`;

const REPORT_CSS = `
*{box-sizing:border-box}
body{margin:0;background:#eef2f7;color:#0f172a;font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;-webkit-font-smoothing:antialiased}
.sheet{max-width:840px;margin:24px auto;background:#fff;border:1px solid #e2e8f0;border-radius:14px;box-shadow:0 18px 48px -28px rgba(15,23,42,.5);overflow:hidden}
.hero{padding:26px 30px 24px;background:linear-gradient(135deg,#0b1220 0%,#123047 55%,#0e7490 100%);color:#f8fafc}
.brand{font-size:22px;font-weight:800;letter-spacing:.06em}
.sub{margin-top:5px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;opacity:.8}
.meta{margin-top:16px;display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;opacity:.8}
.status{display:inline-block;padding:3px 10px;border-radius:999px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25)}
.print{margin-top:18px;background:#22d3ee;color:#05202b;border:0;border-radius:9px;padding:9px 16px;font-size:13px;font-weight:700;cursor:pointer}
.print:hover{background:#67e8f9}
section{padding:24px 30px;border-top:1px solid #e2e8f0}
h2{margin:0 0 14px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:#0e7490}
h3{margin:20px 0 8px;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#64748b}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid #eef2f7;vertical-align:middle}
thead th{background:#f8fafc;color:#475569;font-size:11px;letter-spacing:.08em;text-transform:uppercase}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
.bar{height:8px;min-width:100px;border-radius:999px;background:#e2e8f0;overflow:hidden}
.bar i{display:block;height:100%;border-radius:999px}
.chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:11px;font-weight:700;white-space:nowrap}
.kv th{width:42%;background:#f8fafc;color:#334155;font-weight:600}
.note{font-size:12px;color:#64748b;margin:0 0 12px}
.q{font-size:13.5px;color:#334155;margin:0 0 10px}
.q b{color:#0f172a}
.answer{white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 18px;font-size:13.5px;line-height:1.7}
ul.src{margin:0;padding-left:18px;font-size:12.5px;color:#334155}
ul.src li{margin:7px 0;word-break:break-word;line-height:1.55}
ul.src a{color:#0e7490}
.empty{color:#94a3b8;font-style:italic;font-size:13.5px}
.foot{padding:18px 30px;background:#f8fafc;border-top:1px solid #e2e8f0;font-size:11.5px;line-height:1.6;color:#64748b}
@media print{
  body{background:#fff}
  .sheet{margin:0;border:0;border-radius:0;box-shadow:none}
  .print{display:none}
  section,.hero,.foot,tr{break-inside:avoid}
}
`;

function buildSummaryReport(opts) {
  const {
    gravity,
    oxygen,
    pressure,
    airflow,
    fuel,
    ranking,
    rankNote,
    analysis,
    apiOnline,
    geminiOn,
    generatedAt,
  } = opts;

  const mode = GRAVITY_MODES.find((g) => g.value === gravity);
  const stamp = generatedAt instanceof Date ? generatedAt : new Date();
  const rows = [...ranking].sort((a, b) => b.score - a.score);

  const settings = [
    ['Gravity mode', mode ? `${mode.label} - ${mode.sub}` : gravity],
    ['Oxygen concentration', `${oxygen} %`],
    ['Ambient pressure', `${pressure} kPa`],
    ['Forced airflow velocity', `${airflow} cm/s`],
    ['Fuel specimen', fuel],
  ]
    .map(
      ([label, value]) =>
        `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`
    )
    .join('');

  const matrix = rows
    .map((r) => {
      const hex = TIER_HEX[r.tier] || TIER_HEX.Low;
      const score = Math.round(r.score * 100);
      return (
        '<tr>' +
        `<td><b>${escapeHtml(r.material)}</b></td>` +
        `<td><span class="chip" style="background:${hex.bg};color:${hex.fg}">${escapeHtml(r.tier)} risk</span></td>` +
        `<td class="num">${score}</td>` +
        `<td><div class="bar"><i style="width:${score}%;background:${hex.bar}"></i></div></td>` +
        '</tr>'
      );
    })
    .join('');

  let analysisBlock;
  if (!analysis) {
    analysisBlock =
      '<p class="empty">No AI analysis has been produced yet. Ask a question in the ' +
      'AI agent query console, then export again.</p>';
  } else if (analysis.error) {
    analysisBlock =
      `<p class="empty">The latest assistant message was an error, so no analysis ` +
      `was captured: ${escapeHtml(analysis.text)}</p>`;
  } else {
    analysisBlock =
      (analysis.question
        ? `<p class="q"><b>Question:</b> ${escapeHtml(analysis.question)}</p>`
        : '') + `<div class="answer">${escapeHtml(analysis.text)}</div>`;
  }

  const sources = analysis?.sources?.length
    ? analysis.sources
    : analysis?.citations || [];
  const sourcesBlock = sources.length
    ? `<ul class="src">${sources
        .map((s) => {
          const bits = [escapeHtml(s.title || s.filename || 'NASA source')];
          if (s.page != null && s.page !== '') bits.push(`p. ${escapeHtml(s.page)}`);
          if (s.confidence != null) bits.push(`${pct(s.confidence)}% confidence`);
          const label = bits.join(' &middot; ');
          return s.url
            ? `<li>${label} - <a href="${escapeHtml(s.url)}" target="_blank" rel="noreferrer">${escapeHtml(s.url)}</a></li>`
            : `<li>${label}</li>`;
        })
        .join('')}</ul>`
    : '<p class="empty">No sources were attached to this analysis.</p>';

  const meta =
    `<span class="status">Gravity ${escapeHtml(gravity)}</span>` +
    `<span class="status">FastAPI ${apiOnline ? 'connected' : 'offline'}</span>` +
    `<span class="status">Gemini ${geminiOn ? 'ready' : 'key not set'}</span>` +
    `<span>Generated ${escapeHtml(stamp.toLocaleString())}</span>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flame in Freefall - Summary Report</title>
<style>${REPORT_CSS}</style>
</head>
<body>
<div class="sheet">
  <header class="hero">
    <div class="brand">FLAME IN FREEFALL</div>
    <div class="sub">NASA Microgravity Combustion &amp; Fire Safety AI Platform</div>
    <div class="meta">${meta}</div>
    <button class="print" type="button" onclick="window.print()">Print / Save as PDF</button>
  </header>

  <section>
    <h2>1 &middot; Experiment settings</h2>
    <table class="kv"><tbody>${settings}</tbody></table>
  </section>

  <section>
    <h2>2 &middot; Material flammability risk matrix</h2>
    <p class="note">${escapeHtml(rankNote)}</p>
    <table>
      <thead>
        <tr>
          <th>Material</th>
          <th>Risk tier</th>
          <th style="text-align:right">Score</th>
          <th>0 - 100</th>
        </tr>
      </thead>
      <tbody>${matrix}</tbody>
    </table>
  </section>

  <section>
    <h2>3 &middot; Latest AI analysis</h2>
    ${analysisBlock}
    <h3>Retrieved sources</h3>
    ${sourcesBlock}
  </section>

  <div class="foot">
    Scores are 0-100 flammability estimates produced by the dashboard ranking
    heuristic (or the backend ranking when FastAPI is connected). They are
    illustrative, not verified NASA data. The flame behavior index table uses
    demo seed values and is not reproduced in this report.
  </div>
</div>
</body>
</html>`;
}

const DEFAULT_RANK_NOTE = 'Live estimate from the sliders. Run a simulation to query the backend.';

/* ------------------------------------------------------------------ */
/* Main component                                                      */
/* ------------------------------------------------------------------ */
export default function Dashboard() {
  // Global + experiment state
  const [gravity, setGravity] = useState('0g');
  const [oxygen, setOxygen] = useState(21);
  const [pressure, setPressure] = useState(101.3);
  const [airflow, setAirflow] = useState(15);
  const [fuel, setFuel] = useState(FUELS[0]);

  // Backend status and ranking
  const [apiOnline, setApiOnline] = useState(true);
  const [geminiOn, setGeminiOn] = useState(false);
  const [running, setRunning] = useState(false);
  const [remoteRank, setRemoteRank] = useState(null);
  const [rankNote, setRankNote] = useState(DEFAULT_RANK_NOTE);

  // Chat
  const [messages, setMessages] = useState([WELCOME]);
  const [draft, setDraft] = useState('');
  const [thinking, setThinking] = useState(false);
  const [openSource, setOpenSource] = useState(null);
  const endRef = useRef(null);

  // Table
  const [showAllGravity, setShowAllGravity] = useState(false);
  const [riskFilter, setRiskFilter] = useState('All');
  const [sort, setSort] = useState({ key: 'spread', dir: 'desc' });

  const params = { oxygen, pressure, airflow, gravity, fuel };
  const paramsKey = `${oxygen}|${pressure}|${airflow}|${gravity}|${fuel}`;

  // Any parameter change invalidates the last backend ranking. Done while
  // rendering rather than in an effect: same outcome, without the extra
  // cascading render flagged by react-hooks/set-state-in-effect.
  const [rankParams, setRankParams] = useState(paramsKey);
  if (rankParams !== paramsKey) {
    setRankParams(paramsKey);
    setRemoteRank(null);
    setRankNote(DEFAULT_RANK_NOTE);
  }

  // The status badge should reflect reality, so probe the backend on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const base of API_CANDIDATES) {
        try {
          const res = await fetch(`${base}/`);
          if (res.ok && isJson(res)) {
            activeApi = base;
            if (!cancelled) {
              setApiOnline(true);
              // Health payload tells us whether a Gemini key was picked up, so
              // the header can say if answers will be synthesised or just
              // retrieved passages.
              try {
                const health = await res.json();
                setGeminiOn(Boolean(health?.gemini_configured));
              } catch {
                /* unexpected body - leave the badge on "unknown" */
              }
            }
            return;
          }
        } catch {
          /* nothing listening on this port - try the next one */
        }
      }
      if (!cancelled) setApiOnline(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, thinking]);

  const ranking = useMemo(
    () => remoteRank || localRank(params),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [remoteRank, oxygen, pressure, airflow, gravity]
  );

  const byTier = useMemo(() => {
    const out = { High: [], Medium: [], Low: [] };
    [...ranking].sort((a, b) => b.score - a.score).forEach((r) => out[r.tier]?.push(r));
    return out;
  }, [ranking]);

  const selectedId = fuelToId(fuel);

  // Latest assistant reply (excluding the welcome banner) captured in reports.
  const lastAnalysis = useMemo(
    () =>
      [...messages]
        .reverse()
        .find((m) => m.role === 'assistant' && m.id !== 'welcome'),
    [messages]
  );

  function exportSummary() {
    const now = new Date();
    const html = buildSummaryReport({
      gravity,
      oxygen,
      pressure,
      airflow,
      fuel,
      ranking,
      rankNote,
      analysis: lastAnalysis,
      apiOnline,
      geminiOn,
      generatedAt: now,
    });
    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `flame-in-freefall-summary-${reportStamp(now)}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Revoke once the download has had a chance to start.
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  async function runSimulation() {
    setRunning(true);
    try {
      const res = await apiFetch('/rank-materials', {
        oxygen_conc: oxygen,
        gravity,
        airflow,
        material: fuel,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const parsed = normalizeRanking(await res.json());
      if (!parsed.length) throw new Error('Empty ranking');
      setRemoteRank(parsed);
      setApiOnline(true);
      setRankNote(`Ranked by the backend at ${gravity}, ${oxygen}% O₂.`);
    } catch {
      setRemoteRank(null);
      setApiOnline(false);
      setRankNote('Backend unreachable. Showing the local estimate instead.');
    } finally {
      setRunning(false);
    }
  }

  async function sendQuery(e) {
    e?.preventDefault();
    const query = draft.trim();
    if (!query || thinking) return;
    setMessages((m) => [...m, { id: `u${Date.now()}`, role: 'user', text: query }]);
    setDraft('');
    setThinking(true);
    try {
      const res = await apiFetch('/query-agent', {
        query,
        gravity_filter: gravity,
        oxygen_conc: oxygen,
        pressure,
        airflow,
        fuel,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const a = normalizeAnswer(await res.json());
      setApiOnline(true);
      setMessages((m) => [...m, { id: `a${Date.now()}`, role: 'assistant', ...a }]);
    } catch {
      setApiOnline(false);
      setMessages((m) => [
        ...m,
        {
          id: `a${Date.now()}`,
          role: 'assistant',
          error: true,
          text: `I could not reach the backend on ${API_CANDIDATES.join(', ')}. Start the FastAPI server and try again.`,
          citations: [],
          confidence: null,
        },
      ]);
    } finally {
      setThinking(false);
    }
  }

  const rows = useMemo(() => {
    let r = SEED_RUNS.filter(
      (x) => (showAllGravity || x.gravity === gravity) && (riskFilter === 'All' || x.risk === riskFilter)
    );
    const dir = sort.dir === 'asc' ? 1 : -1;
    r = [...r].sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      return (typeof av === 'number' ? av - bv : String(av).localeCompare(String(bv))) * dir;
    });
    return r;
  }, [gravity, showAllGravity, riskFilter, sort]);

  const columns = [
    { key: 'fuel', label: 'Fuel material' },
    { key: 'gravity', label: 'Gravity' },
    { key: 'o2', label: 'Oxygen %', num: true },
    { key: 'airflow', label: 'Airflow (cm/s)', num: true },
    { key: 'spread', label: 'Spread rate (mm/s)', num: true },
    { key: 'risk', label: 'Risk' },
    { key: null, label: 'Citation' },
  ];

  const toggleSort = (key) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'desc' }));

  return (
    <div className="relative min-h-screen text-slate-100">
      {/* Ambient backdrop: soft accent glows plus a grid that fades out */}
      <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute -left-40 -top-52 h-[34rem] w-[34rem] rounded-full bg-cyan-500/20 blur-[130px]" />
        <div className="absolute -right-40 top-1/4 h-[38rem] w-[38rem] rounded-full bg-indigo-500/20 blur-[140px]" />
        <div className="absolute -bottom-52 left-1/3 h-[32rem] w-[32rem] rounded-full bg-fuchsia-500/10 blur-[130px]" />
        <div
          className="absolute inset-0"
          style={{
            backgroundImage:
              'linear-gradient(to right, rgba(148,163,184,0.07) 1px, transparent 1px), linear-gradient(to bottom, rgba(148,163,184,0.07) 1px, transparent 1px)',
            backgroundSize: '44px 44px',
            maskImage: 'radial-gradient(ellipse at 50% 0%, black 0%, transparent 70%)',
            WebkitMaskImage: 'radial-gradient(ellipse at 50% 0%, black 0%, transparent 70%)',
          }}
        />
      </div>

      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-white/10 bg-[#05070d]/75 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center justify-between gap-4 px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-linear-to-br from-cyan-300 via-cyan-400 to-indigo-500 shadow-[0_10px_26px_-8px_#22d3ee]">
              <Flame className="h-6 w-6 text-slate-950" aria-hidden />
            </div>
            <div>
              <h1 className="bg-linear-to-r from-white via-cyan-100 to-cyan-300 bg-clip-text text-lg font-bold leading-tight tracking-tight text-transparent">
                FLAME IN FREEFALL
              </h1>
              <p className="text-[11px] font-medium tracking-wide text-slate-400">
                NASA Microgravity Combustion &amp; Fire Safety AI Platform
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-medium ${
                apiOnline
                  ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200'
                  : 'border-amber-400/30 bg-amber-400/10 text-amber-200'
              }`}
            >
              <span className="relative flex h-1.5 w-1.5">
                <span
                  className={`halo absolute inline-flex h-full w-full rounded-full ${
                    apiOnline ? 'bg-emerald-400' : 'bg-amber-400'
                  }`}
                />
                <span
                  className={`relative inline-flex h-1.5 w-1.5 rounded-full ${
                    apiOnline ? 'bg-emerald-300' : 'bg-amber-300'
                  }`}
                />
              </span>
              {apiOnline ? 'FastAPI: Connected' : 'FastAPI: Offline (demo mode)'}
            </span>

            <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-[11px] font-medium text-cyan-200">
              <Database className="h-3 w-3" aria-hidden />
              ChromaDB: Indexed
            </span>

            <span
              title={
                geminiOn
                  ? 'Answers are synthesised by Gemini and grounded in the retrieved papers.'
                  : 'Add GEMINI_API_KEY to .env next to main.py, then restart the backend.'
              }
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[11px] font-medium ${
                geminiOn
                  ? 'border-violet-400/30 bg-violet-400/10 text-violet-200'
                  : 'border-white/10 bg-white/[0.04] text-slate-400'
              }`}
            >
              <Sparkles className="h-3 w-3" aria-hidden />
              {geminiOn ? 'Gemini: Ready' : 'Gemini: add key to .env'}
            </span>

            <button
              type="button"
              onClick={exportSummary}
              title="Download a report of the current settings, risk matrix scores and latest AI analysis"
              className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs font-medium text-slate-200 transition hover:border-cyan-400/40 hover:bg-cyan-400/10 hover:text-cyan-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60"
            >
              <Download className="h-3.5 w-3.5" aria-hidden />
              Export Summary Report
            </button>

            <div role="group" aria-label="Gravity mode" className="flex rounded-full border border-white/10 bg-white/[0.04] p-1">
              {GRAVITY_MODES.map((g) => (
                <button
                  key={g.value}
                  type="button"
                  onClick={() => setGravity(g.value)}
                  aria-pressed={gravity === g.value}
                  className={`rounded-full px-3.5 py-1.5 text-left transition duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 ${
                    gravity === g.value
                      ? 'bg-linear-to-r from-cyan-400 to-indigo-400 font-bold text-slate-950 shadow-[0_6px_18px_-8px_#22d3ee]'
                      : 'text-slate-400 hover:bg-white/[0.06] hover:text-slate-200'
                  }`}
                >
                  <span className="block text-xs">{g.label}</span>
                  <span className="hidden text-[10px] opacity-80 sm:block">{g.sub}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-[1500px] space-y-5 px-4 py-6 sm:px-6 sm:py-8">
        <div className="grid gap-5 lg:grid-cols-12">
          {/* Left column */}
          <div className="space-y-5 lg:col-span-5">
            <Panel title="Experiment controls" icon={Wind} delay={0}>
              <div className="space-y-6 p-5">
                <Slider id="o2" label="Oxygen concentration" unit="%" value={oxygen} min={15} max={35} onChange={setOxygen} />
                <Slider id="p" label="Ambient pressure" unit="kPa" value={pressure} min={50} max={150} step={0.1} onChange={setPressure} />
                <Slider id="v" label="Forced airflow velocity" unit="cm/s" value={airflow} min={0} max={50} onChange={setAirflow} />

                <div>
                  <label htmlFor="fuel" className="mb-2 block text-[13px] font-medium text-slate-300">
                    Fuel specimen
                  </label>
                  <select
                    id="fuel"
                    value={fuel}
                    onChange={(e) => setFuel(e.target.value)}
                    className="w-full rounded-xl border border-white/10 bg-white/[0.05] px-3 py-2.5 text-sm text-slate-100 transition hover:border-white/20 focus:border-cyan-400/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40"
                  >
                    {FUELS.map((f) => (
                      <option key={f} className="bg-slate-950 text-slate-100">
                        {f}
                      </option>
                    ))}
                  </select>
                </div>

                <button
                  type="button"
                  onClick={runSimulation}
                  disabled={running}
                  className="flex w-full items-center justify-center gap-2 rounded-xl bg-linear-to-r from-cyan-400 to-indigo-400 px-4 py-3 text-sm font-bold tracking-wide text-slate-950 shadow-[0_14px_34px_-14px_#22d3ee] transition duration-200 hover:-translate-y-0.5 hover:shadow-[0_18px_40px_-14px_#818cf8] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:ring-offset-2 focus-visible:ring-offset-[#05070d] disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:translate-y-0"
                >
                  {running ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
                  {running ? 'Ranking…' : 'RUN SIMULATION & RANK RISK'}
                </button>
              </div>
            </Panel>

            <Panel
              title="Material flammability risk matrix"
              icon={AlertTriangle}
              delay={60}
              right={
                <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 font-mono text-[11px] text-slate-400">
                  {gravity}
                </span>
              }
            >
              <div className="p-5">
                <div className="grid gap-3 sm:grid-cols-3">
                  {['High', 'Medium', 'Low'].map((tier) => {
                    const t = TIERS[tier];
                    return (
                      <div key={tier} className={`rounded-xl border p-3.5 backdrop-blur-sm transition ${t.column}`}>
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <h3 className={`text-[13px] font-semibold ${t.heading}`}>{t.title}</h3>
                          <span
                            className={`rounded-full border px-1.5 py-0.5 text-[10px] font-bold tabular-nums ${t.chip}`}
                          >
                            {byTier[tier].length}
                          </span>
                        </div>
                        <p className="mb-3 text-[11px] leading-snug text-slate-400">{t.hint}</p>
                        <ul className="space-y-2">
                          {byTier[tier].length === 0 && (
                            <li className="text-xs text-slate-500">None at these settings</li>
                          )}
                          {byTier[tier].map((r) => (
                            <li
                              key={r.material}
                              className={`rounded-lg border px-2.5 py-2 transition ${
                                r.material === selectedId
                                  ? 'border-cyan-400/50 bg-slate-950/70 shadow-[0_0_16px_-6px_#22d3ee]'
                                  : 'border-white/[0.08] bg-slate-950/40 hover:border-white/[0.16]'
                              }`}
                            >
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-medium text-slate-200">{r.material}</span>
                                <span className="font-mono font-semibold tabular-nums text-slate-300">
                                  {Math.round(r.score * 100)}
                                </span>
                              </div>
                              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.08]">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${t.bar}`}
                                  style={{ width: `${r.score * 100}%` }}
                                />
                              </div>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-3.5 flex items-start gap-1.5 text-xs leading-relaxed text-slate-400">
                  <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan-400" aria-hidden />
                  {rankNote}
                </p>
              </div>
            </Panel>
          </div>

          {/* Right column: chat */}
          <Panel
            title="AI agent query console"
            icon={Sparkles}
            delay={120}
            className="flex min-h-[560px] flex-col lg:col-span-7"
            right={
              <span className="rounded-full border border-white/10 bg-white/[0.05] px-2.5 py-1 text-[11px] text-slate-400">
                Searching {gravity} records
              </span>
            }
          >
            <div className="flex-1 space-y-4 overflow-y-auto p-5 lg:max-h-[640px]" aria-live="polite">
              {messages.map((m) => (
                <div key={m.id} className={`flex gap-3 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <span
                    className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border ${
                      m.role === 'user'
                        ? 'border-cyan-400/30 bg-cyan-400/15 text-cyan-200'
                        : 'border-indigo-400/30 bg-indigo-400/15 text-indigo-200'
                    }`}
                  >
                    {m.role === 'user' ? (
                      <User className="h-4 w-4" aria-hidden />
                    ) : (
                      <Sparkles className="h-4 w-4" aria-hidden />
                    )}
                  </span>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      m.role === 'user'
                        ? 'rounded-br-md border border-cyan-400/25 bg-linear-to-br from-cyan-500/25 to-indigo-500/20 text-cyan-50'
                        : m.error
                        ? 'rounded-bl-md border border-amber-400/30 bg-amber-400/10 text-amber-100'
                        : 'rounded-bl-md border border-white/10 bg-white/[0.05] text-slate-200 shadow-[inset_0_1px_0_0_#ffffff0f]'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">
                      {m.role === 'assistant' ? (
                        <AnswerText text={m.text} sources={m.sources} onOpen={setOpenSource} />
                      ) : (
                        m.text
                      )}
                    </p>
                    {m.role === 'assistant' && m.id !== 'welcome' && !m.error && <Citation message={m} />}
                  </div>
                </div>
              ))}
              {thinking && (
                <div className="flex items-center gap-2.5 self-start rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs text-slate-400">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-300" aria-hidden /> Searching NASA records…
                </div>
              )}
              <div ref={endRef} />
            </div>

            <form onSubmit={sendQuery} className="flex gap-2 border-t border-white/[0.07] bg-white/[0.02] p-4">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="Ask about flame spread, extinction limits or materials"
                aria-label="Ask the agent"
                className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/[0.05] px-4 py-2.5 text-sm text-slate-100 transition placeholder:text-slate-500 focus:border-cyan-400/50 focus:bg-white/[0.07] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40"
              />
              <button
                type="submit"
                disabled={!draft.trim() || thinking}
                className="flex shrink-0 items-center gap-2 rounded-xl bg-linear-to-r from-cyan-400 to-indigo-400 px-4 py-2.5 text-sm font-bold text-slate-950 shadow-[0_10px_26px_-14px_#22d3ee] transition hover:-translate-y-0.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0"
              >
                <Send className="h-4 w-4" aria-hidden />
                Send
              </button>
            </form>
          </Panel>
        </div>

        {/* Flame behavior index */}
        <Panel
          title="Flame behavior index"
          icon={Activity}
          delay={180}
          right={
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-white/10 bg-white/[0.05] px-2.5 py-1.5 text-slate-300 transition hover:border-white/20">
                <input
                  type="checkbox"
                  checked={showAllGravity}
                  onChange={(e) => setShowAllGravity(e.target.checked)}
                  className="h-3.5 w-3.5 accent-cyan-400"
                />
                All gravity levels
              </label>
              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                aria-label="Filter by risk"
                className="rounded-lg border border-white/10 bg-white/[0.05] px-2.5 py-1.5 text-xs text-slate-200 transition hover:border-white/20 focus:border-cyan-400/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/40"
              >
                {['All', 'High', 'Medium', 'Low'].map((r) => (
                  <option key={r} className="bg-slate-950">
                    {r}
                  </option>
                ))}
              </select>
            </div>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead>
                <tr className="border-b border-white/10 bg-white/[0.03] text-[11px] uppercase tracking-wider text-slate-400">
                  {columns.map((c) => (
                    <th
                      key={c.label}
                      scope="col"
                      className={`px-4 py-3 font-semibold ${c.num ? 'text-right' : ''}`}
                    >
                      {c.key ? (
                        <button
                          type="button"
                          onClick={() => toggleSort(c.key)}
                          className="inline-flex items-center gap-1 transition hover:text-cyan-300 focus:outline-none focus-visible:text-cyan-300"
                        >
                          {c.label}
                          {sort.key === c.key &&
                            (sort.dir === 'asc' ? (
                              <ArrowUp className="h-3 w-3" aria-hidden />
                            ) : (
                              <ArrowDown className="h-3 w-3" aria-hidden />
                            ))}
                        </button>
                      ) : (
                        c.label
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-sm text-slate-500">
                      No runs match these filters. Try “All gravity levels”.
                    </td>
                  </tr>
                )}
                {rows.map((r) => (
                  <tr
                    key={r.id}
                    className={`border-b border-white/[0.06] transition last:border-0 hover:bg-cyan-400/[0.07] ${
                      r.fuel === selectedId
                        ? 'bg-cyan-400/[0.09]'
                        : r.id % 2 === 0
                        ? 'bg-white/[0.02]'
                        : ''
                    }`}
                  >
                    <td className="px-4 py-3 font-medium text-slate-100">{r.fuel}</td>
                    <td className="px-4 py-3 font-mono text-slate-300">{r.gravity}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">{r.o2}</td>
                    <td className="px-4 py-3 text-right font-mono tabular-nums text-slate-300">{r.airflow}</td>
                    <td className="px-4 py-3 text-right font-mono font-semibold tabular-nums text-cyan-300">
                      {r.spread.toFixed(1)}
                    </td>
                    <td className="px-4 py-3">
                      <RiskChip tier={r.risk} />
                    </td>
                    <td className="px-4 py-3">
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium text-cyan-300 transition hover:bg-cyan-400/10 hover:text-cyan-200"
                      >
                        Source <ExternalLink className="h-3 w-3" aria-hidden />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-white/[0.07] px-5 py-3 text-[11px] leading-relaxed text-slate-500">
            Demo seed data. Spread rates are placeholders until they are replaced with metrics extracted from the NASA reports.
          </p>
        </Panel>
      </main>

      {openSource && (
        <SourceModal source={openSource} onClose={() => setOpenSource(null)} />
      )}
    </div>
  );
}
