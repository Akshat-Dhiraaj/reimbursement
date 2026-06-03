import { useCallback, useEffect, useRef, useState } from 'react'

// Map the pipeline's triage decision to a binary "approved / not approved" headline plus the
// precise sub-state (review vs reject) and a colour. approve = green, review = amber, reject = red.
const DECISION = {
  approve: { headline: 'Approved', detail: 'Looks valid for reimbursement.', tone: 'ok' },
  review: { headline: 'Not approved', detail: 'Needs a human review.', tone: 'warn' },
  reject: { headline: 'Not approved', detail: 'Rejected — a concrete problem was found.', tone: 'bad' },
}

const FIELD_LABELS = {
  vendor: 'Vendor', date: 'Date', currency: 'Currency', subtotal: 'Subtotal', tax: 'Tax',
  service_charge: 'Service', discount: 'Discount', total: 'Total', tax_id: 'Tax ID', country: 'Country',
}

const CHECK_LABELS = {
  arithmetic_consistent: 'Arithmetic',
  date_valid: 'Date',
  ai_or_edit_suspected: 'AI / edit',
}

function fmt(v) {
  if (v === null || v === undefined || v === '') return '—'
  return String(v)
}

// A check is "good" when true — except ai_or_edit_suspected, where true is the bad outcome.
function checkTone(key, value) {
  if (value === null || value === undefined) return 'muted'
  const good = key === 'ai_or_edit_suspected' ? value === false : value === true
  return good ? 'ok' : 'bad'
}

function checkText(key, value) {
  if (value === null || value === undefined) return 'n/a'
  if (key === 'ai_or_edit_suspected') return value ? 'suspected' : 'clean'
  return value ? 'consistent' : 'inconsistent'
}

export default function App() {
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [fileName, setFileName] = useState('')
  const [dragging, setDragging] = useState(false)
  const [health, setHealth] = useState(null)
  const inputRef = useRef(null)

  // On load, tell the user up front whether the server has a usable API key.
  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  const submit = useCallback(async (file) => {
    if (!file) return
    setError('')
    setResult(null)
    setFileName(file.name)
    setStatus('loading')
    try {
      const body = new FormData()
      body.append('file', file)
      const res = await fetch('/api/validate', { method: 'POST', body })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Request failed (HTTP ${res.status})`)
      setResult(data)
      setStatus('done')
    } catch (e) {
      setError(e.message || String(e))
      setStatus('error')
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) submit(e.dataTransfer.files[0])
  }, [submit])

  const reset = () => {
    setStatus('idle')
    setResult(null)
    setError('')
    setFileName('')
  }

  return (
    <div className="page">
      <header>
        <h1>slipguard</h1>
        <p className="tagline">Drop a reimbursement receipt or invoice — get an instant validity verdict.</p>
      </header>

      {health && !health.ok && (
        <div className="banner bad">
          No API key configured on the server. Set <code>GROQ_API_KEY</code> or{' '}
          <code>GEMINI_API_KEY</code> (e.g. in <code>.env</code>) and restart. {health.detail || ''}
        </div>
      )}

      <div
        className={`dropzone${dragging ? ' dragging' : ''}${status === 'loading' ? ' busy' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => status !== 'loading' && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.gif,.pdf,image/*,application/pdf"
          hidden
          onChange={(e) => submit(e.target.files?.[0])}
        />
        {status === 'loading' ? (
          <div className="dz-inner">
            <div className="spinner" />
            <p>Analysing <strong>{fileName}</strong>…</p>
          </div>
        ) : (
          <div className="dz-inner">
            <div className="dz-icon">⬆</div>
            <p><strong>Drag &amp; drop</strong> a receipt here, or <span className="link">browse</span></p>
            <p className="hint">JPG · PNG · WEBP · GIF · PDF</p>
          </div>
        )}
      </div>

      {status === 'error' && (
        <div className="banner bad">
          <strong>Couldn’t validate.</strong> {error}
          <button className="ghost" onClick={reset}>Try again</button>
        </div>
      )}

      {status === 'done' && result && <Verdict result={result} onReset={reset} />}

      <footer>
        Validity = an LLM judge (Groq / Gemini) <em>cross-checked</em> by deterministic
        arithmetic, date and tax-id rules. A clean result is triage, not forensic proof.
      </footer>
    </div>
  )
}

function Verdict({ result, onReset }) {
  const meta = DECISION[result.decision] || DECISION.review
  const conf = typeof result.confidence === 'number'
    ? `${Math.round(result.confidence * 100)}%` : null
  const fields = result.fields || {}
  const checks = result.checks || {}

  return (
    <section className="result">
      <div className={`status ${meta.tone}`}>
        <div className="status-main">
          <span className="status-headline">{meta.headline}</span>
          <span className="status-detail">{meta.detail}</span>
        </div>
        <div className="status-side">
          <span className={`pill ${meta.tone}`}>{result.decision}</span>
          {conf && <span className="conf">confidence {conf}</span>}
        </div>
      </div>

      {result.summary && <p className="summary">{result.summary}</p>}

      <h3>Why</h3>
      {result.reasons && result.reasons.length > 0 ? (
        <ul className="reasons">
          {result.reasons.map((r, i) => (
            <li key={i}>
              <span className={`tag ${r.source}`}>
                {r.source === 'deterministic' ? 'cross-check' : 'model'}
              </span>
              {r.text}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No specific reasons returned.</p>
      )}

      {result.ai_or_edit_signs && result.ai_or_edit_signs.length > 0 && (
        <>
          <h3>Possible edit cues <span className="muted">(triage, not proof)</span></h3>
          <ul className="reasons">
            {result.ai_or_edit_signs.map((s, i) => <li key={i}><span className="tag model">model</span>{s}</li>)}
          </ul>
        </>
      )}

      <h3>Extracted fields</h3>
      <div className="fields">
        {Object.keys(FIELD_LABELS).map((k) => (
          <div className="field" key={k}>
            <span className="field-label">{FIELD_LABELS[k]}</span>
            <span className="field-value">{fmt(fields[k])}</span>
          </div>
        ))}
      </div>

      <h3>Checks</h3>
      <div className="checks">
        {Object.keys(CHECK_LABELS).map((k) => (
          <span className={`check ${checkTone(k, checks[k])}`} key={k}>
            {CHECK_LABELS[k]}: {checkText(k, checks[k])}
          </span>
        ))}
      </div>

      <div className="decisions muted">
        Model said <strong>{result.llm_decision}</strong>
        {result.deterministic_decision &&
          <> · deterministic cross-check said <strong>{result.deterministic_decision}</strong> · final is the stricter</>}
        {result.provider && <> · via {result.provider}</>}
      </div>

      <button className="primary" onClick={onReset}>Validate another</button>
    </section>
  )
}
