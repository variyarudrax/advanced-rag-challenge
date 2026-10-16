import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

function Sources({ sources }) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(null)
  if (!sources?.length) return null
  const pages = [...new Set(sources.map((s) => s.page))]

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen((o) => !o)}>
        {open ? '▾' : '▸'} Sources · pages {pages.join(', ')}
      </button>
      {open && (
        <ul className="sources-list">
          {sources.map((s, i) => (
            <li key={i} className="source-item">
              <button className="source-head" onClick={() => setActive(active === i ? null : i)}>
                <span className="page-tag">p. {s.page}</span>
                <span className="source-preview">{s.text.slice(0, 90)}…</span>
                <span className="score">{s.score.toFixed(2)}</span>
              </button>
              {active === i && <p className="source-text">{s.text}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function App() {
  const [status, setStatus] = useState(null)
  const [statusError, setStatusError] = useState(null)
  const [samples, setSamples] = useState([])

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [asking, setAsking] = useState(false)

  const bottomRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatusError('Cannot reach the backend at ' + API_BASE))
    fetch(`${API_BASE}/sample-questions`)
      .then((r) => r.json())
      .then((data) => setSamples(data.questions || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  async function sendQuestion(text) {
    const question = (text ?? input).trim()
    if (!question || asking) return
    setInput('')
    setMessages((m) => [...m, { role: 'user', text: question }])
    setAsking(true)
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setMessages((m) => [
        ...m,
        { role: 'bot', text: data.answer, sources: data.sources, notFound: !data.found },
      ])
    } catch (err) {
      setMessages((m) => [...m, { role: 'bot', text: `Error: ${err.message}`, error: true }])
    } finally {
      setAsking(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion()
    }
  }

  const ready = status?.ready

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          GPO Annual Report Assistant
        </div>
        {ready && (
          <div className="doc-pill" title={status.document}>
            <span className="doc-pill-icon">📄</span>
            <span className="doc-pill-name">{status.document}</span>
            <span className="doc-pill-meta">{status.chunks} chunks</span>
          </div>
        )}
      </header>

      {statusError || (status && !ready) ? (
        <main className="notice-stage">
          <div className="notice">
            <p className="notice-title">
              {statusError
                ? 'Backend unavailable'
                : status.chunks > 0
                  ? `Index incomplete (${status.chunks}/${status.total_chunks ?? '?'} chunks)`
                  : 'Index not built yet'}
            </p>
            <p className="notice-sub">
              {statusError ? (
                <>Start it with <code>uvicorn app.main:app --reload --port 8000</code> in <code>backend/</code>.</>
              ) : (
                <>Run <code>python -m app.ingest</code> in <code>backend/</code> (it resumes), then refresh.</>
              )}
            </p>
          </div>
        </main>
      ) : (
        <main className="chat-stage">
          <div className="chat-log">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Ask anything about the <strong>GPO Annual Report</strong>.</p>
                <p className="empty-sub">Answers come only from the report, with page citations.</p>
                {samples.length > 0 && (
                  <div className="samples">
                    {samples.map((q) => (
                      <button key={q} className="sample-chip" onClick={() => sendQuestion(q)} disabled={!ready}>
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`bubble-row ${m.role}`}>
                <div className={`bubble ${m.role} ${m.error ? 'error' : ''} ${m.notFound ? 'not-found' : ''}`}>
                  {m.text}
                  {m.role === 'bot' && <Sources sources={m.sources} />}
                </div>
              </div>
            ))}
            {asking && (
              <div className="bubble-row bot">
                <div className="bubble bot typing">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question about the GPO Annual Report…"
              rows={1}
              disabled={!ready}
            />
            <button onClick={() => sendQuestion()} disabled={!ready || asking || !input.trim()}>
              Send
            </button>
          </div>
        </main>
      )}
    </div>
  )
}

export default App
