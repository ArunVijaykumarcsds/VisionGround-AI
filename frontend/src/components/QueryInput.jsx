/**
 * src/components/QueryInput.jsx
 * ================================
 * Detection query form with:
 *   - Text input for the natural-language query
 *   - Generation mode selector (hybrid / fast / slow)
 *   - Submit button with loading state
 *   - Example query chips
 */

import React, { useState } from 'react'

const EXAMPLES = [
  'find all cars',
  'locate people',
  'find car, person, bicycle',
  'detect all text',
  'find the traffic light',
  'locate buildings',
]

const MODES = [
  { value: 'hybrid', label: 'Hybrid', desc: 'Balanced speed + accuracy (default)' },
  { value: 'fast',   label: 'Fast',   desc: 'Faster, slightly lower accuracy' },
  { value: 'slow',   label: 'Slow',   desc: 'Slower, higher accuracy' },
]

export default function QueryInput({ onDetect, loading, disabled }) {
  const [query, setQuery]   = useState('')
  const [mode,  setMode]    = useState('hybrid')

  function handleSubmit(e) {
    e.preventDefault()
    if (!query.trim() || loading || disabled) return
    onDetect(query.trim(), { generation_mode: mode })
  }

  return (
    <div className="card space-y-4">
      <h3 className="text-sm font-semibold text-gray-100">Detection Query</h3>

      <form onSubmit={handleSubmit} className="space-y-3">
        {/* Query input */}
        <div>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="find all cars, locate people wearing red…"
            disabled={disabled || loading}
            className="input-field"
          />
        </div>

        {/* Generation mode */}
        <div className="flex gap-2">
          {MODES.map(m => (
            <button
              key={m.value}
              type="button"
              title={m.desc}
              onClick={() => setMode(m.value)}
              disabled={loading}
              className={`flex-1 text-xs py-1.5 rounded-lg border transition-all font-medium
                ${mode === m.value
                  ? 'bg-nvidia-green/10 border-nvidia-green/50 text-nvidia-green'
                  : 'border-gray-600 text-gray-400 hover:border-gray-400 hover:text-gray-200'
                } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={!query.trim() || loading || disabled}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Running inference…
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              Detect
            </>
          )}
        </button>
      </form>

      {/* Example chips */}
      <div>
        <p className="text-xs text-gray-600 mb-2 uppercase tracking-wider font-medium">Examples</p>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.map(ex => (
            <button
              key={ex}
              type="button"
              onClick={() => setQuery(ex)}
              disabled={loading}
              className="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-600/50
                         hover:border-gray-500 text-gray-400 hover:text-gray-200
                         px-2.5 py-1 rounded-full transition-all disabled:opacity-40"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
