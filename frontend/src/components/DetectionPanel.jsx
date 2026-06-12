/**
 * src/components/DetectionPanel.jsx
 * ====================================
 * Renders the structured detection results:
 * - Summary header (count, timing)
 * - Scrollable list of bounding boxes with labels and coordinates
 * - Points list
 * - Raw model output (collapsible)
 */

import React, { useState } from 'react'

const PALETTE = [
  '#76b900','#3b82f6','#f59e0b','#ef4444',
  '#8b5cf6','#06b6d4','#f97316','#10b981',
  '#ec4899','#6366f1','#84cc16','#14b8a6',
]

function ColourDot({ idx }) {
  return (
    <span
      className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
      style={{ backgroundColor: PALETTE[idx % PALETTE.length] }}
    />
  )
}

export default function DetectionPanel({ result }) {
  const [showRaw, setShowRaw] = useState(false)

  if (!result) return null

  const { query, detections = [], points = [], raw_answer, inference_time_ms, generation_mode_used } = result
  const total = detections.length + points.length

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-gray-100">Detection Results</h3>
          <p className="text-xs text-gray-400 mt-0.5 line-clamp-1">Query: <span className="text-gray-300">{query}</span></p>
        </div>
        <div className="text-right flex-shrink-0">
          <span className="text-xs font-mono text-nvidia-green">{inference_time_ms?.toFixed(0)} ms</span>
          <p className="text-xs text-gray-500">{generation_mode_used}</p>
        </div>
      </div>

      {/* Summary pill */}
      <div className="flex gap-2 flex-wrap">
        {detections.length > 0 && (
          <span className="text-xs bg-nvidia-green/10 text-nvidia-green border border-nvidia-green/20 px-2.5 py-1 rounded-full font-medium">
            {detections.length} bounding box{detections.length !== 1 ? 'es' : ''}
          </span>
        )}
        {points.length > 0 && (
          <span className="text-xs bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2.5 py-1 rounded-full font-medium">
            {points.length} point{points.length !== 1 ? 's' : ''}
          </span>
        )}
        {total === 0 && (
          <span className="text-xs bg-gray-700/50 text-gray-400 border border-gray-600/30 px-2.5 py-1 rounded-full">
            No detections found
          </span>
        )}
      </div>

      {/* Bounding boxes list */}
      {detections.length > 0 && (
        <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
          {detections.map((det, idx) => (
            <div
              key={idx}
              className="flex items-center gap-2.5 bg-gray-800/60 rounded-lg px-3 py-2"
            >
              <ColourDot idx={idx} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-200 truncate">{det.label}</p>
                <p className="text-xs text-gray-500 font-mono">
                  [{det.bbox.map(v => Math.round(v)).join(', ')}]
                </p>
              </div>
              <span className="text-xs font-mono text-gray-400 flex-shrink-0">
                {(det.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Points list */}
      {points.length > 0 && (
        <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
          <p className="text-xs text-gray-500 uppercase tracking-wider font-medium">Points</p>
          {points.map((pt, idx) => (
            <div key={idx} className="flex items-center gap-2.5 bg-gray-800/60 rounded-lg px-3 py-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-400 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-200 truncate">{pt.label}</p>
                <p className="text-xs text-gray-500 font-mono">
                  x={Math.round(pt.x)}, y={Math.round(pt.y)}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Raw output (collapsible) */}
      <div className="border-t border-gray-700/50 pt-3">
        <button
          onClick={() => setShowRaw(v => !v)}
          className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
        >
          <svg
            className={`w-3 h-3 transition-transform ${showRaw ? 'rotate-90' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Raw model output
        </button>
        {showRaw && (
          <pre className="mt-2 text-xs text-gray-400 bg-gray-950 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap break-words font-mono max-h-40 overflow-y-auto">
            {raw_answer || '(empty)'}
          </pre>
        )}
      </div>
    </div>
  )
}
