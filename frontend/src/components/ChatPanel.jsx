/**
 * src/components/ChatPanel.jsx
 * ==============================
 * Conversational chat interface for the /chat endpoint.
 * Displays message history with timestamps, detection summaries,
 * and an auto-scrolling message feed.
 */

import React, { useRef, useEffect, useState } from 'react'

function Timestamp({ ts }) {
  const d = new Date(ts)
  return (
    <span className="text-xs text-gray-600">
      {d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
    </span>
  )
}

function UserBubble({ msg }) {
  return (
    <div className="flex justify-end gap-2">
      <div className="max-w-[80%]">
        <div className="bg-nvidia-green/20 border border-nvidia-green/25 text-gray-100 text-sm rounded-2xl rounded-tr-sm px-4 py-2.5">
          {msg.content}
        </div>
        <div className="flex justify-end mt-1 pr-1">
          <Timestamp ts={msg.ts} />
        </div>
      </div>
    </div>
  )
}

function AssistantBubble({ msg }) {
  const hasDetections = msg.detections?.length > 0
  const hasPoints = msg.points?.length > 0

  return (
    <div className="flex gap-2">
      {/* Avatar */}
      <div className="w-7 h-7 rounded-full bg-nvidia-green/20 border border-nvidia-green/30 flex items-center justify-center flex-shrink-0 mt-0.5">
        <svg className="w-3.5 h-3.5 text-nvidia-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
          />
        </svg>
      </div>

      <div className="max-w-[85%] space-y-1.5">
        <div className={`text-sm rounded-2xl rounded-tl-sm px-4 py-2.5 ${
          msg.isError
            ? 'bg-red-900/30 border border-red-700/30 text-red-300'
            : 'bg-gray-800 border border-gray-700/50 text-gray-100'
        }`}>
          {msg.content}
        </div>

        {/* Detection summary pills */}
        {(hasDetections || hasPoints) && (
          <div className="flex gap-1.5 flex-wrap px-1">
            {hasDetections && (
              <span className="text-xs bg-nvidia-green/10 text-nvidia-green border border-nvidia-green/20 px-2 py-0.5 rounded-full">
                {msg.detections.length} box{msg.detections.length !== 1 ? 'es' : ''}
              </span>
            )}
            {hasPoints && (
              <span className="text-xs bg-blue-500/10 text-blue-400 border border-blue-500/20 px-2 py-0.5 rounded-full">
                {msg.points.length} pt{msg.points.length !== 1 ? 's' : ''}
              </span>
            )}
            {msg.inference_time_ms && (
              <span className="text-xs text-gray-600 font-mono">
                {msg.inference_time_ms.toFixed(0)} ms
              </span>
            )}
          </div>
        )}

        <div className="pl-1">
          <Timestamp ts={msg.ts} />
        </div>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-2 items-center">
      <div className="w-7 h-7 rounded-full bg-nvidia-green/20 border border-nvidia-green/30 flex items-center justify-center flex-shrink-0">
        <svg className="w-3.5 h-3.5 text-nvidia-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
          />
        </svg>
      </div>
      <div className="bg-gray-800 border border-gray-700/50 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1">
          {[0,1,2].map(i => (
            <div
              key={i}
              className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

export default function ChatPanel({ chatHistory, loading, onSend, disabled }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom on new messages.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, loading])

  function handleSubmit(e) {
    e.preventDefault()
    if (!input.trim() || loading || disabled) return
    onSend(input.trim())
    setInput('')
    inputRef.current?.focus()
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="card flex flex-col h-[520px]">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-gray-700/50 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-nvidia-green animate-pulse" />
          <h3 className="text-sm font-semibold text-gray-100">Visual Chat</h3>
        </div>
        <span className="text-xs text-gray-500">{chatHistory.length} message{chatHistory.length !== 1 ? 's' : ''}</span>
      </div>

      {/* Messages feed */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 min-h-0">
        {chatHistory.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-12 h-12 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
            <p className="text-sm text-gray-500">Upload an image and start chatting</p>
            <p className="text-xs text-gray-600 mt-1">e.g. "Locate all cars" or "Find the person on the left"</p>
          </div>
        )}

        {chatHistory.map((msg, i) =>
          msg.role === 'user'
            ? <UserBubble key={i} msg={msg} />
            : <AssistantBubble key={i} msg={msg} />
        )}

        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="mt-3 pt-3 border-t border-gray-700/50">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={disabled ? 'Upload an image first…' : 'Locate all buildings…'}
            disabled={disabled || loading}
            className="input-field flex-1 text-sm py-2"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading || disabled}
            className="btn-primary px-4 py-2 flex-shrink-0"
          >
            {loading ? (
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            )}
          </button>
        </div>
        <p className="text-xs text-gray-600 mt-1.5 pl-1">Enter to send · Shift+Enter for newline</p>
      </form>
    </div>
  )
}
