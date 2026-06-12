/**
 * src/App.jsx
 * ============
 * Root application component.
 *
 * Layout:
 *   StatusBar  (health polling banner at very top)
 *   Header     (branding + Detect / Chat tab switcher)
 *   Main       (two-column: left = image panel, right = controls/results)
 *
 * Detect tab:  ImageUpload / ImageCanvas  +  QueryInput + DetectionPanel
 * Chat tab:    ImageUpload / ImageCanvas  +  ChatPanel
 */

import React, { useState } from 'react'
import Header        from './components/Header.jsx'
import StatusBar     from './components/StatusBar.jsx'
import ImageUpload   from './components/ImageUpload.jsx'
import ImageCanvas   from './components/ImageCanvas.jsx'
import QueryInput    from './components/QueryInput.jsx'
import DetectionPanel from './components/DetectionPanel.jsx'
import ChatPanel     from './components/ChatPanel.jsx'
import ErrorBanner   from './components/ErrorBanner.jsx'
import { useDetection } from './hooks/useDetection.js'
import { useHealth }    from './hooks/useHealth.js'

export default function App() {
  const [activeTab, setActiveTab] = useState('detect')

  const {
    imageFile, previewUrl, result, chatHistory,
    loading, error,
    selectImage, clearImage,
    runDetect, runChat, clearChat,
  } = useDetection()

  const { health } = useHealth(10_000)
  const modelReady = health?.model_loaded === true

  function handleTabChange(tab) {
    setActiveTab(tab)
  }

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      <StatusBar />
      <Header activeTab={activeTab} onTabChange={handleTabChange} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">

        {/* Model loading notice */}
        {health && !health.model_loaded && health.status !== 'unreachable' && (
          <div className="mb-4 flex items-center gap-3 bg-yellow-900/20 border border-yellow-700/30 rounded-xl px-4 py-3">
            <svg className="w-4 h-4 text-yellow-400 flex-shrink-0 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <p className="text-sm text-yellow-300">
              LocateAnything-3B is loading into memory. This typically takes 1–3 minutes on first start.
              Inference will be available once loading completes.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* ── Left column: Image panel ── */}
          <div className="space-y-4">
            {!previewUrl ? (
              <ImageUpload onSelect={selectImage} disabled={loading} />
            ) : (
              <>
                <ImageCanvas previewUrl={previewUrl} result={result} />
                <div className="flex justify-between items-center px-1">
                  <p className="text-xs text-gray-500 truncate max-w-[60%]">
                    {imageFile?.name}
                    <span className="ml-2 text-gray-600">
                      ({(imageFile?.size / 1024).toFixed(0)} KB)
                    </span>
                  </p>
                  <button
                    onClick={() => { clearImage(); clearChat() }}
                    className="text-xs text-gray-500 hover:text-red-400 transition-colors flex items-center gap-1"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    Change image
                  </button>
                </div>
              </>
            )}

            {/* Error banner lives under image */}
            {error && (
              <ErrorBanner message={error} />
            )}
          </div>

          {/* ── Right column: Controls + Results ── */}
          <div className="space-y-4">
            {activeTab === 'detect' && (
              <>
                <QueryInput
                  onDetect={runDetect}
                  loading={loading}
                  disabled={!previewUrl || !modelReady}
                />
                {result && activeTab === 'detect' && (
                  <DetectionPanel result={result} />
                )}
              </>
            )}

            {activeTab === 'chat' && (
              <ChatPanel
                chatHistory={chatHistory}
                loading={loading}
                onSend={(msg, opts) => runChat(msg, opts)}
                disabled={!previewUrl || !modelReady}
              />
            )}

            {/* Placeholder when no image */}
            {!previewUrl && (
              <div className="card flex flex-col items-center justify-center py-16 text-center">
                <svg className="w-12 h-12 text-gray-700 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <p className="text-gray-500 text-sm">Upload an image to get started</p>
                <p className="text-gray-600 text-xs mt-1">Supports JPEG, PNG, WebP, BMP up to 20 MB</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-3 text-center">
        <p className="text-xs text-gray-600">
          Powered by{' '}
          <a
            href="https://huggingface.co/nvidia/LocateAnything-3B"
            target="_blank"
            rel="noopener noreferrer"
            className="text-nvidia-green hover:underline"
          >
            nvidia/LocateAnything-3B
          </a>
          {' '}· FastAPI · React · Vite
        </p>
      </footer>
    </div>
  )
}
