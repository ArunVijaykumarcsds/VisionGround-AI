/**
 * src/components/Header.jsx
 * ===========================
 * Application header with NVIDIA branding, title, and Detect / Chat tab switcher.
 */

import React from 'react'

export default function Header({ activeTab, onTabChange }) {
  const tabs = [
    { id: 'detect', label: 'Detect', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    )},
    { id: 'chat', label: 'Chat', icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
      </svg>
    )},
  ]

  return (
    <header className="bg-gray-950 border-b border-gray-800 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4 flex-wrap">
        {/* Branding */}
        <div className="flex items-center gap-3">
          {/* NVIDIA green square logo placeholder */}
          <div className="w-8 h-8 rounded bg-nvidia-green flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-black" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.477 2 2 6.477 2 12s4.477 10 10 10 10-4.477 10-10S17.523 2 12 2zm0 2a8 8 0 110 16A8 8 0 0112 4zm-1 3v10l7-5-7-5z"/>
            </svg>
          </div>
          <div>
            <h1 className="text-base font-bold text-gray-100 leading-tight">
              Locate Anything
            </h1>
            <p className="text-xs text-gray-500 leading-tight">
              nvidia/LocateAnything-3B
            </p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 bg-gray-900 rounded-lg p-1 border border-gray-700/50">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-md text-sm font-medium
                transition-all duration-150
                ${activeTab === tab.id
                  ? 'bg-nvidia-green text-black shadow-sm'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}
