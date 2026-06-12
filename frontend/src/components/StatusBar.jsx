/**
 * src/components/StatusBar.jsx
 * ==============================
 * Top-of-page status bar showing backend / model health.
 * Polls GET /health every 10 s via useHealth hook.
 */

import React from 'react'
import { useHealth } from '../hooks/useHealth'

function Dot({ colour }) {
  const map = {
    green:  'bg-green-400',
    yellow: 'bg-yellow-400',
    red:    'bg-red-400',
    gray:   'bg-gray-500',
  }
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${map[colour] ?? map.gray} animate-pulse`} />
  )
}

export default function StatusBar() {
  const { health, checking } = useHealth(10_000)

  if (checking && !health) {
    return (
      <div className="bg-gray-900 border-b border-gray-800 px-4 py-1.5 flex items-center gap-2">
        <Dot colour="gray" />
        <span className="text-xs text-gray-500">Connecting to backend…</span>
      </div>
    )
  }

  const unreachable = health?.status === 'unreachable'
  const modelLoaded = health?.model_loaded === true

  return (
    <div className="bg-gray-900 border-b border-gray-800 px-4 py-1.5 flex items-center gap-4 flex-wrap text-xs">
      {/* Backend status */}
      <div className="flex items-center gap-1.5">
        <Dot colour={unreachable ? 'red' : 'green'} />
        <span className="text-gray-400">
          Backend: <span className={unreachable ? 'text-red-400' : 'text-green-400'}>
            {unreachable ? 'unreachable' : 'online'}
          </span>
        </span>
      </div>

      {/* Model status */}
      <div className="flex items-center gap-1.5">
        <Dot colour={modelLoaded ? 'green' : unreachable ? 'gray' : 'yellow'} />
        <span className="text-gray-400">
          Model: <span className={
            modelLoaded ? 'text-green-400' :
            unreachable ? 'text-gray-500' :
            'text-yellow-400'
          }>
            {modelLoaded ? 'ready' : unreachable ? 'unknown' : 'loading…'}
          </span>
        </span>
      </div>

      {/* Device */}
      {health?.device && !unreachable && (
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">device:</span>
          <span className="text-gray-300 font-mono">{health.device}</span>
        </div>
      )}

      {/* GPU VRAM */}
      {health?.memory_info?.gpu_vram_allocated_gb !== undefined && (
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">VRAM:</span>
          <span className="text-gray-300 font-mono">
            {health.memory_info.gpu_vram_allocated_gb} /
            {health.memory_info.gpu_vram_total_gb} GB
          </span>
        </div>
      )}

      {/* RAM */}
      {health?.memory_info?.system_ram_used_gb !== undefined && (
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600">RAM:</span>
          <span className="text-gray-300 font-mono">
            {health.memory_info.system_ram_used_gb} /
            {health.memory_info.system_ram_total_gb} GB
          </span>
        </div>
      )}

      {/* Model not loaded warning */}
      {!modelLoaded && !unreachable && (
        <span className="text-yellow-500/80 ml-auto">
          Model is still loading — inference will be available shortly
        </span>
      )}
    </div>
  )
}
