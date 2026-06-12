/**
 * src/hooks/useHealth.js
 * =======================
 * Polls GET /health every 10 seconds and exposes model readiness state.
 * Used by the status bar in the header and to gate inference buttons.
 */

import { useState, useEffect, useCallback } from 'react'
import { getHealth } from '../utils/api'

export function useHealth(pollIntervalMs = 10_000) {
  const [health, setHealth] = useState(null)   // null = not yet fetched
  const [checking, setChecking] = useState(true)

  const check = useCallback(async () => {
    try {
      const data = await getHealth()
      setHealth(data)
    } catch {
      setHealth({ status: 'unreachable', model_loaded: false })
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    check()
    const id = setInterval(check, pollIntervalMs)
    return () => clearInterval(id)
  }, [check, pollIntervalMs])

  return { health, checking, refresh: check }
}
