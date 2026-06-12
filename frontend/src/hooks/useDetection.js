/**
 * src/hooks/useDetection.js
 * ==========================
 * Custom React hook that manages detection and chat state.
 *
 * Handles:
 *   - Image file selection and preview URL
 *   - Detect / chat API calls with loading and error state
 *   - Detection result storage
 *   - Chat message history
 */

import { useState, useCallback, useRef } from 'react'
import { detect, chat, extractError } from '../utils/api'

/**
 * @typedef {Object} BoundingBox
 * @property {string} label
 * @property {number} confidence
 * @property {number[]} bbox       [x1, y1, x2, y2] pixels
 * @property {number[]} bbox_normalised  [x1, y1, x2, y2] in [0,1]
 */

/**
 * @typedef {Object} DetectResult
 * @property {string}        query
 * @property {BoundingBox[]} detections
 * @property {object[]}      points
 * @property {string}        raw_answer
 * @property {number}        image_width
 * @property {number}        image_height
 * @property {number}        inference_time_ms
 */

export function useDetection() {
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [chatHistory, setChatHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const prevPreviewUrl = useRef(null)

  // -----------------------------------------------------------------------
  // Image selection
  // -----------------------------------------------------------------------

  const selectImage = useCallback((file) => {
    if (!file) return

    // Revoke previous object URL to avoid memory leaks.
    if (prevPreviewUrl.current) {
      URL.revokeObjectURL(prevPreviewUrl.current)
    }

    const url = URL.createObjectURL(file)
    prevPreviewUrl.current = url
    setImageFile(file)
    setPreviewUrl(url)
    setResult(null)
    setError(null)
  }, [])

  const clearImage = useCallback(() => {
    if (prevPreviewUrl.current) {
      URL.revokeObjectURL(prevPreviewUrl.current)
      prevPreviewUrl.current = null
    }
    setImageFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
  }, [])

  // -----------------------------------------------------------------------
  // Detect
  // -----------------------------------------------------------------------

  const runDetect = useCallback(async (query, opts = {}) => {
    if (!imageFile) {
      setError('Please upload an image first.')
      return
    }
    if (!query.trim()) {
      setError('Please enter a detection query.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await detect(imageFile, query.trim(), opts)
      setResult(data)
    } catch (err) {
      setError(extractError(err))
    } finally {
      setLoading(false)
    }
  }, [imageFile])

  // -----------------------------------------------------------------------
  // Chat
  // -----------------------------------------------------------------------

  const runChat = useCallback(async (message, opts = {}) => {
    if (!imageFile) {
      setError('Please upload an image first.')
      return
    }
    if (!message.trim()) {
      setError('Please enter a message.')
      return
    }

    const userMsg = { role: 'user', content: message.trim(), ts: Date.now() }
    setChatHistory(prev => [...prev, userMsg])

    setLoading(true)
    setError(null)

    try {
      const data = await chat(imageFile, message.trim(), opts)
      setResult(data)
      const assistantMsg = {
        role: 'assistant',
        content: data.assistant,
        detections: data.detections,
        points: data.points,
        inference_time_ms: data.inference_time_ms,
        ts: Date.now(),
      }
      setChatHistory(prev => [...prev, assistantMsg])
    } catch (err) {
      const errMsg = extractError(err)
      setError(errMsg)
      setChatHistory(prev => [
        ...prev,
        { role: 'assistant', content: `Error: ${errMsg}`, isError: true, ts: Date.now() },
      ])
    } finally {
      setLoading(false)
    }
  }, [imageFile])

  const clearChat = useCallback(() => {
    setChatHistory([])
    setResult(null)
    setError(null)
  }, [])

  return {
    imageFile,
    previewUrl,
    result,
    chatHistory,
    loading,
    error,
    selectImage,
    clearImage,
    runDetect,
    runChat,
    clearChat,
  }
}
