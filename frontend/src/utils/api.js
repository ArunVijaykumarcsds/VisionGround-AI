/**
 * src/utils/api.js
 * ================
 * Axios-based API client for the FastAPI backend.
 *
 * All endpoints use multipart/form-data because image uploads
 * require FormData. JSON responses are automatically parsed.
 */

import axios from 'axios'

// Base URL: Vite dev server proxies /detect, /chat, /health to localhost:8000.
// In production (Docker / deployed), set VITE_API_URL in your .env.
const BASE_URL = import.meta.env.VITE_API_URL || ''

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2-minute timeout for large images / slow GPU
})

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

/**
 * GET /health
 * @returns {{ status, model_loaded, device, model_path, memory_info }}
 */
export async function getHealth() {
  const { data } = await client.get('/health')
  return data
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * POST /detect
 * @param {File}   imageFile      - Image file from the file picker
 * @param {string} query          - Natural-language detection query
 * @param {object} [opts]         - Optional overrides
 * @param {string} [opts.generation_mode]
 * @param {number} [opts.max_new_tokens]
 * @returns {Promise<DetectResponse>}
 */
export async function detect(imageFile, query, opts = {}) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('query', query)
  if (opts.generation_mode) form.append('generation_mode', opts.generation_mode)
  if (opts.max_new_tokens)  form.append('max_new_tokens', String(opts.max_new_tokens))

  const { data } = await client.post('/detect', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

/**
 * POST /chat
 * @param {File}   imageFile - Image file
 * @param {string} message   - User message
 * @param {object} [opts]
 * @param {string} [opts.generation_mode]
 * @returns {Promise<ChatResponse>}
 */
export async function chat(imageFile, message, opts = {}) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('message', message)
  if (opts.generation_mode) form.append('generation_mode', opts.generation_mode)

  const { data } = await client.post('/chat', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

/**
 * Extract a human-readable error message from an Axios error.
 * @param {unknown} err
 * @returns {string}
 */
export function extractError(err) {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (detail?.message) return detail.message
    if (err.response?.status === 503) return 'Model is still loading. Please wait and try again.'
    if (err.response?.status === 413) return 'Image file is too large. Maximum size is 20 MB.'
    if (err.code === 'ECONNABORTED') return 'Request timed out. The model may still be processing.'
    return err.message
  }
  return String(err)
}
