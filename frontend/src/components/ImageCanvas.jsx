/**
 * src/components/ImageCanvas.jsx
 * ================================
 * Renders the uploaded image with detection overlays.
 *
 * - Draws bounding boxes scaled to the displayed image size.
 * - Draws point markers.
 * - Each box gets a distinct colour from a deterministic palette.
 * - Hovering a box highlights it and shows its label + confidence.
 */

import React, { useRef, useEffect, useState, useCallback } from 'react'

// 12-colour palette for distinct box colours.
const PALETTE = [
  '#76b900', '#3b82f6', '#f59e0b', '#ef4444',
  '#8b5cf6', '#06b6d4', '#f97316', '#10b981',
  '#ec4899', '#6366f1', '#84cc16', '#14b8a6',
]

function colourForIndex(i) {
  return PALETTE[i % PALETTE.length]
}

export default function ImageCanvas({ previewUrl, result }) {
  const imgRef = useRef(null)
  const canvasRef = useRef(null)
  const [hoveredIdx, setHoveredIdx] = useState(null)
  const [imgNaturalSize, setImgNaturalSize] = useState({ w: 1, h: 1 })
  const [displaySize, setDisplaySize] = useState({ w: 1, h: 1 })

  // Recompute display size when image loads or window resizes.
  const syncSize = useCallback(() => {
    const img = imgRef.current
    if (!img) return
    setImgNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
    setDisplaySize({ w: img.clientWidth, h: img.clientHeight })
  }, [])

  useEffect(() => {
    window.addEventListener('resize', syncSize)
    return () => window.removeEventListener('resize', syncSize)
  }, [syncSize])

  // Redraw canvas whenever result, hover state, or size changes.
  useEffect(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !result) return

    const dw = img.clientWidth
    const dh = img.clientHeight
    canvas.width  = dw
    canvas.height = dh

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, dw, dh)

    const scaleX = dw / result.image_width
    const scaleY = dh / result.image_height

    // Draw bounding boxes.
    result.detections?.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox
      const sx1 = x1 * scaleX
      const sy1 = y1 * scaleY
      const sw  = (x2 - x1) * scaleX
      const sh  = (y2 - y1) * scaleY

      const colour = colourForIndex(idx)
      const isHovered = hoveredIdx === idx
      const alpha = isHovered ? 0.35 : 0.15

      // Fill.
      ctx.fillStyle = colour + Math.round(alpha * 255).toString(16).padStart(2, '0')
      ctx.fillRect(sx1, sy1, sw, sh)

      // Border.
      ctx.strokeStyle = colour
      ctx.lineWidth = isHovered ? 3 : 2
      ctx.strokeRect(sx1, sy1, sw, sh)

      // Label pill.
      const label = det.label.length > 28 ? det.label.slice(0, 25) + '…' : det.label
      const conf = `${(det.confidence * 100).toFixed(0)}%`
      const text = `${label}  ${conf}`
      ctx.font = `${isHovered ? 600 : 500} 12px Inter, sans-serif`
      const textW = ctx.measureText(text).width
      const pillH = 20
      const pillY = sy1 - pillH - 2 < 0 ? sy1 + 2 : sy1 - pillH - 2

      ctx.fillStyle = colour
      ctx.beginPath()
      ctx.roundRect(sx1, pillY, textW + 12, pillH, 4)
      ctx.fill()
      ctx.fillStyle = '#000'
      ctx.fillText(text, sx1 + 6, pillY + 14)
    })

    // Draw point markers.
    result.points?.forEach((pt, idx) => {
      const px = pt.x * scaleX
      const py = pt.y * scaleY
      const colour = colourForIndex((result.detections?.length || 0) + idx)

      ctx.beginPath()
      ctx.arc(px, py, 6, 0, Math.PI * 2)
      ctx.fillStyle = colour
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()
    })
  }, [result, hoveredIdx, displaySize])

  // Hit-test: which box is the cursor over?
  const handleMouseMove = useCallback((e) => {
    if (!result?.detections?.length) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const scaleX = canvas.width  / result.image_width
    const scaleY = canvas.height / result.image_height

    let found = null
    result.detections.forEach((det, idx) => {
      const [x1, y1, x2, y2] = det.bbox
      if (mx >= x1 * scaleX && mx <= x2 * scaleX &&
          my >= y1 * scaleY && my <= y2 * scaleY) {
        found = idx
      }
    })
    setHoveredIdx(found)
  }, [result])

  const handleMouseLeave = useCallback(() => setHoveredIdx(null), [])

  if (!previewUrl) return null

  return (
    <div className="relative w-full rounded-xl overflow-hidden bg-gray-950 border border-gray-700/50">
      <img
        ref={imgRef}
        src={previewUrl}
        alt="Uploaded"
        className="w-full block"
        onLoad={syncSize}
        draggable={false}
      />
      {result && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full"
          style={{ cursor: hoveredIdx !== null ? 'crosshair' : 'default' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
      )}
      {/* Detection count badge */}
      {result && (result.detections?.length > 0 || result.points?.length > 0) && (
        <div className="absolute top-2 right-2 bg-black/70 text-nvidia-green text-xs font-mono px-2.5 py-1 rounded-full border border-nvidia-green/30">
          {result.detections?.length > 0 && `${result.detections.length} box${result.detections.length !== 1 ? 'es' : ''}`}
          {result.detections?.length > 0 && result.points?.length > 0 && '  '}
          {result.points?.length > 0 && `${result.points.length} pt${result.points.length !== 1 ? 's' : ''}`}
        </div>
      )}
    </div>
  )
}
