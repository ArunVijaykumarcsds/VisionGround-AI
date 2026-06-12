/**
 * src/components/ImageUpload.jsx
 * ================================
 * Drag-and-drop image upload zone.
 * Uses react-dropzone for cross-browser file handling.
 */

import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/webp': [], 'image/bmp': [] }
const MAX_SIZE = 20 * 1024 * 1024 // 20 MB

export default function ImageUpload({ onSelect, disabled }) {
  const onDrop = useCallback(
    (accepted) => { if (accepted.length > 0) onSelect(accepted[0]) },
    [onSelect]
  )

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE,
    maxFiles: 1,
    disabled,
  })

  const rejection = fileRejections[0]?.errors[0]?.message

  return (
    <div className="space-y-2">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-10
          flex flex-col items-center justify-center gap-3
          cursor-pointer transition-all duration-200 select-none
          ${isDragActive
            ? 'border-nvidia-green bg-nvidia-green/10'
            : 'border-gray-600 hover:border-gray-400 bg-gray-800/40 hover:bg-gray-800/70'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {/* Upload icon */}
        <svg
          className={`w-10 h-10 transition-colors ${isDragActive ? 'text-nvidia-green' : 'text-gray-500'}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>

        <div className="text-center">
          <p className="text-sm font-medium text-gray-300">
            {isDragActive ? 'Drop image here' : 'Drag & drop an image'}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            JPEG, PNG, WebP, BMP — max 20 MB
          </p>
        </div>

        <button
          type="button"
          className="btn-secondary text-sm pointer-events-none"
        >
          Browse files
        </button>
      </div>

      {rejection && (
        <p className="text-xs text-red-400 px-1">{rejection}</p>
      )}
    </div>
  )
}
