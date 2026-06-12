# =============================================================
# Locate Anything Assistant — Frontend Dockerfile
# Stage 1: build  (Node 20, Vite)
# Stage 2: serve  (Nginx Alpine)
# =============================================================

# ── Stage 1: Build ──────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests first for layer cache efficiency.
COPY package.json ./

# Install all dependencies (including devDependencies needed for build).
RUN npm install

# Copy source.
COPY . .

# Set the API base URL so the React app points at the backend.
# Override with --build-arg VITE_API_URL=https://your-backend.example.com
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

# Build the production bundle.
RUN npm run build

# ── Stage 2: Serve ──────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove default Nginx content.
RUN rm -rf /usr/share/nginx/html/*

# Copy Vite build output.
COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config: serve SPA with history-mode fallback.
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
