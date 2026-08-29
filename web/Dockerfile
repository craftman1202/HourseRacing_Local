# Cloud Run 用イメージ。ビルドコンテキストは web/ 自身。
#   docker build -t <image> web
# または: gcloud builds submit --tag <image> web

FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# next.config.ts の output: "standalone" が .next/standalone を生成する
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
# Cloud Run が $PORT を渡してくる。standalone サーバーは process.env.PORT を読む
ENV PORT=8080
EXPOSE 8080

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

CMD ["node", "server.js"]
