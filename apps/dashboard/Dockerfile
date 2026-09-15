# Placeholder Dockerfile — replace once the dashboard framework is chosen.
FROM node:20-slim

WORKDIR /app

COPY apps/dashboard/package*.json ./
RUN npm install || true

COPY apps/dashboard .

EXPOSE 3000

CMD ["npm", "run", "dev"]
