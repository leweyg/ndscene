FROM node:20-alpine

WORKDIR /app

COPY server/frontend/package.json /app/server/frontend/package.json
COPY server/frontend/tsconfig.json /app/server/frontend/tsconfig.json
COPY server/frontend/vite.config.ts /app/server/frontend/vite.config.ts
COPY server/frontend/index.html /app/server/frontend/index.html
COPY server/frontend/src /app/server/frontend/src
COPY ts /app/ts

WORKDIR /app/server/frontend
RUN npm install

ENV VITE_API_BASE_URL=http://localhost:8000/api

EXPOSE 4173

CMD ["npm", "run", "dev"]
