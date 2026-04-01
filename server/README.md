# ndscene server

Two services are defined in `docker-compose.yml`:

- `backend`: FastAPI service exposing the SQLite schema, commit timeline, commit detail, and the Freed Go render/update flow.
- `frontend`: Vite + React review UI that reads commit data from the backend.

## Run

```bash
cd server
docker compose up --build
```

Then open:

- Frontend: `http://localhost:4173`
- Backend API: `http://localhost:8000/api/health`

## Important endpoints

- `GET /api/schema.sql`
- `GET /api/commits`
- `GET /api/commits/{scene_commit_id}`
- `POST /api/render/freed-go/view-1`

The backend reads `py/examples/freed_go.sqlite` by default via `NDSCENE_DB_PATH`.
