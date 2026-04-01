import { useEffect, useMemo, useState } from "react";
import { NdSceneTree } from "@shared/ndsceneTree";

type CommitRow = {
  scene_commit_id: string;
  scene_id: string;
  packet_format: string | null;
  packet_path: string | null;
  is_external: number;
  parent_commit_id: string | null;
  commit_json: Record<string, unknown> | null;
  created_at: string;
  object_count: number;
  tensor_count: number;
  data_count: number;
};

type CommitDetail = {
  commit: Record<string, unknown>;
  objects: Array<Record<string, unknown>>;
  tensors: Array<Record<string, unknown>>;
  data: Array<Record<string, unknown>>;
  labels: Array<Record<string, unknown>>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function formatDate(isoText: string): string {
  return new Date(isoText).toLocaleString();
}

export default function App() {
  const [commits, setCommits] = useState<CommitRow[]>([]);
  const [selectedCommitId, setSelectedCommitId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<CommitDetail | null>(null);
  const [statusText, setStatusText] = useState("Loading commit timeline...");
  const [loading, setLoading] = useState(false);

  async function loadCommits(preferredCommitId?: string) {
    setLoading(true);
    try {
      const payload = await fetchJson<{ commits: CommitRow[] }>("/commits");
      setCommits(payload.commits);
      const nextCommitId = preferredCommitId ?? payload.commits[0]?.scene_commit_id ?? null;
      setSelectedCommitId(nextCommitId);
      setStatusText(`Loaded ${payload.commits.length} commits from the backend.`);
    } catch (error) {
      setStatusText(`Unable to load commits: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function loadCommitDetail(sceneCommitId: string) {
    setLoading(true);
    try {
      const payload = await fetchJson<{ detail: CommitDetail }>(`/commits/${sceneCommitId}`);
      setSelectedDetail(payload.detail);
      setStatusText(`Viewing commit ${sceneCommitId}.`);
    } catch (error) {
      setStatusText(`Unable to load commit detail: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function runRender() {
    setLoading(true);
    try {
      const payload = await fetchJson<{ afterCommitId: string; imageChanged: boolean }>("/render/freed-go/view-1", {
        method: "POST",
      });
      await loadCommits(payload.afterCommitId);
      setStatusText(
        payload.imageChanged
          ? `Rendered Freed Go view and added commit ${payload.afterCommitId}.`
          : `Render completed without image change for ${payload.afterCommitId}.`,
      );
    } catch (error) {
      setStatusText(`Unable to run render: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadCommits();
  }, []);

  useEffect(() => {
    if (selectedCommitId) {
      void loadCommitDetail(selectedCommitId);
    }
  }, [selectedCommitId]);

  const selectedCommit = useMemo(
    () => commits.find((commit) => commit.scene_commit_id === selectedCommitId) ?? null,
    [commits, selectedCommitId],
  );

  const treeValue = useMemo(() => {
    if (!selectedDetail) {
      return null;
    }
    return {
      commit: selectedDetail.commit,
      objects: selectedDetail.objects,
      tensors: selectedDetail.tensors,
      data: selectedDetail.data,
      labels: selectedDetail.labels,
    };
  }, [selectedDetail]);

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">ndscene review lab</p>
          <h1>Commit Timeline And Scene Inspection</h1>
          <p className="hero-copy">
            This view reads the SQLite backend, shows a commit timeline, and expands commit content with the shared
            ndscene tree renderer.
          </p>
        </div>
        <div className="hero-actions">
          <button className="action-button" onClick={() => void loadCommits()} disabled={loading}>
            Refresh commits
          </button>
          <button className="action-button action-button-secondary" onClick={() => void runRender()} disabled={loading}>
            Run Freed Go render
          </button>
        </div>
      </header>

      <div className="status-banner">{statusText}</div>

      <main className="layout">
        <section className="timeline-panel">
          <div className="panel-header">
            <h2>Commit Timeline</h2>
            <span>{commits.length} commits</span>
          </div>
          <div className="timeline-list">
            {commits.map((commit) => {
              const stage = typeof commit.commit_json?.stage === "string" ? commit.commit_json.stage : "commit";
              return (
                <button
                  key={commit.scene_commit_id}
                  className="timeline-item"
                  data-selected={commit.scene_commit_id === selectedCommitId ? "true" : "false"}
                  onClick={() => setSelectedCommitId(commit.scene_commit_id)}
                >
                  <div className="timeline-stage">{stage}</div>
                  <div className="timeline-scene">{commit.scene_id}</div>
                  <div className="timeline-meta">{formatDate(commit.created_at)}</div>
                  <div className="timeline-counters">
                    <span>{commit.object_count} objects</span>
                    <span>{commit.tensor_count} tensors</span>
                    <span>{commit.data_count} data</span>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        <section className="detail-panel">
          <div className="panel-header">
            <h2>Commit Detail</h2>
            <span>{selectedCommit ? selectedCommit.scene_commit_id : "No commit selected"}</span>
          </div>
          {selectedCommit ? (
            <div className="detail-meta">
              <span className="meta-chip">{selectedCommit.scene_id}</span>
              <span className="meta-chip">{selectedCommit.packet_format ?? "no packet format"}</span>
              <span className="meta-chip">{selectedCommit.packet_path ?? "no packet path"}</span>
            </div>
          ) : null}
          <div className="tree-panel">
            {treeValue ? <NdSceneTree value={treeValue} rootLabel="commit_detail" /> : <p>No commit detail loaded.</p>}
          </div>
        </section>
      </main>
    </div>
  );
}
