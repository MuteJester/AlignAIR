import { clsx, type ClassValue } from "clsx";

/** Tailwind-friendly className joiner. */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

/* --------------------------------------------------------------------------
 * Lesson progress, persisted in localStorage (no backend). Keyed by lesson id.
 * ------------------------------------------------------------------------ */
const PROGRESS_KEY = "aa-progress-v1";

type ProgressMap = Record<string, { completed: boolean; step: number }>;

function readProgress(): ProgressMap {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_KEY) ?? "{}") as ProgressMap;
  } catch {
    return {};
  }
}

function writeProgress(map: ProgressMap): void {
  try {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(map));
  } catch {
    /* storage unavailable (private mode) - progress is simply not persisted */
  }
}

export function getLessonProgress(id: string): { completed: boolean; step: number } {
  return readProgress()[id] ?? { completed: false, step: 0 };
}

export function saveLessonProgress(id: string, step: number, completed: boolean): void {
  const map = readProgress();
  const prev = map[id] ?? { completed: false, step: 0 };
  map[id] = { completed: prev.completed || completed, step: Math.max(prev.step, step) };
  writeProgress(map);
}

export function isLessonComplete(id: string): boolean {
  return getLessonProgress(id).completed;
}

/** Theme (light/dark) toggle, synced with the pre-paint script in index.html. */
export function getTheme(): "light" | "dark" {
  return document.documentElement.classList.contains("dark") ? "dark" : "light";
}

export function setTheme(theme: "light" | "dark"): void {
  document.documentElement.classList.toggle("dark", theme === "dark");
  try {
    localStorage.setItem("aa-theme", theme);
  } catch {
    /* ignore */
  }
}
