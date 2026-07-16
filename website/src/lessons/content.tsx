import type { Lesson, Track } from "./types";
import { foundationsTrack } from "./tracks/foundations";
import { predictingTrack } from "./tracks/predicting";
import { trainingTrack } from "./tracks/training";
import { benchmarkingTrack } from "./tracks/benchmarking";
import { integratingTrack } from "./tracks/integrating";

export const tracks: Track[] = [
  foundationsTrack,
  predictingTrack,
  trainingTrack,
  benchmarkingTrack,
  integratingTrack,
];

const allLessons: Lesson[] = tracks.flatMap((t) => t.lessons);

export function getTrack(slug: string): Track | undefined {
  return tracks.find((t) => t.slug === slug);
}

export function getLesson(track: string, slug: string): Lesson | undefined {
  return allLessons.find((l) => l.track === track && l.slug === slug);
}

export function nextLessonOf(id: string): Lesson | undefined {
  const i = allLessons.findIndex((l) => l.id === id);
  return i >= 0 && i + 1 < allLessons.length ? allLessons[i + 1] : undefined;
}

export function totalLessonCount(): number {
  return allLessons.length;
}
